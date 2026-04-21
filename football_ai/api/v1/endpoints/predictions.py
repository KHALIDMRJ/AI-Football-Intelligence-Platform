"""
AI match-outcome prediction endpoints (v1) — Phase 5.

Routes
------
GET  /predictions/ping              — RBAC smoke test (Phase 3)
POST /predictions/match/{match_id}  — run/update prediction (pro_analyst+)
GET  /predictions/match/{match_id}  — fetch latest prediction (pro_analyst+)
GET  /predictions/history           — paged history (pro_analyst+)
GET  /predictions/accuracy          — overall + per-league hit rate (pro_analyst+)

The model is loaded into ``MatchPredictor.instance()`` on first POST and
reused for every subsequent call. Re-prediction overwrites the same
``(match_id, model_version)`` row; new training cycles produce a new
``model_version`` and therefore a new row, preserving accuracy history.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.api.dependencies import CurrentUser, get_db, require_role
from football_ai.api.schemas.predictions import (
    AccuracyReport,
    LeagueAccuracy,
    PredictionHistoryPage,
    PredictionOut,
)
from football_ai.cache import cached
from football_ai.core.exceptions import DataNotFoundError
from football_ai.crud import match as match_crud
from football_ai.crud import prediction as prediction_crud
from football_ai.ml.serving.match_predictor import MatchPredictor, predicted_class
from football_ai.models.prediction import AIPrediction
from football_ai.models.user import UserRole
from football_ai.ratelimit import role_rate_limit

router = APIRouter(prefix="/predictions", tags=["predictions"])


def _to_out(row: AIPrediction) -> PredictionOut:
    """Project an AIPrediction ORM row into the response shape."""
    return PredictionOut(
        id=row.id,
        match_id=row.match_id,
        model_version=row.model_version,
        home_win_prob=row.home_win_prob,
        draw_prob=row.draw_prob,
        away_win_prob=row.away_win_prob,
        predicted_outcome=predicted_class(row),
        confidence_score=row.confidence_score,
        key_factors=row.key_factors,
        actual_result=row.actual_result,
        was_correct=row.was_correct,
        created_at=row.created_at,
    )


@router.get(
    "/ping",
    summary="Pro-gate smoke test (pro_analyst+)",
    dependencies=[Depends(require_role(UserRole.pro_analyst))],
)
async def predictions_ping(current: CurrentUser) -> dict[str, str]:
    return {"status": "ok", "role": current.role.value}


@router.post(
    "/match/{match_id}",
    response_model=PredictionOut,
    status_code=status.HTTP_200_OK,
    summary="Run (or refresh) the AI match-outcome prediction",
    dependencies=[
        Depends(require_role(UserRole.pro_analyst)),
        # Role-aware rate limit: running the predictor pins CPU, so we
        # pinch free/pro tiers harder than scout/admin. Values are per
        # minute (token bucket); bursts are smoothed over the window.
        Depends(role_rate_limit(
            free_per_minute=0,       # already blocked by require_role — belt & braces
            pro_per_minute=30,
            scout_per_minute=120,
            admin_per_minute=600,
            namespace="predict_match",
        )),
    ],
    responses={
        404: {"description": "Match not found in the catalogue."},
        409: {"description": "Model not trained — run the training script first."},
        422: {"description": "Inference failed (feature gap, model error)."},
        429: {"description": "Rate limit exceeded for the caller's role."},
    },
)
async def predict_match(
    match_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PredictionOut:
    match = await match_crud.get_match(db, match_id)
    if match is None:
        raise DataNotFoundError(f"Match {match_id} not found.")

    predictor = MatchPredictor.instance()
    row = await predictor.predict(db, match)
    await db.commit()
    await db.refresh(row)
    return _to_out(row)


@router.get(
    "/match/{match_id}",
    response_model=PredictionOut,
    summary="Fetch the most recent stored prediction for a match",
    dependencies=[Depends(require_role(UserRole.pro_analyst))],
    responses={404: {"description": "No prediction yet for this match."}},
)
async def get_prediction(
    match_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    model_version: str | None = Query(
        None, description="Pin a specific model version (default: latest)."
    ),
) -> PredictionOut:
    row = await prediction_crud.get_for_match(
        db, match_id, model_version=model_version
    )
    if row is None:
        raise DataNotFoundError(
            f"No prediction recorded for match {match_id}."
        )
    return _to_out(row)


@router.get(
    "/history",
    response_model=PredictionHistoryPage,
    summary="Paged prediction history (pro_analyst+)",
    dependencies=[Depends(require_role(UserRole.pro_analyst))],
)
async def list_predictions(
    db: Annotated[AsyncSession, Depends(get_db)],
    league: str | None = Query(None),
    model_version: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> PredictionHistoryPage:
    rows, total = await prediction_crud.list_history(
        db,
        model_version=model_version,
        league=league,
        limit=limit,
        offset=offset,
    )
    return PredictionHistoryPage(
        total=total,
        limit=limit,
        offset=offset,
        items=[_to_out(r) for r in rows],
    )


@router.get(
    "/accuracy",
    response_model=AccuracyReport,
    summary="Overall + per-league hit rate (pro_analyst+)",
    description=(
        "Aggregates only over predictions whose ``actual_result`` has been "
        "backfilled — in-flight matches don't pollute the rate."
    ),
    dependencies=[Depends(require_role(UserRole.pro_analyst))],
)
@cached(ttl=60, namespace="accuracy", response_model=AccuracyReport)
async def prediction_accuracy(
    db: Annotated[AsyncSession, Depends(get_db)],
    model_version: str | None = Query(None),
) -> AccuracyReport:
    total, correct, per_league = await prediction_crud.accuracy_overall_and_by_league(
        db, model_version=model_version
    )
    return AccuracyReport(
        total_scored=total,
        total_correct=correct,
        overall_accuracy=(correct / total) if total else 0.0,
        by_league=[
            LeagueAccuracy(
                league=league,
                total=t,
                correct=c,
                accuracy=(c / t) if t else 0.0,
            )
            for league, t, c in per_league
        ],
        model_version=model_version,
    )
