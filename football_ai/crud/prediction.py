"""
Prediction CRUD — stored AIPrediction reads + accuracy aggregation.

The predictor service (``ml.serving.match_predictor``) handles the upsert
on inference; this module is for the read paths (history, accuracy,
backfill of actual_result).
"""

from __future__ import annotations

import uuid

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from football_ai.models.match import Match
from football_ai.models.prediction import AIPrediction, PredictedOutcome


async def get_for_match(
    db: AsyncSession,
    match_id: uuid.UUID,
    *,
    model_version: str | None = None,
) -> AIPrediction | None:
    stmt = select(AIPrediction).where(AIPrediction.match_id == match_id)
    if model_version is not None:
        stmt = stmt.where(AIPrediction.model_version == model_version)
    stmt = stmt.order_by(AIPrediction.created_at.desc()).limit(1)
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_history(
    db: AsyncSession,
    *,
    model_version: str | None = None,
    league: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[AIPrediction], int]:
    stmt = select(AIPrediction).options(selectinload(AIPrediction.match))
    count_stmt = select(func.count(AIPrediction.id))

    if model_version is not None:
        stmt = stmt.where(AIPrediction.model_version == model_version)
        count_stmt = count_stmt.where(AIPrediction.model_version == model_version)
    if league is not None:
        stmt = stmt.join(Match, AIPrediction.match_id == Match.id).where(
            Match.league == league
        )
        count_stmt = count_stmt.join(
            Match, AIPrediction.match_id == Match.id
        ).where(Match.league == league)

    total = (await db.execute(count_stmt)).scalar_one()
    rows = (
        await db.execute(
            stmt.order_by(AIPrediction.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).scalars().all()
    return list(rows), int(total)


async def accuracy_overall_and_by_league(
    db: AsyncSession,
    *,
    model_version: str | None = None,
) -> tuple[int, int, list[tuple[str, int, int]]]:
    """Return (total_scored, total_correct, per_league: [(league, total, correct)])."""
    correct_int = case((AIPrediction.was_correct.is_(True), 1), else_=0)

    base = (
        select(
            Match.league,
            func.count(AIPrediction.id).label("total"),
            func.sum(correct_int).label("correct"),
        )
        .join(Match, AIPrediction.match_id == Match.id)
        .where(AIPrediction.actual_result.is_not(None))
        .group_by(Match.league)
    )
    if model_version is not None:
        base = base.where(AIPrediction.model_version == model_version)

    rows = (await db.execute(base)).all()
    per_league = [(r.league, int(r.total), int(r.correct or 0)) for r in rows]
    total = sum(t for _, t, _ in per_league)
    correct = sum(c for _, _, c in per_league)
    return total, correct, per_league


async def backfill_actual_result(
    db: AsyncSession,
    match_id: uuid.UUID,
    actual: PredictedOutcome,
) -> int:
    """Stamp every prediction for ``match_id`` with the real outcome.

    Returns the number of rows updated. Caller commits.
    """
    rows = (
        await db.execute(
            select(AIPrediction).where(AIPrediction.match_id == match_id)
        )
    ).scalars().all()

    from football_ai.ml.serving.match_predictor import predicted_class

    n = 0
    for r in rows:
        r.actual_result = actual
        r.was_correct = predicted_class(r) == actual
        n += 1
    return n
