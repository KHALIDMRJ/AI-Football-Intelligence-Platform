"""
Matches endpoints (v1).

Two surfaces under ``/matches``:

* Pipeline / analytics (``/``, ``/{match_id}``, ``/{match_id}/report``) —
  parquet + tactical pipeline output via ``MatchService``. Pre-existing.
* DB fixtures (``/fixtures``, ``/fixtures/{id}``, ``/fixtures/{id}/events``,
  ``/live``) — Postgres/SQLite-backed core schedule. Added in Phase 4.

Route order matters: ``/live`` and ``/fixtures`` are registered before the
``/{match_id}`` catch-all to avoid FastAPI matching them as match IDs.

RBAC
----
Public: ``/fixtures`` list + ``/fixtures/{id}`` detail.
pro_analyst+: ``/live`` (live feed is premium) and ``/fixtures/{id}/events``
(event-level granularity drives the prediction engine — paywalled to
protect the feature moat).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.api.dependencies import get_db, require_role
from football_ai.api.schemas.catalog import (
    MatchEventOut,
    MatchFixtureOut,
    PagedResponse,
)
from football_ai.api.schemas.responses import (
    MatchDetailResponse,
    MatchListResponse,
    MatchReportResponse,
)
from football_ai.api.services.match_service import MatchService
from football_ai.cache import cached
from football_ai.core.exceptions import DataNotFoundError
from football_ai.crud import match as match_crud
from football_ai.models.match import MatchStatus
from football_ai.models.user import UserRole

router = APIRouter(prefix="/matches", tags=["matches"])


def _match_service() -> MatchService:
    return MatchService()


@router.get(
    "",
    response_model=MatchListResponse,
    summary="List all processed matches",
    description=(
        "Returns a compact summary for every match that has been "
        "processed through the full pipeline (Phases 2–7)."
    ),
)
async def list_matches(
    service: MatchService = Depends(_match_service),
) -> MatchListResponse:
    matches = service.get_match_list()
    return MatchListResponse(count=len(matches), matches=matches)


# ── Phase 4: DB-backed fixtures ──────────────────────────────────────────────
#
# IMPORTANT: these routes MUST be declared before ``/{match_id}`` so FastAPI
# doesn't match ``/live`` or ``/fixtures`` as a pipeline match-id string.

@router.get(
    "/fixtures",
    response_model=PagedResponse[MatchFixtureOut],
    summary="List fixtures (DB)",
    description=(
        "Paged list of matches in the core DB, with filters for league, "
        "season, status, involved team, and date window."
    ),
)
@cached(ttl=30, namespace="fixtures", response_model=PagedResponse[MatchFixtureOut])
async def list_fixtures(
    db: Annotated[AsyncSession, Depends(get_db)],
    league: str | None = Query(None),
    season: str | None = Query(None),
    match_status: MatchStatus | None = Query(
        None, description="scheduled | live | finished | postponed | cancelled"
    ),
    team_id: uuid.UUID | None = Query(None, description="Home or away team."),
    from_date: datetime | None = Query(None, description="ISO timestamp, inclusive."),
    to_date: datetime | None = Query(None, description="ISO timestamp, inclusive."),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> PagedResponse[MatchFixtureOut]:
    rows, total = await match_crud.list_matches(
        db,
        league=league,
        season=season,
        status=match_status,
        team_id=team_id,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
        offset=offset,
    )
    return PagedResponse[MatchFixtureOut](
        total=total,
        limit=limit,
        offset=offset,
        items=[MatchFixtureOut.model_validate(m) for m in rows],
    )


@router.get(
    "/live",
    response_model=list[MatchFixtureOut],
    summary="Currently live matches (pro_analyst+)",
    description=(
        "All matches with status=live. Pagination is intentionally absent "
        "— the live slate is bounded (dozens of matches worldwide at peak)."
    ),
    dependencies=[Depends(require_role(UserRole.pro_analyst))],
)
async def list_live_fixtures(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[MatchFixtureOut]:
    rows = await match_crud.list_live_matches(db)
    return [MatchFixtureOut.model_validate(m) for m in rows]


@router.get(
    "/fixtures/{match_id}",
    response_model=MatchFixtureOut,
    summary="Fixture detail (DB)",
    responses={404: {"description": "Fixture not found in the catalogue."}},
)
async def get_fixture(
    match_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MatchFixtureOut:
    row = await match_crud.get_match(db, match_id)
    if row is None:
        raise DataNotFoundError(f"Match {match_id} not found.")
    return MatchFixtureOut.model_validate(row)


@router.get(
    "/fixtures/{match_id}/events",
    response_model=list[MatchEventOut],
    summary="Match event timeline (pro_analyst+)",
    description=(
        "Full chronological ``MatchEvent`` list for a fixture: goals, cards, "
        "subs, VAR. Event-level detail feeds the Phase-5 prediction engine "
        "and is therefore gated to pro_analyst+."
    ),
    dependencies=[Depends(require_role(UserRole.pro_analyst))],
)
async def get_fixture_events(
    match_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[MatchEventOut]:
    if (await match_crud.get_match(db, match_id)) is None:
        raise DataNotFoundError(f"Match {match_id} not found.")
    rows = await match_crud.list_match_events(db, match_id)
    return [MatchEventOut.model_validate(e) for e in rows]


# ── Pipeline / VAEP analytics (existing) ─────────────────────────────────────

@router.get(
    "/{match_id}",
    response_model=MatchDetailResponse,
    summary="Full match analysis",
    description=(
        "Returns xG, VAEP totals, team stats, formations, and the "
        "most valuable actions for a single match."
    ),
    responses={
        404: {"description": "Match not found — run the pipeline first."},
    },
)
async def get_match(
    match_id: str,
    service: MatchService = Depends(_match_service),
) -> MatchDetailResponse:
    try:
        return service.get_match_detail(match_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/{match_id}/report",
    response_model=MatchReportResponse,
    summary="Tactical report JSON",
    description=(
        "Returns the full tactical intelligence JSON report generated "
        "by Phase 7. Includes formations, weaknesses, and player rankings."
    ),
    responses={
        404: {"description": "Tactical report not found — run Phase 7 first."},
    },
)
async def get_match_report(
    match_id: str,
    service: MatchService = Depends(_match_service),
) -> MatchReportResponse:
    try:
        report = service.get_match_report(match_id)
        return MatchReportResponse(match_id=match_id, report=report)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
