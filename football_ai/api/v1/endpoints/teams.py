"""
Teams endpoints (v1).

Two surfaces under ``/teams``:

* Pipeline / tactical analytics (``/{match_id}``, ``/{match_id}/{team_id}``)
  — VAEP + formation output via ``TacticalService``. Pre-existing.
* DB registry (``/registry``, ``/registry/{id}``, ``/registry/{id}/squad``)
  — team catalogue + current squad. Added in Phase 4.

Public reads only — writes will arrive in Phase 7 (API-Football sync).
``/registry`` is declared before the catch-all pipeline routes so
``registry`` is never mis-parsed as a match_id.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.api.dependencies import get_db
from football_ai.api.schemas.catalog import (
    PagedResponse,
    SquadMemberOut,
    TeamRegistryOut,
)
from football_ai.api.schemas.responses import TeamDetailResponse, TeamListResponse
from football_ai.api.services.tactical_service import TacticalService
from football_ai.core.exceptions import DataNotFoundError
from football_ai.crud import team as team_crud

router = APIRouter(prefix="/teams", tags=["teams"])


def _tactical_service() -> TacticalService:
    return TacticalService()


# ── Phase 4: DB-backed registry ──────────────────────────────────────────────
#
# Declared before the pipeline catch-alls so ``/registry`` isn't parsed as a
# match_id.

@router.get(
    "/registry",
    response_model=PagedResponse[TeamRegistryOut],
    summary="List teams (DB)",
    description="Paged list of teams in the core DB with country/league filters.",
)
async def list_team_registry(
    db: Annotated[AsyncSession, Depends(get_db)],
    country: str | None = Query(None),
    league: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> PagedResponse[TeamRegistryOut]:
    rows, total = await team_crud.list_teams(
        db, country=country, league=league, limit=limit, offset=offset
    )
    return PagedResponse[TeamRegistryOut](
        total=total,
        limit=limit,
        offset=offset,
        items=[TeamRegistryOut.model_validate(t) for t in rows],
    )


@router.get(
    "/registry/{team_id}",
    response_model=TeamRegistryOut,
    summary="Team detail (DB)",
    responses={404: {"description": "Team not found."}},
)
async def get_team_registry_entry(
    team_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TeamRegistryOut:
    row = await team_crud.get_team(db, team_id)
    if row is None:
        raise DataNotFoundError(f"Team {team_id} not found.")
    return TeamRegistryOut.model_validate(row)


@router.get(
    "/registry/{team_id}/squad",
    response_model=list[SquadMemberOut],
    summary="Current squad for a team",
    description=(
        "Every player whose ``current_team_id`` points here. Sorted by "
        "jersey number (unassigned numbers sort last)."
    ),
    responses={404: {"description": "Team not found."}},
)
async def get_team_squad(
    team_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[SquadMemberOut]:
    if (await team_crud.get_team(db, team_id)) is None:
        raise DataNotFoundError(f"Team {team_id} not found.")
    rows = await team_crud.list_team_squad(db, team_id)
    return [SquadMemberOut.model_validate(p) for p in rows]


# ── Pipeline / tactical analytics (existing) ─────────────────────────────────

@router.get(
    "/{match_id}",
    response_model=TeamListResponse,
    summary="List teams in a match",
    description=(
        "Returns both teams present in a match, each with their VAEP totals, "
        "formation, and the most dangerous zones they conceded in."
    ),
    responses={
        404: {"description": "Match not found."},
    },
)
async def list_teams(
    match_id: str,
    service: TacticalService = Depends(_tactical_service),
) -> TeamListResponse:
    try:
        teams = service.get_teams(match_id)
        return TeamListResponse(
            count=len(teams),
            match_id=match_id,
            teams=teams,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/{match_id}/{team_id}",
    response_model=TeamDetailResponse,
    summary="Team profile",
    description=(
        "Returns the full team profile: VAEP totals, detected formation, "
        "number of critical zones, and a ranked list of tactical weaknesses."
    ),
    responses={
        404: {"description": "Team or match not found."},
    },
)
async def get_team(
    match_id: str,
    team_id: str,
    service: TacticalService = Depends(_tactical_service),
) -> TeamDetailResponse:
    try:
        return service.get_team(match_id=match_id, team_id=team_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
