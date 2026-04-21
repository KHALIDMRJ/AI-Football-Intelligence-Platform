"""
Players endpoints (v1).

Two surfaces live side-by-side under ``/players``:

* Pipeline / VAEP analytics (``/``, ``/rankings``, ``/{player_id}``) —
  parquet-backed, served by ``PlayerService``. Pre-existing.
* DB catalogue (``/catalog``, ``/catalog/{id}``, ``/catalog/{id}/stats``)
  — Postgres/SQLite-backed, serves fan-facing listings and the per-player
  ``PlayerStats`` history. Added in Phase 4.

Split by sub-path rather than router so Swagger keeps them grouped under
the same "players" tag and the analytics routes continue to work as-is.

RBAC
----
Public: ``/catalog`` list + ``/catalog/{id}`` detail — fan-facing discovery.
pro_analyst+: ``/catalog/{id}/stats`` — per-match stat history is premium
intelligence (scouting layer).
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.api.dependencies import get_db, require_role
from football_ai.api.schemas.catalog import (
    PagedResponse,
    PlayerCatalogOut,
    PlayerStatsOut,
)
from football_ai.api.schemas.predictions import ScoutingReportOut
from football_ai.api.schemas.responses import (
    PlayerDetailResponse,
    PlayerListResponse,
    PlayerRankingResponse,
)
from football_ai.api.services.player_service import PlayerService
from football_ai.cache import cached
from football_ai.core.exceptions import DataNotFoundError
from football_ai.crud import player as player_crud
from football_ai.ml.services.scouting_service import ScoutingService
from football_ai.models.user import UserRole

# Process-wide singleton — keeps the Anthropic client warm across requests.
_scouting_service = ScoutingService()


def get_scouting_service() -> ScoutingService:
    """FastAPI dep — overridden in tests with a stub client."""
    return _scouting_service

router = APIRouter(prefix="/players", tags=["players"])


def _player_service() -> PlayerService:
    return PlayerService()


@router.get(
    "",
    response_model=PlayerListResponse,
    summary="List players",
    description=(
        "Returns all players from a specific match (or the first available "
        "match if no match_id is given), optionally filtered by team."
    ),
)
async def list_players(
    match_id: str | None = Query(
        None,
        description="Filter to a specific match. If omitted, all matches are combined.",
    ),
    team_id: str | None = Query(
        None,
        description="Filter to a specific team ID.",
    ),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of players."),
    service: PlayerService = Depends(_player_service),
) -> PlayerListResponse:
    try:
        players, used_match_id = service.get_players(
            match_id=match_id, team_id=team_id, limit=limit
        )
        return PlayerListResponse(
            count=len(players),
            match_id=used_match_id,
            players=players,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/rankings",
    response_model=PlayerRankingResponse,
    summary="Player rankings",
    description=(
        "Returns players ranked by a chosen VAEP metric. "
        "Available metrics: vaep_total, vaep_offensive, vaep_defensive, "
        "vaep_per_90, xg_total, xt_total."
    ),
    responses={
        400: {"description": "Invalid metric parameter."},
        404: {"description": "No data found."},
    },
)
async def player_rankings(
    match_id: str | None = Query(None, description="Restrict to a specific match."),
    metric: str = Query(
        "vaep_total",
        description=(
            "Ranking metric. One of: vaep_total, vaep_offensive, "
            "vaep_defensive, vaep_per_90, xg_total, xt_total."
        ),
    ),
    limit: int = Query(20, ge=1, le=100),
    service: PlayerService = Depends(_player_service),
) -> PlayerRankingResponse:
    try:
        players, used_match_id = service.get_rankings(
            match_id=match_id, metric=metric, limit=limit
        )
        return PlayerRankingResponse(
            metric=metric,
            match_id=used_match_id,
            count=len(players),
            players=players,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


# ── Phase 4: DB-backed catalogue ─────────────────────────────────────────────
#
# IMPORTANT: registered before ``/{player_id}`` so FastAPI doesn't match
# ``/catalog`` as a pipeline player-id string.

@router.get(
    "/catalog",
    response_model=PagedResponse[PlayerCatalogOut],
    summary="List players from the platform catalogue (DB)",
    description=(
        "Paged list of players stored in the core DB (as opposed to the "
        "pipeline-derived VAEP list above). Filters compose — e.g. "
        "`league=Premier%20League&position=ST&min_age=18&max_age=24`."
    ),
)
@cached(ttl=60, namespace="player_catalog", response_model=PagedResponse[PlayerCatalogOut])
async def list_player_catalog(
    db: Annotated[AsyncSession, Depends(get_db)],
    league: str | None = Query(None, description="Match team league name."),
    position: str | None = Query(None, description="Free-text position code (e.g. ST, CB)."),
    team_id: uuid.UUID | None = Query(None, description="Filter to this team's squad."),
    min_age: int | None = Query(None, ge=14, le=60),
    max_age: int | None = Query(None, ge=14, le=60),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> PagedResponse[PlayerCatalogOut]:
    rows, total = await player_crud.list_players(
        db,
        league=league,
        position=position,
        team_id=team_id,
        min_age=min_age,
        max_age=max_age,
        limit=limit,
        offset=offset,
    )
    return PagedResponse[PlayerCatalogOut](
        total=total,
        limit=limit,
        offset=offset,
        items=[PlayerCatalogOut.model_validate(p) for p in rows],
    )


@router.get(
    "/catalog/{player_id}",
    response_model=PlayerCatalogOut,
    summary="Player detail from the catalogue (DB)",
    responses={404: {"description": "Player not found in the catalogue."}},
)
async def get_player_catalog_entry(
    player_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PlayerCatalogOut:
    row = await player_crud.get_player(db, player_id)
    if row is None:
        raise DataNotFoundError(f"Player {player_id} not found.")
    return PlayerCatalogOut.model_validate(row)


@router.get(
    "/catalog/{player_id}/stats",
    response_model=list[PlayerStatsOut],
    summary="Player statistics history (pro_analyst+)",
    description=(
        "Returns every ``player_stats`` row for this player. Match lines "
        "sort ahead of season aggregates; pass ``season=2024-25`` to slice."
    ),
    dependencies=[Depends(require_role(UserRole.pro_analyst))],
)
async def get_player_stats(
    player_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    season: str | None = Query(None, description="Season code, e.g. 2024-25."),
    match_id: uuid.UUID | None = Query(None, description="Single-match filter."),
) -> list[PlayerStatsOut]:
    # 404 early so a gated endpoint doesn't return an empty list for a
    # non-existent player — that would mask typos in client code.
    if (await player_crud.get_player(db, player_id)) is None:
        raise DataNotFoundError(f"Player {player_id} not found.")
    rows = await player_crud.list_player_stats(
        db, player_id, season=season, match_id=match_id
    )
    return [PlayerStatsOut.model_validate(r) for r in rows]


@router.get(
    "/catalog/{player_id}/scouting-report",
    response_model=ScoutingReportOut,
    summary="LLM-generated scouting report (club_scout+)",
    description=(
        "Generates a 200–300 word scouting report grounded in the player's "
        "stats. Returns 503 when ANTHROPIC_API_KEY is not configured — the "
        "feature is opt-in and shouldn't fail catastrophically without a key."
    ),
    dependencies=[Depends(require_role(UserRole.club_scout))],
    responses={
        404: {"description": "Player not found."},
        502: {"description": "LLM call failed (network/auth/rate-limit)."},
        503: {"description": "Scouting feature not configured."},
    },
)
async def get_player_scouting_report(
    player_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    service: Annotated[ScoutingService, Depends(get_scouting_service)],
) -> ScoutingReportOut:
    player = await player_crud.get_player(db, player_id)
    if player is None:
        raise DataNotFoundError(f"Player {player_id} not found.")
    stats = await player_crud.list_player_stats(db, player_id)

    # Anthropic client is sync — run in a thread to keep the loop free.
    report = await run_in_threadpool(service.generate, player, stats)
    return ScoutingReportOut(
        player_id=uuid.UUID(report.player_id),
        prose=report.prose,
        model=report.model,
        generated_at=report.generated_at,
        token_count=report.token_count,
    )


# ── Pipeline / VAEP analytics (existing) ─────────────────────────────────────

@router.get(
    "/{player_id}",
    response_model=PlayerDetailResponse,
    summary="Player profile",
    description="Returns the full VAEP profile for a single player.",
    responses={
        404: {"description": "Player not found."},
    },
)
async def get_player(
    player_id: str,
    match_id: str | None = Query(None, description="Match context."),
    service: PlayerService = Depends(_player_service),
) -> PlayerDetailResponse:
    try:
        return service.get_player(player_id=player_id, match_id=match_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
