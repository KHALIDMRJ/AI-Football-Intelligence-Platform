"""
Player CRUD — DB-backed catalogue queries for the Phase-4 core endpoints.

All list helpers return ``(rows, total)`` so the HTTP layer can paginate
without a second COUNT(*) round-trip from the handler — we let SQL do it
via ``func.count`` on the filtered query.

Soft-deleted players are filtered out here (``is_deleted = False``) so
handlers don't need to remember the flag; admin-only endpoints that need
to see deleted rows will call dedicated helpers in Phase 7.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from football_ai.models.player import Player, PlayerStats
from football_ai.models.team import Team


def _apply_player_filters(
    stmt: Select,
    *,
    league: str | None,
    position: str | None,
    team_id: uuid.UUID | None,
    min_age: int | None,
    max_age: int | None,
) -> Select:
    """Shared filter application so list() and count() agree on scope."""
    stmt = stmt.where(Player.is_deleted.is_(False))
    if position is not None:
        stmt = stmt.where(Player.position == position)
    if team_id is not None:
        stmt = stmt.where(Player.current_team_id == team_id)
    if min_age is not None:
        stmt = stmt.where(Player.age >= min_age)
    if max_age is not None:
        stmt = stmt.where(Player.age <= max_age)
    if league is not None:
        # league lives on the team — join via current_team_id
        stmt = stmt.join(Team, Player.current_team_id == Team.id).where(
            Team.league == league
        )
    return stmt


async def list_players(
    db: AsyncSession,
    *,
    league: str | None = None,
    position: str | None = None,
    team_id: uuid.UUID | None = None,
    min_age: int | None = None,
    max_age: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Player], int]:
    base = select(Player).options(selectinload(Player.current_team))
    filtered = _apply_player_filters(
        base,
        league=league,
        position=position,
        team_id=team_id,
        min_age=min_age,
        max_age=max_age,
    )

    count_stmt = _apply_player_filters(
        select(func.count(Player.id)),
        league=league,
        position=position,
        team_id=team_id,
        min_age=min_age,
        max_age=max_age,
    )
    total = (await db.execute(count_stmt)).scalar_one()

    rows = (
        await db.execute(
            filtered.order_by(Player.name).limit(limit).offset(offset)
        )
    ).scalars().all()
    return list(rows), int(total)


async def get_player(
    db: AsyncSession, player_id: uuid.UUID
) -> Player | None:
    stmt = (
        select(Player)
        .options(selectinload(Player.current_team))
        .where(Player.id == player_id, Player.is_deleted.is_(False))
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_player_stats(
    db: AsyncSession,
    player_id: uuid.UUID,
    *,
    season: str | None = None,
    match_id: uuid.UUID | None = None,
    include_season_aggregate: bool = True,
) -> list[PlayerStats]:
    """Stats rows for a player.

    ``match_id`` filter is exclusive: callers asking for a specific match
    line must opt in; by default both per-match lines and (when enabled)
    season aggregates are returned ordered newest-first.
    """
    stmt = select(PlayerStats).where(PlayerStats.player_id == player_id)
    if season is not None:
        stmt = stmt.where(PlayerStats.season == season)
    if match_id is not None:
        stmt = stmt.where(PlayerStats.match_id == match_id)
    elif not include_season_aggregate:
        stmt = stmt.where(PlayerStats.match_id.is_not(None))

    # Season aggregates (match_id IS NULL) sort last so match lines lead.
    stmt = stmt.order_by(
        PlayerStats.season.desc(),
        PlayerStats.match_id.is_(None),  # False < True → match-lines first
        PlayerStats.created_at.desc(),
    )
    return list((await db.execute(stmt)).scalars().all())
