"""
Team CRUD — registry list, detail, squad view.

``list_team_squad`` leans on the ``ix_players_current_team_id`` index; it
only returns active (non-soft-deleted) players and sorts by jersey number
so the UI can render a classic roster.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.models.player import Player
from football_ai.models.team import Team


def _apply_team_filters(
    stmt: Select,
    *,
    country: str | None,
    league: str | None,
) -> Select:
    stmt = stmt.where(Team.is_deleted.is_(False))
    if country is not None:
        stmt = stmt.where(Team.country == country)
    if league is not None:
        stmt = stmt.where(Team.league == league)
    return stmt


async def list_teams(
    db: AsyncSession,
    *,
    country: str | None = None,
    league: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Team], int]:
    filtered = _apply_team_filters(
        select(Team), country=country, league=league
    )
    count_stmt = _apply_team_filters(
        select(func.count(Team.id)), country=country, league=league
    )
    total = (await db.execute(count_stmt)).scalar_one()

    rows = (
        await db.execute(
            filtered.order_by(Team.name).limit(limit).offset(offset)
        )
    ).scalars().all()
    return list(rows), int(total)


async def get_team(db: AsyncSession, team_id: uuid.UUID) -> Team | None:
    stmt = select(Team).where(
        Team.id == team_id, Team.is_deleted.is_(False)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_team_squad(
    db: AsyncSession, team_id: uuid.UUID
) -> list[Player]:
    """Active squad for this team — ordered by jersey number, nulls last."""
    stmt = (
        select(Player)
        .where(
            Player.current_team_id == team_id,
            Player.is_deleted.is_(False),
        )
        .order_by(
            Player.jersey_number.is_(None),  # False < True → numbered first
            Player.jersey_number.asc(),
            Player.name.asc(),
        )
    )
    return list((await db.execute(stmt)).scalars().all())
