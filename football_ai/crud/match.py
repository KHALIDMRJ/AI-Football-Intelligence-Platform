"""
Match CRUD — fixture listing, detail, event timeline, live feed.

Handlers call these with pre-parsed query params. Filters compose; the
``_apply_match_filters`` helper is shared by list+count so totals never
drift from the paginated page.

Soft-deleted matches are hidden (``is_deleted = False``). Status-based
live-match lookup is its own helper because the query always picks up
the ``ix_matches_date_status`` index.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Select, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from football_ai.models.match import Match, MatchEvent, MatchStatus


def _apply_match_filters(
    stmt: Select,
    *,
    league: str | None,
    season: str | None,
    status: MatchStatus | None,
    team_id: uuid.UUID | None,
    from_date: datetime | None,
    to_date: datetime | None,
) -> Select:
    stmt = stmt.where(Match.is_deleted.is_(False))
    if league is not None:
        stmt = stmt.where(Match.league == league)
    if season is not None:
        stmt = stmt.where(Match.season == season)
    if status is not None:
        stmt = stmt.where(Match.status == status)
    if team_id is not None:
        # One team filter across both sides of the fixture.
        stmt = stmt.where(
            or_(Match.home_team_id == team_id, Match.away_team_id == team_id)
        )
    if from_date is not None:
        stmt = stmt.where(Match.match_date >= from_date)
    if to_date is not None:
        stmt = stmt.where(Match.match_date <= to_date)
    return stmt


async def list_matches(
    db: AsyncSession,
    *,
    league: str | None = None,
    season: str | None = None,
    status: MatchStatus | None = None,
    team_id: uuid.UUID | None = None,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Match], int]:
    base = select(Match).options(
        selectinload(Match.home_team),
        selectinload(Match.away_team),
    )
    filtered = _apply_match_filters(
        base,
        league=league, season=season, status=status,
        team_id=team_id, from_date=from_date, to_date=to_date,
    )

    count_stmt = _apply_match_filters(
        select(func.count(Match.id)),
        league=league, season=season, status=status,
        team_id=team_id, from_date=from_date, to_date=to_date,
    )
    total = (await db.execute(count_stmt)).scalar_one()

    rows = (
        await db.execute(
            filtered.order_by(Match.match_date.desc()).limit(limit).offset(offset)
        )
    ).scalars().all()
    return list(rows), int(total)


async def list_live_matches(db: AsyncSession) -> list[Match]:
    """Every match with status=live — no pagination: a live slate is small."""
    stmt = (
        select(Match)
        .options(
            selectinload(Match.home_team),
            selectinload(Match.away_team),
        )
        .where(
            Match.is_deleted.is_(False),
            Match.status == MatchStatus.live,
        )
        .order_by(Match.match_date.asc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_match(db: AsyncSession, match_id: uuid.UUID) -> Match | None:
    stmt = (
        select(Match)
        .options(
            selectinload(Match.home_team),
            selectinload(Match.away_team),
        )
        .where(Match.id == match_id, Match.is_deleted.is_(False))
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_match_events(
    db: AsyncSession, match_id: uuid.UUID
) -> list[MatchEvent]:
    """Chronological event timeline — leans on ix_match_events_match_minute."""
    stmt = (
        select(MatchEvent)
        .where(MatchEvent.match_id == match_id)
        .order_by(MatchEvent.minute.asc(), MatchEvent.created_at.asc())
    )
    return list((await db.execute(stmt)).scalars().all())
