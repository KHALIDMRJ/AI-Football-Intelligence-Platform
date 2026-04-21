"""
Fixture sync — pull a league/season schedule from API-Football and
upsert into ``Match``.

Idempotency
-----------
Each fixture's ``api_football_id`` is the natural key. We look up by it,
update if present, insert if not. A re-run two minutes later with
identical upstream data produces zero writes (we early-return when the
materialised payload matches the existing row).

Team resolution
---------------
The provider sends teams by their own integer id. We look up our
``Team`` row by ``api_football_id``; if missing we create a stub Team
(name + api_football_id). A subsequent ``sync_team_squad`` enriches the
stub with country/league/logo. This keeps fixture sync self-sufficient
even on a cold DB.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.external.api_football import APIFootballClient
from football_ai.logger import get_logger
from football_ai.models.match import Match, MatchStatus
from football_ai.models.team import Team

from .base import SyncResult

logger = get_logger(__name__)


# Provider status code → our enum. Every other provider status maps to
# ``scheduled`` because we'd rather under-state than mis-state.
_STATUS_MAP: dict[str, MatchStatus] = {
    "TBD": MatchStatus.scheduled,
    "NS": MatchStatus.scheduled,
    "1H": MatchStatus.live,
    "HT": MatchStatus.live,
    "2H": MatchStatus.live,
    "ET": MatchStatus.live,
    "BT": MatchStatus.live,
    "P": MatchStatus.live,
    "LIVE": MatchStatus.live,
    "FT": MatchStatus.finished,
    "AET": MatchStatus.finished,
    "PEN": MatchStatus.finished,
    "PST": MatchStatus.postponed,
    "CANC": MatchStatus.cancelled,
    "ABD": MatchStatus.cancelled,
    "AWD": MatchStatus.finished,
    "WO": MatchStatus.finished,
}


async def sync_league_fixtures(
    db: AsyncSession,
    client: APIFootballClient,
    *,
    league: int,
    season: int,
) -> SyncResult:
    """Pull every fixture for ``league`` × ``season`` and upsert."""
    result = SyncResult(target=f"fixtures:{league}:{season}")
    fixtures = await client.list_fixtures(league=league, season=season)
    result.fetched = len(fixtures)

    for raw in fixtures:
        try:
            await _upsert_fixture(
                db,
                raw,
                league_name_default=str(league),
                season=str(season),
                result=result,
            )
        except Exception as exc:  # narrow individual failures
            result.add_warning(
                f"fixture {raw.get('fixture', {}).get('id', '?')} skipped: {exc}"
            )
            result.skipped += 1

    await db.commit()
    logger.info(
        "Fixture sync done: %s — created=%d updated=%d skipped=%d warn=%d",
        result.target, result.created, result.updated, result.skipped, len(result.warnings),
    )
    return result


# ── Internals ────────────────────────────────────────────────────────────────

async def _upsert_fixture(
    db: AsyncSession,
    raw: dict[str, Any],
    *,
    league_name_default: str,
    season: str,
    result: SyncResult,
) -> None:
    fixture_block = raw["fixture"]
    teams_block = raw["teams"]
    league_block = raw.get("league", {})
    score_block = raw.get("goals", {})
    venue_block = fixture_block.get("venue", {}) or {}

    api_id: int = fixture_block["id"]
    home = await _ensure_team(db, teams_block["home"])
    away = await _ensure_team(db, teams_block["away"])

    match_date = _parse_iso(fixture_block.get("date"))
    status = _STATUS_MAP.get(
        (fixture_block.get("status") or {}).get("short", ""),
        MatchStatus.scheduled,
    )

    league_name = league_block.get("name") or league_name_default
    home_score = score_block.get("home")
    away_score = score_block.get("away")

    existing = (
        await db.execute(select(Match).where(Match.api_football_id == api_id))
    ).scalar_one_or_none()

    if existing is None:
        db.add(
            Match(
                home_team_id=home.id,
                away_team_id=away.id,
                league=league_name,
                season=season,
                match_date=match_date,
                status=status,
                home_score=home_score,
                away_score=away_score,
                venue=venue_block.get("name"),
                referee=fixture_block.get("referee"),
                api_football_id=api_id,
            )
        )
        result.created += 1
        return

    # Update only when something actually changed; lets us re-run cheaply.
    # Datetime equality needs care: SQLite stores naive UTC, so the same
    # instant can compare unequal across read/write cycles. We normalise
    # both sides to naive UTC before the check.
    changed = False
    for attr, new in (
        ("status", status),
        ("home_score", home_score),
        ("away_score", away_score),
        ("match_date", match_date),
        ("venue", venue_block.get("name")),
        ("referee", fixture_block.get("referee")),
    ):
        current = getattr(existing, attr)
        if attr == "match_date":
            if _same_instant(current, new):
                continue
        elif current == new:
            continue
        setattr(existing, attr, new)
        changed = True

    if changed:
        result.updated += 1
    else:
        result.skipped += 1


def _same_instant(a: datetime | None, b: datetime | None) -> bool:
    if a is None or b is None:
        return a is b
    return _to_naive_utc(a) == _to_naive_utc(b)


def _to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(UTC).replace(tzinfo=None)


async def _ensure_team(db: AsyncSession, team_block: dict[str, Any]) -> Team:
    """Return an existing ``Team`` keyed on ``api_football_id`` or create a stub."""
    api_id = team_block["id"]
    row = (
        await db.execute(select(Team).where(Team.api_football_id == api_id))
    ).scalar_one_or_none()
    if row is not None:
        return row

    row = Team(
        name=team_block.get("name") or f"Team {api_id}",
        logo_url=team_block.get("logo"),
        api_football_id=api_id,
    )
    db.add(row)
    # Flush so subsequent fixtures in this loop find the row by api_football_id
    # without colliding on the unique constraint.
    await db.flush()
    return row


def _parse_iso(value: str | None) -> datetime:
    if not value:
        raise ValueError("fixture date missing")
    # API-Football emits "2024-08-12T17:30:00+00:00" — fromisoformat handles it.
    return datetime.fromisoformat(value)
