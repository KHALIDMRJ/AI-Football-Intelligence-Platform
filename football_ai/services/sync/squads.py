"""
Squad sync — pull a team's roster and upsert into ``Player``.

The provider's squad payload is shallower than its per-player endpoint
(no market value, foot, etc.). We deliberately don't enrich here; that's
a separate (and more quota-expensive) sync we don't ship in this phase.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.external.api_football import APIFootballClient
from football_ai.logger import get_logger
from football_ai.models.player import Player
from football_ai.models.team import Team

from .base import SyncResult

logger = get_logger(__name__)


async def sync_team_squad(
    db: AsyncSession,
    client: APIFootballClient,
    *,
    team_id: uuid.UUID,
) -> SyncResult:
    """Fetch the squad for ``team_id`` (our UUID) and upsert players."""
    team = (
        await db.execute(select(Team).where(Team.id == team_id))
    ).scalar_one_or_none()
    if team is None:
        raise ValueError(f"Team {team_id} not found")
    if team.api_football_id is None:
        raise ValueError(
            f"Team {team_id} has no api_football_id — sync fixtures first or seed manually."
        )

    result = SyncResult(target=f"squad:{team.api_football_id}")
    players = await client.get_squad(team=team.api_football_id)
    result.fetched = len(players)

    for raw in players:
        try:
            await _upsert_player(db, raw, team=team, result=result)
        except Exception as exc:
            result.add_warning(f"player {raw.get('id', '?')} skipped: {exc}")
            result.skipped += 1

    await db.commit()
    logger.info(
        "Squad sync done: %s — created=%d updated=%d skipped=%d",
        result.target, result.created, result.updated, result.skipped,
    )
    return result


# ── Internals ────────────────────────────────────────────────────────────────

async def _upsert_player(
    db: AsyncSession,
    raw: dict[str, Any],
    *,
    team: Team,
    result: SyncResult,
) -> None:
    api_id = raw["id"]
    existing = (
        await db.execute(select(Player).where(Player.api_football_id == api_id))
    ).scalar_one_or_none()

    name = raw.get("name") or f"Player {api_id}"
    age = raw.get("age")
    position = raw.get("position")
    jersey = raw.get("number")

    if existing is None:
        db.add(
            Player(
                name=name,
                age=age,
                position=position,
                jersey_number=jersey,
                current_team_id=team.id,
                api_football_id=api_id,
            )
        )
        result.created += 1
        return

    changed = False
    for attr, new in (
        ("name", name),
        ("age", age),
        ("position", position),
        ("jersey_number", jersey),
        ("current_team_id", team.id),
    ):
        if getattr(existing, attr) != new:
            setattr(existing, attr, new)
            changed = True

    if changed:
        result.updated += 1
    else:
        result.skipped += 1
