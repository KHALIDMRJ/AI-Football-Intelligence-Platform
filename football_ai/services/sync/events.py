"""
Match events sync — pull goals/cards/subs/VAR for a fixture and upsert.

Idempotency
-----------
Events don't carry a stable provider id, so we synthesise the natural
key from ``(match_id, minute, event_type, player_id_or_zero)``. Two
syncs of the same fixture's complete event list produce zero new rows.

Live broadcast
--------------
Each newly-inserted event is also published onto the live hub so any
WebSocket subscriber for that fixture receives it immediately. Hub
publish is best-effort: a failure logs but doesn't roll back the DB
write — persistence is the source of truth.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.external.api_football import APIFootballClient
from football_ai.logger import get_logger
from football_ai.models.match import EventType, Match, MatchEvent
from football_ai.models.player import Player
from football_ai.realtime import get_match_hub

from .base import SyncResult

logger = get_logger(__name__)


# Provider event labels → our EventType enum.
_TYPE_MAP: dict[tuple[str, str], EventType] = {
    ("Goal", "Normal Goal"): EventType.goal,
    ("Goal", "Own Goal"): EventType.own_goal,
    ("Goal", "Penalty"): EventType.goal,
    ("Goal", "Missed Penalty"): EventType.penalty_missed,
    ("Card", "Yellow Card"): EventType.yellow_card,
    ("Card", "Red Card"): EventType.red_card,
    ("Card", "Second Yellow card"): EventType.red_card,
    ("subst", ""): EventType.substitution,
    ("Var", ""): EventType.var,
}


async def sync_fixture_events(
    db: AsyncSession,
    client: APIFootballClient,
    *,
    match_id: uuid.UUID,
) -> SyncResult:
    match = (
        await db.execute(select(Match).where(Match.id == match_id))
    ).scalar_one_or_none()
    if match is None:
        raise ValueError(f"Match {match_id} not found")
    if match.api_football_id is None:
        raise ValueError(
            f"Match {match_id} has no api_football_id — sync fixtures first."
        )

    result = SyncResult(target=f"events:{match.api_football_id}")
    raw_events = await client.list_fixture_events(fixture=match.api_football_id)
    result.fetched = len(raw_events)

    hub = get_match_hub()
    for raw in raw_events:
        try:
            inserted = await _upsert_event(db, raw, match_id=match.id, result=result)
            if inserted is not None:
                await _broadcast_event(hub, match.id, inserted)
        except Exception as exc:
            result.add_warning(f"event @ {raw.get('time', {}).get('elapsed', '?')}': {exc}")
            result.skipped += 1

    await db.commit()
    logger.info(
        "Event sync done: %s — created=%d skipped=%d",
        result.target,
        result.created,
        result.skipped,
    )
    return result


# ── Internals ────────────────────────────────────────────────────────────────

async def _upsert_event(
    db: AsyncSession,
    raw: dict[str, Any],
    *,
    match_id: uuid.UUID,
    result: SyncResult,
) -> MatchEvent | None:
    minute = (raw.get("time") or {}).get("elapsed")
    if minute is None:
        raise ValueError("event missing minute")

    event_type = _resolve_type(raw)
    if event_type is None:
        # Unknown event — skip silently rather than warn (provider adds new
        # types occasionally).
        result.skipped += 1
        return None

    player_block = raw.get("player") or {}
    api_player_id = player_block.get("id")
    player_uuid = await _resolve_player(db, api_player_id) if api_player_id else None

    # Natural-key dedupe — same minute + type + player on same match.
    existing = (
        await db.execute(
            select(MatchEvent).where(
                MatchEvent.match_id == match_id,
                MatchEvent.minute == int(minute),
                MatchEvent.event_type == event_type,
                MatchEvent.player_id == player_uuid,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        result.skipped += 1
        return None

    extra = {
        "detail": raw.get("detail"),
        "comments": raw.get("comments"),
    }
    if (assist := raw.get("assist") or {}).get("id"):
        extra["assist"] = assist

    row = MatchEvent(
        match_id=match_id,
        minute=int(minute),
        event_type=event_type,
        player_id=player_uuid,
        extra_info={k: v for k, v in extra.items() if v is not None},
    )
    db.add(row)
    await db.flush()
    result.created += 1
    return row


def _resolve_type(raw: dict[str, Any]) -> EventType | None:
    type_label = (raw.get("type") or "").strip()
    detail_label = (raw.get("detail") or "").strip()
    return (
        _TYPE_MAP.get((type_label, detail_label))
        or _TYPE_MAP.get((type_label, ""))
    )


async def _resolve_player(db: AsyncSession, api_id: int) -> uuid.UUID | None:
    row = (
        await db.execute(select(Player).where(Player.api_football_id == api_id))
    ).scalar_one_or_none()
    return row.id if row else None


async def _broadcast_event(hub, match_id: uuid.UUID, event: MatchEvent) -> None:
    try:
        await hub.publish(
            match_id,
            {
                "type": "event",
                "match_id": str(match_id),
                "minute": event.minute,
                "kind": event.event_type.value,
                "player_id": str(event.player_id) if event.player_id else None,
                "extra": event.extra_info or {},
            },
        )
    except Exception as exc:  # broadcast is best-effort
        logger.warning("Live broadcast failed for %s: %s", match_id, exc)
