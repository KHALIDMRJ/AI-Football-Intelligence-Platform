"""
LiveMatchHub — fan-out for live ``MatchEvent`` payloads over WebSockets.

Threading / async model
-----------------------
* The hub is a single process-local object; concurrent registers and
  publishes are guarded by an :class:`asyncio.Lock` because we mutate
  dict-of-set state.
* Send is best-effort: if a client's send raises (closed socket, slow
  reader), we drop them from the set and continue. We don't await every
  client serially — failures don't slow down the rest of the room.
* Payloads are JSON-serialisable dicts. We don't try to be type-safe at
  this layer because the publishers (event ingestion, prediction
  refresh, manual admin pushes) emit different shapes; consumers
  discriminate on a ``type`` field.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from football_ai.logger import get_logger

logger = get_logger(__name__)


class LiveMatchHub:
    """Maintains the set of WebSockets subscribed to each match.

    Channels are keyed by ``match_id`` (UUID). The hub never persists —
    if the process restarts, clients reconnect and re-subscribe.
    """

    def __init__(self) -> None:
        # match_id -> set of connected WebSocket objects
        self._channels: dict[uuid.UUID, set[Any]] = {}
        self._lock = asyncio.Lock()

    # ── Subscription lifecycle ────────────────────────────────────────────────

    async def register(self, match_id: uuid.UUID, websocket: Any) -> None:
        """Add a WebSocket to a match's subscriber set.

        The caller must already have ``await ws.accept()``-ed the
        connection — the hub doesn't manage the handshake.
        """
        async with self._lock:
            self._channels.setdefault(match_id, set()).add(websocket)
        logger.debug("WS registered for match %s (now %d subs)",
                     match_id, self.subscriber_count(match_id))

    async def unregister(self, match_id: uuid.UUID, websocket: Any) -> None:
        async with self._lock:
            subs = self._channels.get(match_id)
            if subs:
                subs.discard(websocket)
                if not subs:
                    self._channels.pop(match_id, None)
        logger.debug("WS unregistered from match %s", match_id)

    # ── Publish ───────────────────────────────────────────────────────────────

    async def publish(self, match_id: uuid.UUID, payload: dict[str, Any]) -> int:
        """Broadcast ``payload`` to every subscriber of ``match_id``.

        Returns the number of successful sends. Drops dead sockets
        silently — the next publish iteration won't see them.
        """
        async with self._lock:
            subs = list(self._channels.get(match_id, ()))
        if not subs:
            return 0

        sent = 0
        dead: list[Any] = []
        for ws in subs:
            try:
                await ws.send_json(payload)
                sent += 1
            except Exception as exc:
                logger.debug("Dropping dead WS for %s (%s)", match_id, exc)
                dead.append(ws)

        if dead:
            async with self._lock:
                bucket = self._channels.get(match_id)
                if bucket:
                    for ws in dead:
                        bucket.discard(ws)
                    if not bucket:
                        self._channels.pop(match_id, None)

        return sent

    # ── Introspection (used by tests + admin) ─────────────────────────────────

    def subscriber_count(self, match_id: uuid.UUID) -> int:
        return len(self._channels.get(match_id, ()))

    def total_subscribers(self) -> int:
        return sum(len(s) for s in self._channels.values())

    def active_channels(self) -> list[uuid.UUID]:
        return list(self._channels.keys())


# ── Process-wide singleton ───────────────────────────────────────────────────

_hub = LiveMatchHub()


def get_match_hub() -> LiveMatchHub:
    """FastAPI dependency — returns the shared hub for this process."""
    return _hub
