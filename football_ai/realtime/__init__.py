"""
Realtime layer — WebSocket connection hubs and broadcast helpers.

The match hub fan-outs live ``MatchEvent`` payloads to every client
subscribed to a fixture. Background tasks (Phase 6 arq workers) and the
external API ingest path (Phase 7) call :meth:`LiveMatchHub.publish` to
push updates; the WebSocket endpoint just forwards them.

Why a process-local hub
-----------------------
A hub-per-process is fine for a single uvicorn worker, which is what the
portfolio deploy targets. A multi-worker deployment would back this with
Redis pub/sub — the surface intentionally matches that future shape
(``publish(channel, payload)``) so the swap is mechanical.
"""

from __future__ import annotations

from .hub import LiveMatchHub, get_match_hub

__all__ = ["LiveMatchHub", "get_match_hub"]
