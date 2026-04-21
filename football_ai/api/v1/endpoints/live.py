"""
Live match WebSocket endpoint — fan-out for in-progress fixtures.

Routes
------
GET /api/v1/live/health                    — hub introspection (admin)
WS  /api/v1/live/matches/{match_id}        — subscribe to a match's event stream
POST /api/v1/live/matches/{match_id}/push  — publish an event (admin only)

Auth
----
WebSocket auth uses the ``token`` query string parameter (browsers can't
attach an ``Authorization`` header to a WS handshake natively). The
endpoint enforces ``pro_analyst+`` because the live event feed is the
primary premium feature — it's what differentiates the paid tier from
free fixture browsing.

Wire format
-----------
Server → client messages are JSON objects with at minimum::

    {"type": "<kind>", "match_id": "<uuid>", ...}

``type`` values currently emitted:
* ``"hello"``       — sent on connect with the current match snapshot
* ``"event"``       — a new ``MatchEvent`` (goal/card/sub/...)
* ``"prediction"``  — refreshed model probabilities
* ``"ping"``        — heartbeat (every 25s) so proxies don't idle-close

Client → server is currently ignored — this is a one-way feed. We read
in a tiny background loop only to detect disconnects promptly.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.api.dependencies import get_db, require_role
from football_ai.core.exceptions import AuthError, DataNotFoundError
from football_ai.core.security import decode_token
from football_ai.crud import match as match_crud
from football_ai.crud import user as user_crud
from football_ai.logger import get_logger
from football_ai.models.user import UserRole
from football_ai.realtime import LiveMatchHub, get_match_hub

logger = get_logger(__name__)

router = APIRouter(prefix="/live", tags=["live"])

_HEARTBEAT_INTERVAL = 25.0  # seconds — under most proxy idle-close thresholds
_PRO_ROLE_RANK = 1          # pro_analyst (mirrors api/dependencies._ROLE_ORDER)


# ── HTTP helpers ─────────────────────────────────────────────────────────────

@router.get(
    "/health",
    summary="Hub introspection (admin)",
    dependencies=[Depends(require_role(UserRole.admin))],
)
async def live_health(
    hub: Annotated[LiveMatchHub, Depends(get_match_hub)],
) -> dict[str, Any]:
    """Quick visibility into how many sockets the hub is currently holding."""
    return {
        "active_channels": [str(m) for m in hub.active_channels()],
        "total_subscribers": hub.total_subscribers(),
    }


@router.post(
    "/matches/{match_id}/push",
    summary="Publish a synthetic event to the live channel (admin)",
    description=(
        "Used by the admin demo page (and by the arq event-ingest task) to "
        "push a payload onto the live channel without going through the "
        "external API. ``payload`` is forwarded verbatim — the only field "
        "we add is ``match_id``."
    ),
    dependencies=[Depends(require_role(UserRole.admin))],
)
async def push_live_event(
    match_id: uuid.UUID,
    payload: dict[str, Any],
    hub: Annotated[LiveMatchHub, Depends(get_match_hub)],
) -> dict[str, Any]:
    payload = {**payload, "match_id": str(match_id)}
    delivered = await hub.publish(match_id, payload)
    return {"delivered": delivered, "subscribers": hub.subscriber_count(match_id)}


# ── WebSocket ────────────────────────────────────────────────────────────────

@router.websocket("/matches/{match_id}")
async def live_match_socket(
    websocket: WebSocket,
    match_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    hub: Annotated[LiveMatchHub, Depends(get_match_hub)],
    token: str = Query(..., description="JWT access token (pro_analyst+ required)."),
) -> None:
    # ── Auth: WS handshakes can't carry an Authorization header in browsers,
    #         so we accept the token via query string and validate it the same
    #         way `get_current_user` does.
    try:
        payload = decode_token(token, expected_type="access")
    except AuthError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
        return

    user = await user_crud.get_by_id(db, payload["sub"])
    if user is None or not user.is_active:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="User inactive")
        return
    # Inline role check — we can't use `require_role` from a WS context.
    if _role_rank(user.role) < _PRO_ROLE_RANK:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="pro_analyst role required",
        )
        return

    # ── Match must exist before we hold the socket open.
    match = await match_crud.get_match(db, match_id)
    if match is None:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Match not found")
        # 1011 used over 1008 because this is a "server can't fulfil" condition;
        # mirrors how a REST endpoint would 404 vs 403.
        raise DataNotFoundError(f"Match {match_id} not found.")

    await websocket.accept()
    await hub.register(match_id, websocket)

    # Greet the client with a small snapshot so the UI has something to render
    # immediately, before any event arrives.
    await websocket.send_json({
        "type": "hello",
        "match_id": str(match_id),
        "status": match.status.value,
        "home_score": match.home_score,
        "away_score": match.away_score,
    })

    heartbeat_task = asyncio.create_task(_heartbeat(websocket, match_id))

    try:
        # Read loop — we don't process inbound messages today, but reading is
        # how we notice the client closed the socket.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        heartbeat_task.cancel()
        await hub.unregister(match_id, websocket)


async def _heartbeat(websocket: WebSocket, match_id: uuid.UUID) -> None:
    """Send a periodic ping so idle proxies don't drop the connection."""
    try:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            await websocket.send_json({"type": "ping", "match_id": str(match_id)})
    except (WebSocketDisconnect, asyncio.CancelledError):
        return
    except Exception as exc:
        logger.debug("Heartbeat for %s ended: %s", match_id, exc)


def _role_rank(role: UserRole) -> int:
    # Local copy of the ranking dict to avoid an import cycle with dependencies.
    return {
        UserRole.free_user: 0,
        UserRole.pro_analyst: 1,
        UserRole.club_scout: 2,
        UserRole.admin: 3,
    }[role]
