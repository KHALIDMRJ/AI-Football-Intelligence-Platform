"""
Unit tests for the LiveMatchHub.

We use a tiny ``FakeSocket`` to avoid spinning up a real WebSocket — the
hub's contract is "call ``send_json`` on every registered socket and
drop the ones that fail", which we can verify with stand-ins.
"""

from __future__ import annotations

import uuid

import pytest

from football_ai.realtime.hub import LiveMatchHub


class FakeSocket:
    def __init__(self) -> None:
        self.received: list[dict] = []
        self.alive = True

    async def send_json(self, payload):
        if not self.alive:
            raise ConnectionError("dead socket")
        self.received.append(payload)


@pytest.mark.asyncio
async def test_register_then_publish_delivers_payload():
    hub = LiveMatchHub()
    match_id = uuid.uuid4()
    ws = FakeSocket()
    await hub.register(match_id, ws)

    delivered = await hub.publish(match_id, {"type": "event", "minute": 12})

    assert delivered == 1
    assert ws.received == [{"type": "event", "minute": 12}]


@pytest.mark.asyncio
async def test_publish_with_no_subscribers_is_zero():
    hub = LiveMatchHub()
    delivered = await hub.publish(uuid.uuid4(), {"type": "ping"})
    assert delivered == 0


@pytest.mark.asyncio
async def test_dead_sockets_are_evicted():
    hub = LiveMatchHub()
    match_id = uuid.uuid4()
    alive, dead = FakeSocket(), FakeSocket()
    dead.alive = False
    await hub.register(match_id, alive)
    await hub.register(match_id, dead)

    delivered = await hub.publish(match_id, {"type": "event"})

    assert delivered == 1
    assert hub.subscriber_count(match_id) == 1


@pytest.mark.asyncio
async def test_unregister_removes_subscriber():
    hub = LiveMatchHub()
    match_id = uuid.uuid4()
    ws = FakeSocket()
    await hub.register(match_id, ws)
    await hub.unregister(match_id, ws)
    assert hub.subscriber_count(match_id) == 0
    assert match_id not in hub.active_channels()
