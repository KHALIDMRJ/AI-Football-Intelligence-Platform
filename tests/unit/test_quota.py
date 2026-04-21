"""
Unit tests for the daily API quota tracker.

The tracker stores its counter in the cache layer — the autouse
``_isolated_cache`` fixture in conftest guarantees a clean in-memory
backend per test, so counts don't leak.
"""

from __future__ import annotations

import pytest

from football_ai.external.quota import QuotaExceeded, QuotaTracker


@pytest.mark.asyncio
async def test_new_tracker_starts_at_zero():
    q = QuotaTracker("test_provider", daily_limit=5)
    assert await q.used_today() == 0
    assert await q.remaining() == 5


@pytest.mark.asyncio
async def test_consume_increments_counter():
    q = QuotaTracker("test_provider", daily_limit=5)
    await q.consume(1)
    await q.consume(1)
    assert await q.used_today() == 2
    assert await q.remaining() == 3


@pytest.mark.asyncio
async def test_exceeding_limit_raises_and_counter_unchanged():
    q = QuotaTracker("test_provider", daily_limit=2)
    await q.consume(1)
    await q.consume(1)
    with pytest.raises(QuotaExceeded):
        await q.consume(1)
    assert await q.used_today() == 2  # failed consume must not increment


@pytest.mark.asyncio
async def test_unlimited_provider_never_blocks():
    q = QuotaTracker("unlimited", daily_limit=-1)
    for _ in range(1000):
        await q.consume(1)
    assert await q.remaining() is None


@pytest.mark.asyncio
async def test_snapshot_shape():
    q = QuotaTracker("snap_provider", daily_limit=10)
    await q.consume(3)
    snap = await q.snapshot()
    assert snap == {
        "provider": "snap_provider",
        "limit": 10,
        "used": 3,
        "remaining": 7,
    }
