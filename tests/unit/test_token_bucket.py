"""
Unit tests for the token-bucket rate limiter.

We drive ``time.monotonic`` directly (not ``asyncio.sleep``) so the tests
are deterministic and fast — refilling over 60 wall-seconds would be
silly to test in real time.
"""

from __future__ import annotations

import pytest

from football_ai.cache.backends import InMemoryCache
from football_ai.ratelimit.token_bucket import TokenBucket


@pytest.mark.asyncio
async def test_first_call_starts_full_bucket():
    bucket = TokenBucket(InMemoryCache(), capacity=5, refill_per_second=1.0)
    result = await bucket.consume("user:1")
    assert result.allowed is True
    # After taking one token, 4 remain out of a capacity of 5.
    assert result.remaining == 4
    assert result.retry_after == 0.0


@pytest.mark.asyncio
async def test_drains_then_rejects():
    bucket = TokenBucket(InMemoryCache(), capacity=3, refill_per_second=0.5)
    for _ in range(3):
        assert (await bucket.consume("user:1")).allowed is True
    denied = await bucket.consume("user:1")
    assert denied.allowed is False
    assert denied.remaining == 0
    # Refill at 0.5/s means ~2s to earn one token back.
    assert 1.5 <= denied.retry_after <= 2.5


@pytest.mark.asyncio
async def test_refills_over_time(monkeypatch):
    import football_ai.ratelimit.token_bucket as tb

    clock = [1000.0]
    monkeypatch.setattr(tb.time, "monotonic", lambda: clock[0])

    bucket = TokenBucket(InMemoryCache(), capacity=2, refill_per_second=1.0)
    # Drain.
    assert (await bucket.consume("u")).allowed is True
    assert (await bucket.consume("u")).allowed is True
    assert (await bucket.consume("u")).allowed is False

    # Advance time by 1.5s → 1.5 tokens earned (capped at capacity).
    clock[0] += 1.5
    # First post-refill call succeeds (we have 1.5 tokens).
    assert (await bucket.consume("u")).allowed is True
    # Only 0.5 tokens left — the next call is rejected.
    assert (await bucket.consume("u")).allowed is False


@pytest.mark.asyncio
async def test_separate_identities_have_independent_buckets():
    bucket = TokenBucket(InMemoryCache(), capacity=1, refill_per_second=0.01)
    assert (await bucket.consume("alice")).allowed is True
    # Alice is drained, Bob still has a full bucket.
    assert (await bucket.consume("alice")).allowed is False
    assert (await bucket.consume("bob")).allowed is True


@pytest.mark.asyncio
async def test_reset_wipes_bucket_state():
    bucket = TokenBucket(InMemoryCache(), capacity=1, refill_per_second=0.01)
    await bucket.consume("u")
    assert (await bucket.consume("u")).allowed is False
    await bucket.reset("u")
    assert (await bucket.consume("u")).allowed is True


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        TokenBucket(InMemoryCache(), capacity=0, refill_per_second=1.0)
    with pytest.raises(ValueError):
        TokenBucket(InMemoryCache(), capacity=1, refill_per_second=0)
