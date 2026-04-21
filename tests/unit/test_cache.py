"""
Unit tests for the cache layer.

Covers
------
* InMemoryCache get/set/expiry/clear semantics
* @cached decorator: miss → call + store, hit → no recall
* Opaque arguments (sessions, requests) excluded from the key
* Different kwargs produce different cache slots
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from football_ai.cache.backends import InMemoryCache
from football_ai.cache.decorator import cached
from football_ai.cache.factory import (
    override_cache_for_tests,
    reset_cache_for_tests,
)

# ── InMemoryCache ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_in_memory_set_then_get_round_trips():
    cache = InMemoryCache()
    await cache.set("k", b"hello", ttl=10)
    assert await cache.get("k") == b"hello"


@pytest.mark.asyncio
async def test_in_memory_get_missing_returns_none():
    cache = InMemoryCache()
    assert await cache.get("nope") is None


@pytest.mark.asyncio
async def test_in_memory_expiry_drops_value(monkeypatch):
    """Lazy expiry — value disappears once the deadline is past."""
    import football_ai.cache.backends as backends

    fake_now = [1000.0]
    monkeypatch.setattr(backends.time, "monotonic", lambda: fake_now[0])

    cache = InMemoryCache()
    await cache.set("k", b"v", ttl=5)
    assert await cache.get("k") == b"v"

    fake_now[0] += 6  # past TTL
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_in_memory_clear_drops_all():
    cache = InMemoryCache()
    await cache.set("a", b"1", ttl=10)
    await cache.set("b", b"2", ttl=10)
    await cache.clear()
    assert cache.size == 0


# ── @cached decorator ───────────────────────────────────────────────────────

class _Echo(BaseModel):
    value: str


@pytest.fixture
def isolated_cache():
    cache = InMemoryCache()
    override_cache_for_tests(cache)
    yield cache
    reset_cache_for_tests()


@pytest.mark.asyncio
async def test_decorator_caches_second_call(isolated_cache):
    calls: list[int] = []

    @cached(ttl=30, namespace="test", response_model=_Echo)
    async def f(*, name: str) -> _Echo:
        calls.append(1)
        return _Echo(value=f"hello-{name}")

    a = await f(name="ada")
    b = await f(name="ada")
    assert a.value == b.value == "hello-ada"
    assert len(calls) == 1, "second call should hit cache, not re-run"


@pytest.mark.asyncio
async def test_decorator_separates_distinct_kwargs(isolated_cache):
    calls: list[str] = []

    @cached(ttl=30, namespace="test", response_model=_Echo)
    async def f(*, name: str) -> _Echo:
        calls.append(name)
        return _Echo(value=name)

    await f(name="alice")
    await f(name="bob")
    await f(name="alice")  # cache hit
    assert calls == ["alice", "bob"]


@pytest.mark.asyncio
async def test_decorator_excludes_opaque_args_from_key(isolated_cache):
    """An opaque per-request arg (e.g. a session) shouldn't shatter the cache."""

    class AsyncSession:  # local stand-in matching the OPAQUE_TYPE_NAMES list
        def __init__(self, tag: int) -> None:
            self.tag = tag

    calls: list[int] = []

    @cached(ttl=30, namespace="test", response_model=_Echo)
    async def f(*, db: AsyncSession, name: str) -> _Echo:
        calls.append(db.tag)
        return _Echo(value=name)

    await f(db=AsyncSession(1), name="x")
    await f(db=AsyncSession(2), name="x")  # different db, same key
    assert len(calls) == 1
