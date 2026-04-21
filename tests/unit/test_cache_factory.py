"""
Unit tests for the cache factory — init/get/close + fallback + redaction.

These tests exercise the parts of ``factory.py`` that the rest of the
suite skips by installing an :class:`InMemoryCache` directly: the
Redis-probe branch, the lazy ``get_cache`` path, and the singleton
lifecycle. No real Redis is required — we monkeypatch the async
client so the probe either "succeeds" or "fails" deterministically.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from football_ai.cache import factory as cache_factory
from football_ai.cache.backends import InMemoryCache, RedisCache


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test starts with a cold factory — no leaked state."""
    cache_factory.reset_cache_for_tests()
    yield
    cache_factory.reset_cache_for_tests()


# ── init_cache ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_init_cache_falls_back_when_redis_url_blank(monkeypatch):
    monkeypatch.setattr(
        cache_factory.platform_settings, "redis_url", "   ", raising=False
    )
    cache = await cache_factory.init_cache()
    assert isinstance(cache, InMemoryCache)


@pytest.mark.asyncio
async def test_init_cache_uses_redis_when_ping_succeeds(monkeypatch):
    """When the probe succeeds, the factory returns a RedisCache wrapping the client."""
    monkeypatch.setattr(
        cache_factory.platform_settings,
        "redis_url",
        "redis://user:secret@cache.example:6379/0",
        raising=False,
    )

    class _FakeClient:
        async def ping(self) -> bool:
            return True

        async def aclose(self) -> None:
            return None

    fake_mod = SimpleNamespace(from_url=lambda *a, **kw: _FakeClient())
    # The factory does `import redis.asyncio as aioredis` inside init_cache,
    # so we need to intercept the whole module lookup.
    import sys
    monkeypatch.setitem(sys.modules, "redis.asyncio", fake_mod)

    cache = await cache_factory.init_cache()
    assert isinstance(cache, RedisCache)


@pytest.mark.asyncio
async def test_init_cache_is_idempotent(monkeypatch):
    """Second call returns the same singleton — no duplicate pools."""
    monkeypatch.setattr(cache_factory.platform_settings, "redis_url", "", raising=False)
    first = await cache_factory.init_cache()
    second = await cache_factory.init_cache()
    assert first is second


# ── get_cache ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_cache_lazy_initialises_when_lifespan_skipped(monkeypatch):
    """Under pytest, the lifespan hook doesn't always run — get_cache must cope."""
    monkeypatch.setattr(cache_factory.platform_settings, "redis_url", "", raising=False)
    cache = await cache_factory.get_cache()
    assert isinstance(cache, InMemoryCache)


# ── close_cache ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_close_cache_releases_singleton(monkeypatch):
    monkeypatch.setattr(cache_factory.platform_settings, "redis_url", "", raising=False)
    await cache_factory.init_cache()
    assert cache_factory._cache is not None
    await cache_factory.close_cache()
    assert cache_factory._cache is None


# ── _redact ──────────────────────────────────────────────────────────────────

def test_redact_hides_credentials():
    assert cache_factory._redact("redis://user:pw@host:6379/0") == "redis://***@host:6379/0"


def test_redact_leaves_urls_without_creds_intact():
    assert cache_factory._redact("redis://host:6379/0") == "redis://host:6379/0"
