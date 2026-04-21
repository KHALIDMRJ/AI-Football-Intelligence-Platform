"""
Cache factory — picks Redis when reachable, falls back to in-memory.

The decision is made once, at startup, by :func:`init_cache`. After that,
:func:`get_cache` returns the same backend for the lifetime of the process.

Why a singleton
---------------
* Connection pools are expensive — we don't want a new Redis client per
  request, and the in-memory backend is meaningless if it isn't shared.
* FastAPI's dependency-injection is keyed on callable identity, so a
  module-level singleton keeps lifetimes simple and tests easy to override.
"""

from __future__ import annotations

import asyncio

from football_ai.config import platform_settings
from football_ai.logger import get_logger

from .backends import Cache, InMemoryCache, RedisCache

logger = get_logger(__name__)

_cache: Cache | None = None
_init_lock = asyncio.Lock()


async def init_cache() -> Cache:
    """Initialise the process-wide cache. Call from FastAPI lifespan startup.

    Tries Redis first; falls back to in-memory if PING fails for any
    reason (DNS, refused connection, auth). Logs the decision so ops
    can tell which backend is live without grepping config.
    """
    global _cache
    async with _init_lock:
        if _cache is not None:
            return _cache

        url = platform_settings.redis_url.strip()
        if url:
            try:
                import redis.asyncio as aioredis  # local import — optional dep at runtime

                client = aioredis.from_url(url, encoding=None, decode_responses=False)
                # Probe with a short timeout so a misconfigured URL doesn't hang startup.
                await asyncio.wait_for(client.ping(), timeout=1.5)
                _cache = RedisCache(client)
                logger.info("Cache backend: Redis (%s)", _redact(url))
                return _cache
            except Exception as exc:  # pragma: no cover — depends on env
                logger.warning(
                    "Redis unreachable (%s) — falling back to in-memory cache.",
                    exc.__class__.__name__,
                )

        _cache = InMemoryCache()
        logger.info("Cache backend: in-memory (no Redis configured/reachable)")
        return _cache


async def get_cache() -> Cache:
    """FastAPI dependency. Lazy-initialises if lifespan didn't run (tests)."""
    if _cache is None:
        return await init_cache()
    return _cache


async def close_cache() -> None:
    """Release connections. Call from FastAPI lifespan shutdown."""
    global _cache
    if _cache is not None:
        await _cache.close()
        _cache = None


def reset_cache_for_tests() -> None:
    """Wipe the singleton so the next ``get_cache`` re-initialises fresh."""
    global _cache
    _cache = None


def override_cache_for_tests(cache: Cache) -> None:
    """Inject a specific backend (used by the test suite)."""
    global _cache
    _cache = cache


def _redact(url: str) -> str:
    """Hide ``redis://user:password@host`` credentials in logs."""
    if "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.split("@", 1)
    return f"{scheme}://***@{host}"
