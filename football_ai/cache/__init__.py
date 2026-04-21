"""
Cache layer — Redis with transparent in-memory fallback.

Why a fallback
--------------
Football data is read-heavy: rankings, fixtures, accuracy reports get hit
per-page-load by every dashboard user. Redis is the right answer in
production, but forcing every developer (and every CI run) to spin up a
Redis instance just to start the API is hostile. So:

* If ``platform_settings.redis_url`` resolves and ``PING`` succeeds at
  startup → :class:`RedisCache`.
* Otherwise → :class:`InMemoryCache` (process-local, lost on restart).

The two share the :class:`Cache` interface so endpoint code never branches
on backend.

Public surface
--------------
* :func:`get_cache`        — FastAPI dependency, returns the active backend
* :func:`cached`           — endpoint decorator with TTL + key-builder hook
* :class:`Cache`           — protocol all backends implement
* :class:`InMemoryCache`   — TTL-aware dict cache (process-local)
* :class:`RedisCache`      — async redis-py wrapper

Decisions worth knowing
-----------------------
* Values are JSON-serialised via :mod:`pydantic` when the return type is
  a :class:`pydantic.BaseModel`, otherwise via stdlib :mod:`json`. This
  keeps the cache portable across processes (Redis can hold the bytes
  unchanged) and avoids pickle's security footguns.
* Cache misses on the decorator path NEVER swallow user errors — only
  cache infra errors are caught. A failing endpoint is observable; a
  failing cache silently degrades.
"""

from __future__ import annotations

from .backends import Cache, InMemoryCache, RedisCache
from .decorator import cached
from .factory import close_cache, get_cache, init_cache

__all__ = [
    "Cache",
    "InMemoryCache",
    "RedisCache",
    "cached",
    "get_cache",
    "init_cache",
    "close_cache",
]
