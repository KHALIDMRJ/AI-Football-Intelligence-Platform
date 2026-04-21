"""
Cache backends — concrete implementations behind the :class:`Cache` protocol.

InMemoryCache
-------------
Dict-backed, TTL-aware, single-process. We don't try to evict on a timer
— a stale entry is dropped lazily on the next ``get``. That keeps the
hot path lock-free and matches the only realistic use cases (dev, test,
a single uvicorn worker behind no load).

RedisCache
----------
Thin wrapper over :class:`redis.asyncio.Redis`. We deliberately keep the
surface minimal (get/set/delete/clear) and let callers JSON-encode their
own payloads — see :mod:`football_ai.cache.decorator` for the canonical
encoding path.

Errors from Redis (network drops, auth failures) are NOT swallowed at
this layer. The decorator catches them so a Redis outage never breaks an
endpoint, but raw callers (e.g. arq tasks) can still observe them.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    """Backend-agnostic async cache surface.

    All keys are namespaced strings (see :mod:`football_ai.cache.decorator`).
    All values are bytes — encoding/decoding is the caller's job so the
    cache stays serialisation-format agnostic (JSON today, msgpack later).
    """

    async def get(self, key: str) -> bytes | None: ...
    async def set(self, key: str, value: bytes, *, ttl: int) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def clear(self) -> None: ...
    async def close(self) -> None: ...


# ── In-memory backend ────────────────────────────────────────────────────────

class InMemoryCache:
    """Process-local cache. Used in dev/CI when Redis isn't reachable.

    NOT safe across multiple uvicorn workers — each worker keeps its own
    copy. Production deployments must use Redis.
    """

    def __init__(self) -> None:
        # value, expires_at (epoch seconds)
        self._store: dict[str, tuple[bytes, float]] = {}

    async def get(self, key: str) -> bytes | None:
        item = self._store.get(key)
        if item is None:
            return None
        value, expires_at = item
        if expires_at < time.monotonic():
            # Lazy eviction — cheaper than a sweeper thread, fine for our load.
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: bytes, *, ttl: int) -> None:
        self._store[key] = (value, time.monotonic() + max(1, int(ttl)))

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def clear(self) -> None:
        self._store.clear()

    async def close(self) -> None:
        # Nothing to release; method exists to satisfy the protocol.
        self._store.clear()

    # ── Test helpers ─────────────────────────────────────────────────────────
    @property
    def size(self) -> int:
        return len(self._store)


# ── Redis backend ────────────────────────────────────────────────────────────

class RedisCache:
    """Async wrapper over :class:`redis.asyncio.Redis`.

    Constructed via :func:`football_ai.cache.factory.init_cache`, which
    handles the connection probe. Direct instantiation is supported for
    tests that supply a fake client.
    """

    def __init__(self, client) -> None:  # client: redis.asyncio.Redis
        self._client = client

    async def get(self, key: str) -> bytes | None:
        return await self._client.get(key)

    async def set(self, key: str, value: bytes, *, ttl: int) -> None:
        await self._client.set(key, value, ex=max(1, int(ttl)))

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def clear(self) -> None:
        # FLUSHDB only on the configured logical DB — never FLUSHALL.
        await self._client.flushdb()

    async def close(self) -> None:
        await self._client.aclose()
