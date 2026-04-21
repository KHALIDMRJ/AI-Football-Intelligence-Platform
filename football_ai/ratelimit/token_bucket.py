"""
Cache-backed token bucket.

Storage layout
--------------
Per-identity state is serialised as a compact JSON blob::

    {"t": <tokens_remaining_float>, "r": <last_refill_monotonic_float>}

keyed by ``"{namespace}:{identity}"`` in the cache backend. TTL is set
to ``ceil(capacity / refill_rate) + 10`` so idle buckets evict on their
own — the bucket is full after that long anyway, nothing to lose.

Concurrency
-----------
We serialise read-modify-write via an ``asyncio.Lock`` inside the
process. That is *sufficient* for the single-uvicorn-worker dev/test
setup and the small-scale deployments this project targets. At higher
scale (multiple workers, or a sharded deployment), callers should swap
in a Redis-Lua implementation so the check is atomic across processes;
the :class:`TokenBucket` API stays the same, only the storage semantics
change. We prefer the simpler impl now over a complex one we don't
yet need (YAGNI — the hot path is a cheap dict get on the cache).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass

from football_ai.cache.backends import Cache


@dataclass
class _BucketState:
    tokens: float
    last_refill: float

    def encode(self) -> bytes:
        return json.dumps({"t": self.tokens, "r": self.last_refill}).encode()

    @classmethod
    def decode(cls, raw: bytes) -> _BucketState:
        payload = json.loads(raw.decode())
        return cls(tokens=float(payload["t"]), last_refill=float(payload["r"]))


@dataclass
class ConsumeResult:
    allowed: bool
    remaining: int
    retry_after: float  # seconds until one token is available (0 when allowed)


class TokenBucket:
    """Leaky-token bucket over the :class:`Cache` protocol.

    Parameters
    ----------
    cache:
        Backend that implements ``get`` / ``set`` / ``delete``. In prod this
        is Redis; in dev/tests it is :class:`InMemoryCache`.
    capacity:
        Maximum tokens the bucket can hold — i.e. the burst size. A
        first-time caller starts with a full bucket.
    refill_per_second:
        Tokens added per wall-second. ``capacity / refill_per_second`` is
        the time to refill from empty.
    namespace:
        Key prefix in the cache. Separate namespaces per policy (e.g.
        ``"rl:predictions_live"``, ``"rl:admin_sync"``) so two endpoints
        don't share a bucket.
    """

    def __init__(
        self,
        cache: Cache,
        *,
        capacity: float,
        refill_per_second: float,
        namespace: str = "rl",
    ) -> None:
        if capacity <= 0 or refill_per_second <= 0:
            raise ValueError("capacity and refill_per_second must be positive")
        self._cache = cache
        self._capacity = float(capacity)
        self._refill = float(refill_per_second)
        self._namespace = namespace
        self._lock = asyncio.Lock()
        # Add a small pad so the bucket TTL outlives a full refill cycle.
        self._ttl = int(math.ceil(self._capacity / self._refill)) + 10

    async def consume(self, identity: str, *, cost: float = 1.0) -> ConsumeResult:
        """Attempt to take ``cost`` tokens for ``identity``.

        Returns :class:`ConsumeResult` — callers translate it into an HTTP
        429 with ``Retry-After`` when ``allowed`` is False.
        """
        key = f"{self._namespace}:{identity}"
        async with self._lock:
            now = time.monotonic()
            raw = await self._cache.get(key)
            if raw is None:
                state = _BucketState(tokens=self._capacity, last_refill=now)
            else:
                state = _BucketState.decode(raw)
                elapsed = max(0.0, now - state.last_refill)
                state.tokens = min(self._capacity, state.tokens + elapsed * self._refill)
                state.last_refill = now

            if state.tokens >= cost:
                state.tokens -= cost
                await self._cache.set(key, state.encode(), ttl=self._ttl)
                return ConsumeResult(
                    allowed=True,
                    remaining=int(state.tokens),
                    retry_after=0.0,
                )

            retry_after = (cost - state.tokens) / self._refill
            # Persist the drained state so rapid retries don't reset the clock.
            await self._cache.set(key, state.encode(), ttl=self._ttl)
            return ConsumeResult(
                allowed=False,
                remaining=0,
                retry_after=retry_after,
            )

    async def reset(self, identity: str) -> None:
        """Forget a bucket (admin override or test cleanup)."""
        await self._cache.delete(f"{self._namespace}:{identity}")
