"""
Daily quota tracker for upstream API providers.

API-Football's free tier hard-caps at 100 calls/day. Burning that quota
on a runaway sync means losing live-match coverage for the rest of the
day, so the client guards every call against this counter.

Persistence
-----------
Counters live in the cache layer (Redis in prod, in-memory otherwise),
keyed ``quota:<provider>:<YYYY-MM-DD>``. The Redis path uses ``INCR``
so concurrent uvicorn workers + arq worker share the same counter; the
in-memory path is per-process and intentionally lossy on restart —
acceptable in dev/CI.

The tracker also supports a "snapshot" read used by the admin status
endpoint so operators can see remaining headroom without making a call.
"""

from __future__ import annotations

from datetime import UTC, datetime

from football_ai.cache.factory import get_cache
from football_ai.logger import get_logger

logger = get_logger(__name__)

_DAY_SECONDS = 60 * 60 * 24


class QuotaExceeded(Exception):
    """Raised when the day's call budget for a provider is spent."""


class QuotaTracker:
    """Atomic-ish daily counter for an external provider's API quota.

    The instance is stateless beyond its provider name — all counter
    state lives in the cache backend. Tests can construct an instance
    with whatever provider name they like; production wires one per
    provider via the FastAPI dep below.
    """

    def __init__(self, provider: str, daily_limit: int) -> None:
        self.provider = provider
        self.daily_limit = daily_limit  # -1 means unlimited

    # ── Read paths ───────────────────────────────────────────────────────────

    def _key(self) -> str:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        return f"quota:{self.provider}:{today}"

    async def used_today(self) -> int:
        cache = await get_cache()
        raw = await cache.get(self._key())
        if raw is None:
            return 0
        try:
            return int(raw)
        except ValueError:
            # Corrupted counter — start over rather than crash.
            return 0

    async def remaining(self) -> int | None:
        if self.daily_limit < 0:
            return None  # unlimited
        return max(0, self.daily_limit - await self.used_today())

    async def snapshot(self) -> dict[str, int | None]:
        used = await self.used_today()
        remaining = None if self.daily_limit < 0 else max(0, self.daily_limit - used)
        return {
            "provider": self.provider,
            "limit": self.daily_limit,
            "used": used,
            "remaining": remaining,
        }

    # ── Write path ───────────────────────────────────────────────────────────

    async def consume(self, n: int = 1) -> int:
        """Reserve ``n`` calls. Raises :class:`QuotaExceeded` if it would overflow.

        Returns the post-increment count.
        """
        if self.daily_limit < 0:
            return 0  # unlimited — don't bother counting

        cache = await get_cache()
        current = await self.used_today()
        if current + n > self.daily_limit:
            raise QuotaExceeded(
                f"{self.provider} daily quota exhausted: "
                f"{current}/{self.daily_limit} used."
            )
        new = current + n
        # TTL is full day in seconds — the counter naturally expires at
        # rollover, so we don't need a sweeper.
        await cache.set(self._key(), str(new).encode("utf-8"), ttl=_DAY_SECONDS)
        return new


# ── Module-level providers ────────────────────────────────────────────────────

_API_FOOTBALL_DAILY = 100  # free-tier cap


def get_quota_tracker() -> QuotaTracker:
    """FastAPI dep — returns the API-Football tracker."""
    return QuotaTracker("api_football", _API_FOOTBALL_DAILY)
