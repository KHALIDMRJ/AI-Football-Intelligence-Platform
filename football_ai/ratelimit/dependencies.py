"""
FastAPI dependency factories for rate-limiting endpoints.

Usage
-----

Simple per-minute cap (same limit for everyone, keyed by user_id if
authenticated, else by client IP)::

    @router.get("/predictions/live",
                dependencies=[Depends(rate_limit(per_minute=60))])
    async def live_predictions(...): ...

Role-aware cap (football-domain pattern — free users pinched, pros
get headroom, admins unlimited)::

    @router.get("/predictions/live",
                dependencies=[Depends(role_rate_limit(
                    free_per_minute=20,
                    pro_per_minute=120,
                    scout_per_minute=240,
                ))])
    async def live_predictions(current: CurrentUser, ...): ...

Why a factory, not a decorator: FastAPI ``Depends`` expects a callable
it can introspect for other deps (Request, current user). Keeping the
limiter as a dep lets us compose ``require_role`` + ``rate_limit`` in
the endpoint signature naturally.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Depends, Request

from football_ai.api.dependencies import get_current_user
from football_ai.cache.backends import Cache
from football_ai.cache.factory import get_cache
from football_ai.core.exceptions import RateLimitError
from football_ai.models.user import User, UserRole

from .token_bucket import TokenBucket


@dataclass(frozen=True)
class RateLimitPolicy:
    """A per-minute cap + namespace pair. Immutable so we can hash it."""

    per_minute: int
    namespace: str
    burst: int | None = None  # defaults to per_minute when None

    @property
    def capacity(self) -> float:
        return float(self.burst if self.burst is not None else self.per_minute)

    @property
    def refill_per_second(self) -> float:
        return self.per_minute / 60.0


# A small registry so we reuse one :class:`TokenBucket` per policy instead
# of constructing one per request. The cache backend is fetched lazily on
# first use because :func:`get_cache` is async.
_buckets: dict[RateLimitPolicy, TokenBucket] = {}


async def _bucket_for(policy: RateLimitPolicy, cache: Cache) -> TokenBucket:
    existing = _buckets.get(policy)
    if existing is not None:
        return existing
    bucket = TokenBucket(
        cache,
        capacity=policy.capacity,
        refill_per_second=policy.refill_per_second,
        namespace=f"rl:{policy.namespace}",
    )
    _buckets[policy] = bucket
    return bucket


def _reset_buckets_for_tests() -> None:
    """Clear the module-level registry so tests don't share buckets."""
    _buckets.clear()


def _client_identity(request: Request, user: User | None) -> str:
    """Prefer user_id for logged-in callers, else the client IP.

    The IP path handles ``X-Forwarded-For`` — only the left-most hop is
    kept, matching how the observability middleware reports client IP.
    """
    if user is not None:
        return f"user:{user.id}"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    if request.client:
        return f"ip:{request.client.host}"
    return "ip:unknown"


def rate_limit(
    *,
    per_minute: int,
    burst: int | None = None,
    namespace: str | None = None,
) -> Callable:
    """Build a FastAPI dependency that enforces a fixed per-minute budget.

    No auth is required — anonymous callers are keyed by client IP.
    If the caller is authenticated, we still reach them by user id so a
    single user moving between IPs can't double their budget.
    """
    policy = RateLimitPolicy(
        per_minute=per_minute,
        burst=burst,
        namespace=namespace or f"fixed:{per_minute}",
    )

    async def _guard(request: Request) -> None:
        cache = await get_cache()
        bucket = await _bucket_for(policy, cache)
        # Opportunistic user lookup: when the request carries a valid
        # bearer token we key on user id, else we key on IP. We never
        # *require* a token here — that would push every unauth caller
        # into a 401 before we ever check the limit.
        user: User | None = getattr(request.state, "current_user", None)
        identity = _client_identity(request, user)
        result = await bucket.consume(identity)
        if not result.allowed:
            raise RateLimitError(
                "Rate limit exceeded. Try again shortly.",
                retry_after=result.retry_after,
                limit=policy.per_minute,
                remaining=result.remaining,
            )

    return _guard


def role_rate_limit(
    *,
    free_per_minute: int,
    pro_per_minute: int | None = None,
    scout_per_minute: int | None = None,
    admin_per_minute: int | None = None,
    namespace: str | None = None,
) -> Callable:
    """Per-minute cap that scales with ``user.role``.

    Missing tiers inherit the next-highest configured tier (so you can
    set just ``free_per_minute=20, pro_per_minute=120`` and scout/admin
    will also land at 120). Set ``admin_per_minute=0`` to explicitly
    disable for admin, though that's almost never what you want.

    The caller must be authenticated — this dep depends on
    :func:`get_current_user`, so anonymous callers hit 401 before
    reaching the limiter.
    """
    ns = namespace or "role"
    # Pre-resolve the tier table once so the inner guard is O(1).
    free = free_per_minute
    pro = pro_per_minute if pro_per_minute is not None else free
    scout = scout_per_minute if scout_per_minute is not None else pro
    admin = admin_per_minute if admin_per_minute is not None else scout
    per_role: dict[UserRole, int] = {
        UserRole.free_user: free,
        UserRole.pro_analyst: pro,
        UserRole.club_scout: scout,
        UserRole.admin: admin,
    }

    async def _guard(
        request: Request,
        user: User = Depends(get_current_user),
    ) -> None:
        per_minute = per_role[user.role]
        policy = RateLimitPolicy(
            per_minute=per_minute,
            namespace=f"{ns}:{user.role.value}",
        )
        cache = await get_cache()
        bucket = await _bucket_for(policy, cache)
        identity = f"user:{user.id}"
        result = await bucket.consume(identity)
        if not result.allowed:
            raise RateLimitError(
                f"Rate limit for role '{user.role.value}' exceeded.",
                retry_after=result.retry_after,
                limit=per_minute,
                remaining=result.remaining,
            )

    return _guard
