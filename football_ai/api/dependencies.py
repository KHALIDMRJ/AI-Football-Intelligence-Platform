"""
Shared FastAPI dependencies.

get_db           — async SQLAlchemy session (Phase 2)
get_current_user — resolves the bearer token to a live User row (Phase 3)
require_role     — factory gating an endpoint on a minimum role (Phase 3)
get_redis        — Phase 6
check_quota      — Phase 6 (per-role request budget)

Role hierarchy (football-domain)
--------------------------------
admin (3) > club_scout (2) > pro_analyst (1) > free_user (0)

Higher tiers inherit everything the lower tiers can do, matching how a club
scout on the platform also wants to read the same public fixture data a
free user sees. ``require_role(pro_analyst)`` therefore admits pro_analyst,
club_scout, and admin — but rejects free_user with 403.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.core.exceptions import AuthError
from football_ai.core.security import decode_token
from football_ai.crud import user as user_crud
from football_ai.db.session import get_db
from football_ai.models.user import User, UserRole
from football_ai.observability.context import bind_user_id

__all__ = [
    "get_db",
    "get_current_user",
    "require_role",
    "oauth2_scheme",
    "CurrentUser",
]


# Points at the Phase-3 login endpoint so Swagger's Authorize button works.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# Rank roles so "≥ minimum" checks are a single int comparison.
_ROLE_ORDER: dict[UserRole, int] = {
    UserRole.free_user: 0,
    UserRole.pro_analyst: 1,
    UserRole.club_scout: 2,
    UserRole.admin: 3,
}


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Resolve the ``Authorization: Bearer <jwt>`` header to a live User row.

    Raises ``AuthError`` (401) on invalid/expired tokens, a missing user, or
    a deactivated account. We re-read the DB rather than trusting claims
    alone for ``is_active`` so a banned user is kicked on their very next
    request — JWT role claims are cached, but account status is not.
    """
    payload = decode_token(token, expected_type="access")
    user = await user_crud.get_by_id(db, payload["sub"])
    if user is None:
        raise AuthError("User for this token no longer exists.")
    if not user.is_active:
        raise AuthError("User account is disabled.")
    # Stamp the access log + downstream logs with the resolved user id.
    bind_user_id(str(user.id))
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_role(minimum: UserRole) -> Callable[[User], User]:
    """Build a dependency that enforces ``user.role >= minimum``.

    Usage::

        @router.get("/predictions/live",
                    dependencies=[Depends(require_role(UserRole.pro_analyst))])
    """
    required_rank = _ROLE_ORDER[minimum]

    async def _guard(user: CurrentUser) -> User:
        if _ROLE_ORDER[user.role] < required_rank:
            raise AuthError(
                f"This endpoint requires role '{minimum.value}' or higher.",
                status_code=status.HTTP_403_FORBIDDEN,
            )
        return user

    return _guard
