"""
Auth endpoints (v1) — Phase 3.

Routes
------
POST /auth/register   — creates a free_user; 409 on duplicate email
POST /auth/login      — OAuth2 password flow → access + refresh tokens
POST /auth/refresh    — exchange a refresh token for a fresh access token
GET  /auth/me         — current user (role + subscription_tier + flags)
PATCH /auth/me/role   — test-only: self-promote (guarded by ENVIRONMENT != production)

Football-domain rationale for the token split is in ``core/security.py``:
access tokens carry ``role`` so live-match polling stays DB-free, refresh
tokens re-read the DB so a downgraded scout can't keep a premium session.

The ``PATCH /auth/me/role`` shortcut exists only for the integration test
``register → login → 403 → upgrade → 200`` (Phase 9). Production would
replace it with an admin-only endpoint.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.api.dependencies import CurrentUser, get_db
from football_ai.api.schemas.auth import (
    RefreshRequest,
    RoleUpdateRequest,
    TokenResponse,
    UserCreate,
    UserOut,
)
from football_ai.config import platform_settings
from football_ai.core.exceptions import AuthError, FootballAPIError
from football_ai.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from football_ai.crud import user as user_crud
from football_ai.ratelimit import rate_limit

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_response(user) -> TokenResponse:
    """Mint both tokens for a user + include TTL so clients can schedule refresh."""
    return TokenResponse(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
        expires_in=platform_settings.access_token_expires_minutes * 60,
    )


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new free_user account",
    dependencies=[Depends(rate_limit(per_minute=5, burst=10, namespace="auth_register"))],
)
async def register(
    payload: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserOut:
    existing = await user_crud.get_by_email(db, payload.email)
    if existing is not None:
        # 409 Conflict — matches semantic of "resource already exists".
        raise FootballAPIError(
            "A user with this email is already registered.",
            status_code=status.HTTP_409_CONFLICT,
        )
    user = await user_crud.create_user(
        db,
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
    )
    return UserOut.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Exchange email + password for access + refresh tokens",
    dependencies=[Depends(rate_limit(per_minute=10, burst=20, namespace="auth_login"))],
)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    # OAuth2PasswordRequestForm uses "username" — we treat it as email.
    user = await user_crud.get_by_email(db, form.username)
    if user is None or not verify_password(form.password, user.hashed_password):
        # Never leak which of email/password was wrong → same 401 for both.
        raise AuthError("Invalid email or password.")
    if not user.is_active:
        raise AuthError("User account is disabled.")

    await user_crud.update_last_login(db, user)
    return _token_response(user)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate an access token using a valid refresh token",
    dependencies=[Depends(rate_limit(per_minute=20, burst=30, namespace="auth_refresh"))],
)
async def refresh(
    payload: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    claims = decode_token(payload.refresh_token, expected_type="refresh")
    user = await user_crud.get_by_id(db, claims["sub"])
    if user is None or not user.is_active:
        # DB re-read catches: deleted account, deactivated account, banned.
        raise AuthError("Refresh token no longer valid.")
    return _token_response(user)


@router.get(
    "/me",
    response_model=UserOut,
    summary="Return the currently authenticated user",
)
async def me(current: CurrentUser) -> UserOut:
    return UserOut.model_validate(current)


@router.patch(
    "/me/role",
    response_model=UserOut,
    summary="Self-upgrade role (test/dev only — blocked in production)",
)
async def update_own_role(
    payload: RoleUpdateRequest,
    current: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserOut:
    if platform_settings.environment == "production":
        raise AuthError(
            "Self-upgrade is disabled in production.",
            status_code=status.HTTP_403_FORBIDDEN,
        )
    user = await user_crud.update_role(db, current, role=payload.role)
    return UserOut.model_validate(user)
