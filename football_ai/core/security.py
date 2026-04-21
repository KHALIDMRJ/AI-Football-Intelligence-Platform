"""
Security primitives: password hashing (bcrypt) + JWT encode/decode.

Token design
------------
Two JWTs, same secret, distinguished by a ``type`` claim:

* **access**  (short-lived, default 30m) — carries ``role`` + ``tier`` so
  role-gate dependencies don't hit the DB on every request.
* **refresh** (long-lived, default 14d)  — carries only ``sub`` and
  ``type=refresh``; the server re-reads the user row before minting a new
  access token so a banned / downgraded user is dropped within one refresh
  cycle.

Football-domain reason for embedding ``role`` in the access token: match-
data endpoints are latency-sensitive (Phase 6 live-match polling runs a
check every 30s per subscriber). A DB round-trip on every call to
``require_role`` would compound. Refresh re-reads the DB so downgrades
still propagate.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import bcrypt
from jose import JWTError, jwt

from football_ai.config import platform_settings
from football_ai.core.exceptions import AuthError
from football_ai.models.user import User

TokenType = Literal["access", "refresh"]

# bcrypt caps the input at 72 bytes. We call bcrypt directly (rather than via
# passlib) because passlib 1.7 is incompatible with bcrypt 4.x — it probes for
# a private ``__about__`` attribute that was removed, and falls over on the
# 72-byte boundary check. The ``User.password`` schema already limits inputs
# to 128 chars, so we truncate explicitly and warn in docs.
_BCRYPT_MAX_BYTES = 72


# ── Password hashing ─────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """bcrypt-hash a plaintext password. Uses a 12-round cost factor."""
    pwd_bytes = plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(pwd_bytes, bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time compare between plaintext and stored hash."""
    try:
        pwd_bytes = plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]
        return bcrypt.checkpw(pwd_bytes, hashed.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed hash in DB or non-string input — treat as failed verify.
        return False


# ── JWT encoding ─────────────────────────────────────────────────────────────

def _encode(payload: dict[str, Any], expires_delta: timedelta) -> str:
    now = datetime.now(UTC)
    payload = {
        **payload,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    return jwt.encode(
        payload,
        platform_settings.secret_key,
        algorithm=platform_settings.jwt_algorithm,
    )


def create_access_token(user: User) -> str:
    """Short-lived token carrying role + tier for cheap RBAC checks."""
    return _encode(
        {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role.value,
            "tier": user.subscription_tier.value,
            "type": "access",
        },
        timedelta(minutes=platform_settings.access_token_expires_minutes),
    )


def create_refresh_token(user: User) -> str:
    """Long-lived token — ``sub`` only; refresh re-reads the DB for role."""
    return _encode(
        {"sub": str(user.id), "type": "refresh"},
        timedelta(days=platform_settings.refresh_token_expires_days),
    )


# ── JWT decoding ─────────────────────────────────────────────────────────────

def decode_token(token: str, *, expected_type: TokenType) -> dict[str, Any]:
    """
    Decode and validate a JWT.

    Raises
    ------
    AuthError
        If the token is expired, malformed, signed with the wrong key, or of
        the wrong ``type`` (access where refresh expected, or vice versa).
    """
    try:
        payload = jwt.decode(
            token,
            platform_settings.secret_key,
            algorithms=[platform_settings.jwt_algorithm],
        )
    except JWTError as exc:
        raise AuthError("Invalid or expired token.") from exc

    if payload.get("type") != expected_type:
        raise AuthError(f"Expected {expected_type} token.")

    sub = payload.get("sub")
    if not sub:
        raise AuthError("Token missing subject claim.")

    # Normalise `sub` to a UUID for caller convenience.
    try:
        payload["sub"] = uuid.UUID(sub)
    except (ValueError, TypeError) as exc:
        raise AuthError("Token subject is not a valid user id.") from exc

    return payload
