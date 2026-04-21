"""Pydantic DTOs for /api/v1/auth endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from football_ai.models.user import SubscriptionTier, UserRole


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(None, max_length=255)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    """Safe public view of a user — never includes hashed_password."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    full_name: str | None = None
    role: UserRole
    subscription_tier: SubscriptionTier
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login_at: datetime | None = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until access token expires


class RefreshRequest(BaseModel):
    refresh_token: str


class RoleUpdateRequest(BaseModel):
    """Internal / test-only — lets the auth flow test upgrade a user's role."""

    role: UserRole
