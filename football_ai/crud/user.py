"""User CRUD — thin typed helpers over async SQLAlchemy sessions."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.core.security import hash_password
from football_ai.models.user import SubscriptionTier, User, UserRole


async def get_by_id(db: AsyncSession, user_id: uuid.UUID) -> User | None:
    return await db.get(User, user_id)


async def get_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email.lower()))
    return result.scalar_one_or_none()


async def create_user(
    db: AsyncSession,
    *,
    email: str,
    password: str,
    full_name: str | None = None,
    role: UserRole = UserRole.free_user,
    subscription_tier: SubscriptionTier = SubscriptionTier.free,
) -> User:
    user = User(
        email=email.lower(),
        hashed_password=hash_password(password),
        full_name=full_name,
        role=role,
        subscription_tier=subscription_tier,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def update_last_login(db: AsyncSession, user: User) -> None:
    user.last_login_at = datetime.now(UTC)
    await db.commit()


async def update_role(
    db: AsyncSession,
    user: User,
    *,
    role: UserRole,
    subscription_tier: SubscriptionTier | None = None,
) -> User:
    user.role = role
    if subscription_tier is not None:
        user.subscription_tier = subscription_tier
    await db.commit()
    await db.refresh(user)
    return user
