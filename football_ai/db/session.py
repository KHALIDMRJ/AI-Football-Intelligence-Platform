"""
Async SQLAlchemy engine, sessionmaker, and FastAPI dependency.

``get_db`` is the canonical way to acquire a session inside a request
handler; it ensures the session is closed and any pending transaction
is rolled back on unhandled exceptions.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from football_ai.config import platform_settings


def _build_engine() -> AsyncEngine:
    """
    Build the async engine once per process.

    Football-domain reason for ``pool_pre_ping=True``: live-match pollers
    hold long-lived connections; PG kills idle TCP after a while. Pre-ping
    turns a stale-connection 500 into a silent reconnect before the query
    runs.
    """
    kwargs: dict = {
        "echo": platform_settings.database_echo,
        "future": True,
    }
    if not platform_settings.is_sqlite:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = 10
        kwargs["max_overflow"] = 20

    return create_async_engine(platform_settings.database_url, **kwargs)


engine: AsyncEngine = _build_engine()

async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency yielding an ``AsyncSession``.

    Usage:
        async def handler(db: AsyncSession = Depends(get_db)): ...
    """
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Close all pooled connections — call from app shutdown lifespan."""
    await engine.dispose()
