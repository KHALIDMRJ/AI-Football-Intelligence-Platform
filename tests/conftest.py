"""
Top-level pytest fixtures.

The async HTTP fixtures below run the FastAPI app against a disposable
in-memory SQLite database (shared across the async engine via
``?cache=shared`` + ``StaticPool``), so integration tests touch every
layer — routing, deps, ORM, schemas — without needing a live Postgres.

We intentionally do NOT reuse the process-wide ``async_session_factory``
from ``football_ai.db.session``: tests bind to their own engine and
override the ``get_db`` dependency so production state (including the
real ``football_ai.db`` file) is never touched.
"""

from __future__ import annotations  # noqa: I001

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

# Importing the models package registers every table with Base.metadata —
# this must happen before create_all() so the test schema is complete.
import football_ai.models  # noqa: F401
from football_ai.api.dependencies import get_db
from football_ai.api.main import app as fastapi_app
from football_ai.cache.backends import InMemoryCache
from football_ai.cache.factory import override_cache_for_tests, reset_cache_for_tests
from football_ai.db.base import Base


@pytest_asyncio.fixture(autouse=True)
async def _isolated_cache():
    """Force every test to start with a fresh in-memory cache.

    Without this, the process-wide cache singleton would carry stale
    payloads across tests — a GET test might see a previous test's POST
    response and pass for the wrong reason.
    """
    override_cache_for_tests(InMemoryCache())
    yield
    reset_cache_for_tests()


@pytest_asyncio.fixture
async def test_engine():
    """Fresh in-memory SQLite engine per test — clean slate, zero IO."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true",
        connect_args={"uri": True},
        poolclass=StaticPool,
        future=True,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest_asyncio.fixture
async def test_session_factory(test_engine):
    return async_sessionmaker(
        bind=test_engine,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


@pytest_asyncio.fixture
async def client(test_session_factory) -> AsyncGenerator[AsyncClient, None]:
    """httpx AsyncClient bound to the FastAPI app with an overridden get_db."""

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with test_session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    try:
        transport = ASGITransport(app=fastapi_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        fastapi_app.dependency_overrides.pop(get_db, None)
