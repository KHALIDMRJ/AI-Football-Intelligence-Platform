"""
Auth edge-case coverage.

``test_auth_flow`` already walks the happy path + wrong-password + RBAC
gate. This file closes the ``is_active`` gates that matter for a football
platform: a banned scout must not be able to refresh or log in, otherwise
a revoked subscription keeps working until the old access token expires.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from football_ai.crud import user as user_crud

EMAIL = "banned-scout@example.com"
PASSWORD = "long-enough-password-42"


@pytest_asyncio.fixture
async def registered_and_logged_in(client, test_session_factory):
    """Register a user, log in, then deactivate them directly in the DB."""
    await client.post(
        "/api/v1/auth/register",
        json={"email": EMAIL, "password": PASSWORD},
    )
    login = await client.post(
        "/api/v1/auth/login",
        data={"username": EMAIL, "password": PASSWORD},
    )
    tokens = login.json()

    async with test_session_factory() as session:
        user = await user_crud.get_by_email(session, EMAIL)
        user.is_active = False
        await session.commit()

    return tokens


@pytest.mark.asyncio
async def test_login_rejects_deactivated_user(client, test_session_factory):
    """A banned user must get 401 on login, not a fresh session."""
    await client.post(
        "/api/v1/auth/register",
        json={"email": EMAIL, "password": PASSWORD},
    )
    async with test_session_factory() as session:
        user = await user_crud.get_by_email(session, EMAIL)
        user.is_active = False
        await session.commit()

    resp = await client.post(
        "/api/v1/auth/login",
        data={"username": EMAIL, "password": PASSWORD},
    )
    assert resp.status_code == 401
    assert "disabled" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_refresh_rejects_deactivated_user(registered_and_logged_in, client):
    """Refresh must re-check ``is_active`` — cached claims can't bypass a ban."""
    tokens = registered_and_logged_in
    resp = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_bearer_requests_reject_deactivated_user(registered_and_logged_in, client):
    """get_current_user path re-reads the DB; a ban kicks them on the next request."""
    tokens = registered_and_logged_in
    resp = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_refresh_rejects_malformed_token(client):
    bad = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": "not-a-real-jwt"},
    )
    assert bad.status_code == 401
