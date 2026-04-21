"""
Auth + RBAC end-to-end flow.

Verifies the full Phase-3 loop against a live ASGI app:

    1. POST /auth/register        → 201, returns free_user profile
    2. POST /auth/register (dup)  → 409
    3. POST /auth/login (bad pw)  → 401
    4. POST /auth/login           → 200, access + refresh tokens
    5. GET  /auth/me              → 200, role=free_user
    6. GET  /predictions/ping     → 403 (pro_analyst gate rejects free_user)
    7. PATCH /auth/me/role → pro  → 200
    8. POST /auth/login (fresh token with new role)
    9. GET  /predictions/ping     → 200 (pro_analyst passes)
   10. POST /auth/refresh         → 200, new access token valid on /auth/me

No mocks. The test hits SQLAlchemy, bcrypt, and JWT encode/decode for real so
a regression in any of them fails here.
"""

from __future__ import annotations

import pytest

EMAIL = "striker@example.com"
PASSWORD = "s3cure-pass-42"
FULL_NAME = "Test Striker"


@pytest.mark.asyncio
async def test_full_auth_and_role_gate_flow(client):
    # 1. Register ------------------------------------------------------------
    r = await client.post(
        "/api/v1/auth/register",
        json={"email": EMAIL, "password": PASSWORD, "full_name": FULL_NAME},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["email"] == EMAIL
    assert body["role"] == "free_user"
    assert body["subscription_tier"] == "free"
    assert body["is_active"] is True
    assert "hashed_password" not in body  # never leak the hash

    # 2. Duplicate register → 409 -------------------------------------------
    dup = await client.post(
        "/api/v1/auth/register",
        json={"email": EMAIL, "password": PASSWORD},
    )
    assert dup.status_code == 409, dup.text

    # 3. Wrong password → 401 -----------------------------------------------
    bad = await client.post(
        "/api/v1/auth/login",
        data={"username": EMAIL, "password": "wrong-password"},
    )
    assert bad.status_code == 401, bad.text

    # 4. Login → tokens -----------------------------------------------------
    login = await client.post(
        "/api/v1/auth/login",
        data={"username": EMAIL, "password": PASSWORD},
    )
    assert login.status_code == 200, login.text
    tokens = login.json()
    assert tokens["token_type"] == "bearer"
    assert tokens["access_token"] and tokens["refresh_token"]
    assert tokens["expires_in"] > 0

    free_auth = {"Authorization": f"Bearer {tokens['access_token']}"}

    # 5. /me with free token -----------------------------------------------
    me = await client.get("/api/v1/auth/me", headers=free_auth)
    assert me.status_code == 200
    assert me.json()["role"] == "free_user"

    # 6. Pro-gated endpoint rejects free_user ------------------------------
    ping_free = await client.get("/api/v1/predictions/ping", headers=free_auth)
    assert ping_free.status_code == 403, ping_free.text

    # 7. Self-upgrade to pro_analyst (test-only route) ---------------------
    upgrade = await client.patch(
        "/api/v1/auth/me/role",
        headers=free_auth,
        json={"role": "pro_analyst"},
    )
    assert upgrade.status_code == 200, upgrade.text
    assert upgrade.json()["role"] == "pro_analyst"

    # 8. Old token still encodes the old role — mint a fresh one -----------
    login2 = await client.post(
        "/api/v1/auth/login",
        data={"username": EMAIL, "password": PASSWORD},
    )
    assert login2.status_code == 200
    pro_auth = {"Authorization": f"Bearer {login2.json()['access_token']}"}

    # 9. Pro-gated endpoint now accepts the same user ----------------------
    ping_pro = await client.get("/api/v1/predictions/ping", headers=pro_auth)
    assert ping_pro.status_code == 200, ping_pro.text
    assert ping_pro.json() == {"status": "ok", "role": "pro_analyst"}

    # 10. Refresh token exchange --------------------------------------------
    refreshed = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": login2.json()["refresh_token"]},
    )
    assert refreshed.status_code == 200, refreshed.text
    new_access = refreshed.json()["access_token"]
    me_again = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {new_access}"},
    )
    assert me_again.status_code == 200
    assert me_again.json()["email"] == EMAIL


@pytest.mark.asyncio
async def test_refresh_rejects_access_token(client):
    """Refresh endpoint must reject access tokens (different ``type`` claim)."""
    await client.post(
        "/api/v1/auth/register",
        json={"email": "winger@example.com", "password": "password-123"},
    )
    login = await client.post(
        "/api/v1/auth/login",
        data={"username": "winger@example.com", "password": "password-123"},
    )
    access = login.json()["access_token"]

    bad = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": access},  # deliberately wrong kind
    )
    assert bad.status_code == 401, bad.text


@pytest.mark.asyncio
async def test_protected_endpoints_require_bearer(client):
    """No token → 401 on both /auth/me and /predictions/ping."""
    me = await client.get("/api/v1/auth/me")
    assert me.status_code == 401

    ping = await client.get("/api/v1/predictions/ping")
    assert ping.status_code == 401
