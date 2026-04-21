"""
Integration coverage for Phase-10 rate limiting + security headers.

End-to-end paths we cover:

* Security headers land on ordinary responses.
* A pro-analyst's :mod:`role_rate_limit` on ``POST /predictions/match``
  drains after ``pro_per_minute`` hits in quick succession and returns
  429 with ``Retry-After`` + ``X-RateLimit-*``.
* An admin's limit is far higher — the same burst that 429s a pro sails
  through for an admin.
* A malformed oversized body is rejected with 413 before the endpoint
  handler ever runs.
"""

from __future__ import annotations  # noqa: I001

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from football_ai.models.match import Match, MatchStatus
from football_ai.models.team import Team
from football_ai.models.user import UserRole
from football_ai.ratelimit.dependencies import _reset_buckets_for_tests


@pytest.fixture(autouse=True)
def _fresh_buckets():
    """Each test starts with a clean in-process bucket registry."""
    _reset_buckets_for_tests()
    yield
    _reset_buckets_for_tests()


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def scheduled_match(test_session_factory) -> uuid.UUID:
    async with test_session_factory() as s:
        home = Team(name="Arsenal", country="England", league="Premier League")
        away = Team(name="Chelsea", country="England", league="Premier League")
        s.add_all([home, away])
        await s.commit()
        await s.refresh(home)
        await s.refresh(away)

        m = Match(
            home_team_id=home.id,
            away_team_id=away.id,
            match_date=datetime.now(UTC) + timedelta(days=1),
            season="2025-26",
            league="Premier League",
            status=MatchStatus.scheduled,
        )
        s.add(m)
        await s.commit()
        await s.refresh(m)
        return m.id


async def _register_and_login(client, *, email: str, role: UserRole = UserRole.pro_analyst):
    """Register a new user, then upgrade them via the test-only PATCH route."""
    pw = "long-enough-password-42"
    await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": pw},
    )
    login = await client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": pw},
    )
    tokens = login.json()

    if role != UserRole.free_user:
        await client.patch(
            "/api/v1/auth/me/role",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            json={"role": role.value},
        )
        login = await client.post(
            "/api/v1/auth/login",
            data={"username": email, "password": pw},
        )
        tokens = login.json()

    return {"Authorization": f"Bearer {tokens['access_token']}"}


# ── Security headers ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_security_headers_present_on_responses(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


# ── Rate limiting ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pro_analyst_hits_429_after_limit(client, scheduled_match):
    """Pro limit is 30/min on the predict endpoint — the 31st call is rejected."""
    headers = await _register_and_login(
        client, email="pro-burst@example.com", role=UserRole.pro_analyst
    )
    # The predictor will fail (no trained model in tests), but the rate
    # limiter runs BEFORE the endpoint handler, so we count every
    # attempt regardless of its downstream status code.
    path = f"/api/v1/predictions/match/{scheduled_match}"
    statuses: list[int] = []
    for _ in range(31):
        r = await client.post(path, headers=headers)
        statuses.append(r.status_code)

    # Exactly the 31st must be 429 — prior 30 drain the bucket and the
    # handler returns whatever (likely 409 no-model or 422 inference fail).
    assert statuses[:30].count(429) == 0
    assert statuses[30] == 429


@pytest.mark.asyncio
async def test_429_response_carries_rate_limit_headers(client, scheduled_match):
    headers = await _register_and_login(
        client, email="pro-headers@example.com", role=UserRole.pro_analyst
    )
    path = f"/api/v1/predictions/match/{scheduled_match}"
    for _ in range(30):
        await client.post(path, headers=headers)
    r = await client.post(path, headers=headers)
    assert r.status_code == 429
    assert r.headers.get("Retry-After") is not None
    assert r.headers.get("X-RateLimit-Limit") == "30"
    assert r.headers.get("X-RateLimit-Remaining") == "0"


@pytest.mark.asyncio
async def test_admin_has_much_higher_budget(client, scheduled_match):
    """An admin firing 40 calls in a burst must not trip the 30/min pro cap."""
    headers = await _register_and_login(
        client, email="boss@example.com", role=UserRole.admin
    )
    path = f"/api/v1/predictions/match/{scheduled_match}"
    statuses: list[int] = []
    for _ in range(40):
        r = await client.post(path, headers=headers)
        statuses.append(r.status_code)
    assert statuses.count(429) == 0


# ── Body size guard ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_oversized_body_rejected_with_413(client):
    """The 5 MiB guard rejects payloads before FastAPI parses them."""
    huge = "x" * (6 * 1024 * 1024)  # 6 MiB > 5 MiB cap
    resp = await client.post(
        "/api/v1/auth/register",
        content=huge,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413
    assert resp.json()["type"] == "payload_too_large"
