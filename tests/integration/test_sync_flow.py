"""
Integration coverage for the external-sync admin endpoints.

Approach
--------
* We inject an :class:`APIFootballClient` pre-wired with a fake httpx
  client (via :mod:`respx`) into the FastAPI dependency graph. That
  exercises the full admin → service → client → upsert path without a
  live network dependency.
* All tests drive via the same ``client`` fixture the other integration
  tests use, so auth, RBAC, and DB wiring are identical.
"""

from __future__ import annotations  # noqa: I001

import uuid

import httpx
import pytest
import respx

from football_ai.api.main import app
from football_ai.api.v1.endpoints.admin import get_api_football_client
from football_ai.external.api_football import APIFootballClient
from football_ai.external.quota import QuotaTracker
from football_ai.models.team import Team


async def _register_admin(client, email: str = "ops@example.com") -> str:
    await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "password-123"},
    )
    r = await client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": "password-123"},
    )
    token = r.json()["access_token"]
    r = await client.patch(
        "/api/v1/auth/me/role",
        headers={"Authorization": f"Bearer {token}"},
        json={"role": "admin"},
    )
    assert r.status_code == 200
    return token


def _fake_fixture_payload(api_id: int, home_id: int, away_id: int) -> dict:
    return {
        "fixture": {
            "id": api_id,
            "date": "2024-08-12T17:30:00+00:00",
            "status": {"short": "NS"},
            "venue": {"name": "Emirates"},
            "referee": "M. Oliver",
        },
        "league": {"id": 39, "name": "Premier League"},
        "teams": {
            "home": {"id": home_id, "name": f"Team {home_id}", "logo": None},
            "away": {"id": away_id, "name": f"Team {away_id}", "logo": None},
        },
        "goals": {"home": None, "away": None},
    }


def _inject_client(http: httpx.AsyncClient) -> APIFootballClient:
    """Build a pre-configured client and register it as the FastAPI dep."""
    external = APIFootballClient(
        api_key="test-key",
        base_url="https://example.invalid",
        quota=QuotaTracker("api_football_test", daily_limit=100),
        http_client=http,
    )
    app.dependency_overrides[get_api_football_client] = lambda: external
    return external


# ── /admin/sync/status ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sync_status_reports_configured_flag(client):
    token = await _register_admin(client, "status@example.com")
    http = httpx.AsyncClient()
    external = _inject_client(http)
    try:
        r = await client.get(
            "/api/v1/admin/sync/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["api_football"]["configured"] is True
        assert body["api_football"]["quota"]["limit"] == 100
    finally:
        app.dependency_overrides.pop(get_api_football_client, None)
        await external.aclose()
        await http.aclose()


# ── /admin/sync/fixtures — happy path + idempotence ─────────────────────────

@pytest.mark.asyncio
async def test_sync_fixtures_creates_then_reruns_skip(client):
    token = await _register_admin(client, "fx@example.com")
    http = httpx.AsyncClient()
    external = _inject_client(http)

    payload = {
        "response": [
            _fake_fixture_payload(1001, home_id=500, away_id=600),
            _fake_fixture_payload(1002, home_id=500, away_id=700),
        ]
    }

    try:
        with respx.mock(base_url="https://example.invalid") as rx:
            rx.get("/fixtures").mock(return_value=httpx.Response(200, json=payload))

            r = await client.post(
                "/api/v1/admin/sync/fixtures?league=39&season=2024",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["fetched"] == 2
            assert body["created"] == 2
            assert body["updated"] == 0

            # Same payload → second call must not create anything new.
            r = await client.post(
                "/api/v1/admin/sync/fixtures?league=39&season=2024",
                headers={"Authorization": f"Bearer {token}"},
            )
            body = r.json()
            assert body["created"] == 0
            assert body["updated"] == 0
            assert body["skipped"] == 2
    finally:
        app.dependency_overrides.pop(get_api_football_client, None)
        await external.aclose()
        await http.aclose()


@pytest.mark.asyncio
async def test_sync_fixtures_updates_when_score_changes(client):
    token = await _register_admin(client, "fx-score@example.com")
    http = httpx.AsyncClient()
    external = _inject_client(http)

    first = _fake_fixture_payload(2001, home_id=10, away_id=20)
    second = _fake_fixture_payload(2001, home_id=10, away_id=20)
    second["fixture"]["status"]["short"] = "FT"
    second["goals"] = {"home": 2, "away": 1}

    try:
        with respx.mock(base_url="https://example.invalid") as rx:
            # First call returns pre-game; second returns finished.
            rx.get("/fixtures").mock(
                side_effect=[
                    httpx.Response(200, json={"response": [first]}),
                    httpx.Response(200, json={"response": [second]}),
                ]
            )

            await client.post(
                "/api/v1/admin/sync/fixtures?league=39&season=2024",
                headers={"Authorization": f"Bearer {token}"},
            )
            r = await client.post(
                "/api/v1/admin/sync/fixtures?league=39&season=2024",
                headers={"Authorization": f"Bearer {token}"},
            )
            body = r.json()
            assert body["updated"] == 1
            assert body["created"] == 0
    finally:
        app.dependency_overrides.pop(get_api_football_client, None)
        await external.aclose()
        await http.aclose()


# ── /admin/sync/squad ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sync_squad_requires_api_football_id(client, test_session_factory):
    token = await _register_admin(client, "sq1@example.com")
    http = httpx.AsyncClient()
    external = _inject_client(http)

    async with test_session_factory() as s:
        team = Team(name="No-Provider FC", api_football_id=None)
        s.add(team)
        await s.commit()
        await s.refresh(team)

    try:
        r = await client.post(
            f"/api/v1/admin/sync/squad?team_id={team.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 409, r.text
    finally:
        app.dependency_overrides.pop(get_api_football_client, None)
        await external.aclose()
        await http.aclose()


@pytest.mark.asyncio
async def test_sync_squad_creates_players(client, test_session_factory):
    token = await _register_admin(client, "sq2@example.com")
    http = httpx.AsyncClient()
    external = _inject_client(http)

    async with test_session_factory() as s:
        team = Team(name="Mapped FC", api_football_id=999)
        s.add(team)
        await s.commit()
        await s.refresh(team)

    squad = {
        "response": [
            {
                "team": {"id": 999},
                "players": [
                    {"id": 1, "name": "Goalie One", "position": "Goalkeeper", "number": 1},
                    {"id": 2, "name": "Striker Two", "position": "Attacker", "number": 9},
                ],
            }
        ]
    }

    try:
        with respx.mock(base_url="https://example.invalid") as rx:
            rx.get("/players/squads").mock(return_value=httpx.Response(200, json=squad))
            r = await client.post(
                f"/api/v1/admin/sync/squad?team_id={team.id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["created"] == 2
    finally:
        app.dependency_overrides.pop(get_api_football_client, None)
        await external.aclose()
        await http.aclose()


@pytest.mark.asyncio
async def test_sync_squad_unknown_team_404(client):
    token = await _register_admin(client, "sq3@example.com")
    http = httpx.AsyncClient()
    external = _inject_client(http)
    try:
        r = await client.post(
            f"/api/v1/admin/sync/squad?team_id={uuid.uuid4()}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 404
    finally:
        app.dependency_overrides.pop(get_api_football_client, None)
        await external.aclose()
        await http.aclose()


# ── 503 path when no key ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sync_fixtures_503_when_unconfigured(client):
    """No API key → client refuses with 503 (same shape as scouting)."""
    token = await _register_admin(client, "nokey@example.com")
    http = httpx.AsyncClient()
    external = APIFootballClient(
        api_key="",
        base_url="https://example.invalid",
        quota=QuotaTracker("api_football_test", daily_limit=100),
        http_client=http,
    )
    app.dependency_overrides[get_api_football_client] = lambda: external
    try:
        r = await client.post(
            "/api/v1/admin/sync/fixtures?league=39&season=2024",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 503, r.text
    finally:
        app.dependency_overrides.pop(get_api_football_client, None)
        await external.aclose()
        await http.aclose()
