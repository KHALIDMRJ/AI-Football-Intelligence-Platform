"""
Unit tests for the API-Football client.

We mock upstream HTTP with :mod:`respx` so we can assert on request
shape (headers, params) without touching the network.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from football_ai.core.exceptions import ExternalAPIError
from football_ai.external.api_football import APIFootballClient
from football_ai.external.quota import QuotaTracker


@pytest.fixture
def client_factory():
    """Factory that yields a fully-mocked APIFootballClient."""
    async def _make(*, api_key: str = "test-key", quota_limit: int = 100) -> APIFootballClient:
        http = httpx.AsyncClient()
        return APIFootballClient(
            api_key=api_key,
            base_url="https://example.invalid",
            quota=QuotaTracker("api_football_test", daily_limit=quota_limit),
            http_client=http,
        )
    return _make


# ── Configuration / 503 ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_key_raises_503(client_factory):
    client = await client_factory(api_key="")
    with pytest.raises(ExternalAPIError) as exc:
        await client.list_fixtures(league=39, season=2024)
    assert exc.value.status_code == 503
    await client.aclose()


# ── Happy path ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@respx.mock
async def test_list_fixtures_returns_response_array(client_factory):
    respx.get("https://example.invalid/fixtures").mock(
        return_value=httpx.Response(
            200,
            json={"response": [{"fixture": {"id": 1}}, {"fixture": {"id": 2}}]},
        )
    )
    client = await client_factory()
    fixtures = await client.list_fixtures(league=39, season=2024)
    assert [f["fixture"]["id"] for f in fixtures] == [1, 2]

    # Header shape must match the provider's contract.
    call = respx.calls.last
    assert call.request.headers["x-apisports-key"] == "test-key"
    assert call.request.url.params["league"] == "39"
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_get_squad_flattens_players_block(client_factory):
    respx.get("https://example.invalid/players/squads").mock(
        return_value=httpx.Response(
            200,
            json={
                "response": [
                    {
                        "team": {"id": 42},
                        "players": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
                    }
                ]
            },
        )
    )
    client = await client_factory()
    players = await client.get_squad(team=42)
    assert [p["id"] for p in players] == [1, 2]
    await client.aclose()


# ── Failure paths ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_quota_exhaustion_returns_429(client_factory):
    client = await client_factory(quota_limit=0)
    with pytest.raises(ExternalAPIError) as exc:
        await client.list_fixtures(league=39, season=2024)
    assert exc.value.status_code == 429
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_upstream_4xx_becomes_502(client_factory):
    respx.get("https://example.invalid/fixtures").mock(
        return_value=httpx.Response(403, text="forbidden")
    )
    client = await client_factory()
    with pytest.raises(ExternalAPIError) as exc:
        await client.list_fixtures(league=39, season=2024)
    assert exc.value.status_code == 502
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_upstream_5xx_retries_then_fails(client_factory):
    route = respx.get("https://example.invalid/fixtures").mock(
        return_value=httpx.Response(503, text="down")
    )
    client = await client_factory()
    with pytest.raises(ExternalAPIError) as exc:
        await client.list_fixtures(league=39, season=2024)
    assert exc.value.status_code == 502
    # We retry twice after the initial attempt → 3 total calls.
    assert route.call_count == 3
    await client.aclose()
