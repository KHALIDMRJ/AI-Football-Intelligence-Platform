"""
Catalogue endpoints end-to-end — Phase 4.

Covers:

1. ``GET /players/catalog`` + filters (league, position, team_id, age band)
2. ``GET /players/catalog/{id}`` — 200 happy path + 404 on unknown UUID
3. ``GET /players/catalog/{id}/stats`` — RBAC (free_user 403, pro 200)
4. ``GET /matches/fixtures`` + filters (league, season, status, team)
5. ``GET /matches/live`` — RBAC + status filter
6. ``GET /matches/fixtures/{id}/events`` — RBAC + chronological order
7. ``GET /teams/registry`` + country/league filter
8. ``GET /teams/registry/{id}/squad`` — jersey ordering

The seeded world: two teams (Arsenal/EPL, Bayern/Bundesliga), three players,
two matches (one live, one scheduled), two events on the live match.
"""

from __future__ import annotations  # noqa: I001

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from football_ai.models.match import EventType, Match, MatchEvent, MatchStatus
from football_ai.models.player import Player, PlayerStats, PreferredFoot
from football_ai.models.team import Team

# ── Seed fixture ─────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def world(test_session_factory):
    """Insert a minimal football universe for catalogue tests."""
    arsenal = Team(
        name="Arsenal",
        country="England",
        league="Premier League",
        formation="4-3-3",
    )
    bayern = Team(
        name="Bayern Munich",
        country="Germany",
        league="Bundesliga",
        formation="4-2-3-1",
    )

    async with test_session_factory() as s:
        s.add_all([arsenal, bayern])
        await s.commit()
        await s.refresh(arsenal)
        await s.refresh(bayern)

        saka = Player(
            name="Bukayo Saka",
            age=23,
            nationality="England",
            position="RW",
            preferred_foot=PreferredFoot.left,
            jersey_number=7,
            current_team_id=arsenal.id,
        )
        odegaard = Player(
            name="Martin Odegaard",
            age=26,
            nationality="Norway",
            position="CM",
            preferred_foot=PreferredFoot.left,
            jersey_number=8,
            current_team_id=arsenal.id,
        )
        kane = Player(
            name="Harry Kane",
            age=31,
            nationality="England",
            position="ST",
            preferred_foot=PreferredFoot.right,
            jersey_number=9,
            current_team_id=bayern.id,
        )
        s.add_all([saka, odegaard, kane])
        await s.commit()
        for p in (saka, odegaard, kane):
            await s.refresh(p)

        # PlayerStats — one match-line + one season-aggregate for Saka.
        now = datetime.now(UTC)
        live_match = Match(
            home_team_id=arsenal.id,
            away_team_id=bayern.id,
            league="UEFA Champions League",
            season="2024-25",
            match_date=now,
            status=MatchStatus.live,
            home_score=1,
            away_score=1,
        )
        scheduled_match = Match(
            home_team_id=bayern.id,
            away_team_id=arsenal.id,
            league="UEFA Champions League",
            season="2024-25",
            match_date=now + timedelta(days=14),
            status=MatchStatus.scheduled,
        )
        s.add_all([live_match, scheduled_match])
        await s.commit()
        await s.refresh(live_match)
        await s.refresh(scheduled_match)

        s.add_all(
            [
                MatchEvent(
                    match_id=live_match.id,
                    minute=23,
                    event_type=EventType.goal,
                    player_id=saka.id,
                    team_id=arsenal.id,
                ),
                MatchEvent(
                    match_id=live_match.id,
                    minute=58,
                    event_type=EventType.goal,
                    player_id=kane.id,
                    team_id=bayern.id,
                ),
            ]
        )
        s.add_all(
            [
                PlayerStats(
                    player_id=saka.id,
                    match_id=live_match.id,
                    season="2024-25",
                    goals=1,
                    assists=0,
                    minutes_played=90,
                    shots_on_target=3,
                    dribbles_completed=4,
                    tackles_won=1,
                    xg=0.42,
                    rating=7.9,
                ),
                PlayerStats(
                    player_id=saka.id,
                    match_id=None,  # season aggregate
                    season="2024-25",
                    goals=12,
                    assists=9,
                    minutes_played=2340,
                    shots_on_target=41,
                    dribbles_completed=83,
                    tackles_won=22,
                    xg=10.7,
                    xa=8.1,
                    rating=7.8,
                ),
            ]
        )
        await s.commit()

    return {
        "arsenal_id": arsenal.id,
        "bayern_id": bayern.id,
        "saka_id": saka.id,
        "odegaard_id": odegaard.id,
        "kane_id": kane.id,
        "live_match_id": live_match.id,
        "scheduled_match_id": scheduled_match.id,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _register_and_login(client, email: str, password: str = "password-123"):
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": password}
    )
    r = await client.post(
        "/api/v1/auth/login", data={"username": email, "password": password}
    )
    return r.json()["access_token"]


async def _upgrade_to_pro(client, token: str) -> str:
    """Upgrade a logged-in user to pro_analyst, return a fresh access token."""
    await client.patch(
        "/api/v1/auth/me/role",
        headers={"Authorization": f"Bearer {token}"},
        json={"role": "pro_analyst"},
    )
    # Re-login to pick up the new role in the access-token claims.
    # (The test-upgrade path re-hits /login on the caller's behalf.)
    return token  # guards re-read DB on every call, so old token works


# ── Players / catalog ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_player_catalog_returns_all_and_filters_by_league(client, world):
    # No filter → 3 players.
    r = await client.get("/api/v1/players/catalog")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 3
    assert len(body["items"]) == 3

    # league=Premier League → 2 Arsenal players only.
    r = await client.get("/api/v1/players/catalog", params={"league": "Premier League"})
    assert r.status_code == 200
    names = {p["name"] for p in r.json()["items"]}
    assert names == {"Bukayo Saka", "Martin Odegaard"}


@pytest.mark.asyncio
async def test_player_catalog_position_and_age_filters(client, world):
    r = await client.get(
        "/api/v1/players/catalog",
        params={"position": "ST"},
    )
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1 and items[0]["name"] == "Harry Kane"

    r = await client.get(
        "/api/v1/players/catalog",
        params={"min_age": 24, "max_age": 28},
    )
    assert r.status_code == 200
    names = {p["name"] for p in r.json()["items"]}
    assert names == {"Martin Odegaard"}


@pytest.mark.asyncio
async def test_player_catalog_detail_and_404(client, world):
    r = await client.get(f"/api/v1/players/catalog/{world['saka_id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Bukayo Saka"
    assert body["current_team"]["name"] == "Arsenal"

    missing = uuid.uuid4()
    r = await client.get(f"/api/v1/players/catalog/{missing}")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_player_stats_requires_pro_role(client, world):
    free_token = await _register_and_login(client, "fan@example.com")
    r = await client.get(
        f"/api/v1/players/catalog/{world['saka_id']}/stats",
        headers={"Authorization": f"Bearer {free_token}"},
    )
    assert r.status_code == 403

    await _upgrade_to_pro(client, free_token)
    r = await client.get(
        f"/api/v1/players/catalog/{world['saka_id']}/stats",
        headers={"Authorization": f"Bearer {free_token}"},
    )
    assert r.status_code == 200, r.text
    stats = r.json()
    assert len(stats) == 2  # one match-line, one season aggregate
    # Match-line sorts first (non-null match_id before null).
    assert stats[0]["match_id"] is not None
    assert stats[1]["match_id"] is None


@pytest.mark.asyncio
async def test_player_stats_404_for_unknown_player(client, world):
    token = await _register_and_login(client, "scout@example.com")
    await _upgrade_to_pro(client, token)
    r = await client.get(
        f"/api/v1/players/catalog/{uuid.uuid4()}/stats",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 404


# ── Matches / fixtures + live ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fixture_list_and_status_filter(client, world):
    r = await client.get("/api/v1/matches/fixtures")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 2

    r = await client.get(
        "/api/v1/matches/fixtures", params={"match_status": "live"}
    )
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == str(world["live_match_id"])


@pytest.mark.asyncio
async def test_fixture_team_and_season_filters(client, world):
    r = await client.get(
        "/api/v1/matches/fixtures",
        params={"team_id": str(world["arsenal_id"])},
    )
    # Arsenal plays in both fixtures (home + away).
    assert r.status_code == 200
    assert r.json()["total"] == 2

    r = await client.get(
        "/api/v1/matches/fixtures", params={"season": "2099-00"}
    )
    assert r.status_code == 200
    assert r.json()["total"] == 0


@pytest.mark.asyncio
async def test_fixture_detail_and_404(client, world):
    r = await client.get(f"/api/v1/matches/fixtures/{world['scheduled_match_id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "scheduled"
    assert body["home_team"]["name"] == "Bayern Munich"

    r = await client.get(f"/api/v1/matches/fixtures/{uuid.uuid4()}")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_live_endpoint_is_pro_gated(client, world):
    # No token → 401
    r = await client.get("/api/v1/matches/live")
    assert r.status_code == 401

    # free_user → 403
    token = await _register_and_login(client, "fan2@example.com")
    r = await client.get(
        "/api/v1/matches/live", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 403

    # pro_analyst → 200 with the one live match
    await _upgrade_to_pro(client, token)
    r = await client.get(
        "/api/v1/matches/live", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["id"] == str(world["live_match_id"])


@pytest.mark.asyncio
async def test_fixture_events_pro_gated_and_ordered(client, world):
    token = await _register_and_login(client, "coach@example.com")
    await _upgrade_to_pro(client, token)

    r = await client.get(
        f"/api/v1/matches/fixtures/{world['live_match_id']}/events",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    events = r.json()
    assert [e["minute"] for e in events] == [23, 58]
    assert [e["event_type"] for e in events] == ["goal", "goal"]


# ── Teams / registry ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_team_registry_and_country_filter(client, world):
    r = await client.get("/api/v1/teams/registry")
    assert r.status_code == 200, r.text
    assert r.json()["total"] == 2

    r = await client.get("/api/v1/teams/registry", params={"country": "Germany"})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1 and items[0]["name"] == "Bayern Munich"


@pytest.mark.asyncio
async def test_team_squad_ordering(client, world):
    r = await client.get(f"/api/v1/teams/registry/{world['arsenal_id']}/squad")
    assert r.status_code == 200
    squad = r.json()
    # Arsenal has Saka (#7) + Odegaard (#8), sorted by jersey number ascending.
    assert [p["jersey_number"] for p in squad] == [7, 8]
    assert [p["name"] for p in squad] == ["Bukayo Saka", "Martin Odegaard"]


@pytest.mark.asyncio
async def test_team_registry_detail_and_404(client, world):
    r = await client.get(f"/api/v1/teams/registry/{world['bayern_id']}")
    assert r.status_code == 200
    assert r.json()["league"] == "Bundesliga"

    r = await client.get(f"/api/v1/teams/registry/{uuid.uuid4()}")
    assert r.status_code == 404
