"""
Integration coverage for the live WebSocket endpoint and admin task surface.

The WS tests use Starlette's sync :class:`TestClient` (httpx's async client
doesn't speak the WS handshake natively). We bind it to the same
in-memory SQLite engine the rest of the suite uses, then exercise:

* connect → ``hello`` snapshot
* admin POST /push → subscriber receives the event
* unauth or wrong-role → 1008 close
* unknown match → 1011 close + 404-equivalent

Admin task tests run the inline path because that exercises the full
job code (DB read, predictor call, prediction backfill) without
requiring a live arq worker.
"""

from __future__ import annotations  # noqa: I001

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.api.dependencies import get_db
from football_ai.api.main import app
from football_ai.models.match import Match, MatchStatus
from football_ai.models.prediction import AIPrediction, PredictedOutcome
from football_ai.models.team import Team

# ── Shared world fixture ─────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def world(test_session_factory):
    async with test_session_factory() as s:
        home = Team(name="Liverpool", country="England", league="Premier League")
        away = Team(name="Man City", country="England", league="Premier League")
        s.add_all([home, away])
        await s.commit()
        await s.refresh(home)
        await s.refresh(away)

        now = datetime.now(UTC)
        live_match = Match(
            home_team_id=home.id,
            away_team_id=away.id,
            league="Premier League",
            season="2024-25",
            match_date=now,
            status=MatchStatus.live,
            home_score=1,
            away_score=0,
        )
        finished = Match(
            home_team_id=home.id,
            away_team_id=away.id,
            league="Premier League",
            season="2024-25",
            match_date=now - timedelta(hours=2),
            status=MatchStatus.finished,
            home_score=2,
            away_score=2,
        )
        s.add_all([live_match, finished])
        await s.commit()
        await s.refresh(live_match)
        await s.refresh(finished)
    return {
        "home_id": home.id,
        "away_id": away.id,
        "live_id": live_match.id,
        "finished_id": finished.id,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _register_and_promote(client, email: str, role: str) -> str:
    """Register, log in, upgrade role; return the access token."""
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
        json={"role": role},
    )
    assert r.status_code == 200, r.text
    return token


def _sync_client(test_session_factory) -> TestClient:
    """Build a sync TestClient bound to the same in-memory engine."""

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with test_session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app)


# ── WebSocket tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ws_rejects_invalid_token(client, world, test_session_factory):
    """Bad token → handshake closed with 1008."""
    sync = _sync_client(test_session_factory)
    try:
        with pytest.raises(Exception):
            with sync.websocket_connect(
                f"/api/v1/live/matches/{world['live_id']}?token=garbage"
            ):
                pass
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_ws_pro_user_receives_hello_then_pushed_event(
    client, world, test_session_factory
):
    token = await _register_and_promote(client, "ws-pro@example.com", "pro_analyst")
    admin_token = await _register_and_promote(client, "ws-admin@example.com", "admin")
    sync = _sync_client(test_session_factory)

    try:
        url = f"/api/v1/live/matches/{world['live_id']}?token={token}"
        with sync.websocket_connect(url) as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            assert hello["match_id"] == str(world["live_id"])
            assert hello["status"] == "live"

            # Push an event via the admin REST endpoint and assert the
            # subscriber receives it on the same socket.
            r = sync.post(
                f"/api/v1/live/matches/{world['live_id']}/push",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"type": "event", "minute": 33, "kind": "goal"},
            )
            assert r.status_code == 200, r.text
            assert r.json()["delivered"] == 1

            event = ws.receive_json()
            assert event["type"] == "event"
            assert event["minute"] == 33
            assert event["match_id"] == str(world["live_id"])
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_ws_free_user_is_rejected(client, world, test_session_factory):
    """Default role is free_user — should be closed with 1008."""
    # Register but DON'T upgrade.
    await client.post(
        "/api/v1/auth/register",
        json={"email": "ws-free@example.com", "password": "password-123"},
    )
    r = await client.post(
        "/api/v1/auth/login",
        data={"username": "ws-free@example.com", "password": "password-123"},
    )
    token = r.json()["access_token"]

    sync = _sync_client(test_session_factory)
    try:
        with pytest.raises(Exception):
            with sync.websocket_connect(
                f"/api/v1/live/matches/{world['live_id']}?token={token}"
            ):
                pass
    finally:
        app.dependency_overrides.pop(get_db, None)


# ── Admin task tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_admin_lists_available_tasks(client):
    token = await _register_and_promote(client, "ops@example.com", "admin")
    r = await client.get(
        "/api/v1/admin/tasks",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "backfill_finished_predictions" in body["available"]
    assert "refresh_upcoming_predictions" in body["available"]
    assert "warm_caches" in body["available"]


@pytest.mark.asyncio
async def test_admin_task_requires_admin_role(client):
    token = await _register_and_promote(client, "scout-admin@example.com", "club_scout")
    r = await client.post(
        "/api/v1/admin/tasks/warm_caches",
        headers={"Authorization": f"Bearer {token}"},
        json={},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_admin_runs_warm_caches_inline(client):
    token = await _register_and_promote(client, "ops2@example.com", "admin")
    r = await client.post(
        "/api/v1/admin/tasks/warm_caches",
        headers={"Authorization": f"Bearer {token}"},
        json={},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "warm_caches"
    assert body["mode"] == "inline"
    assert body["result"]["cleared"] == "all"


@pytest.mark.asyncio
async def test_admin_unknown_task_returns_404(client):
    token = await _register_and_promote(client, "ops3@example.com", "admin")
    r = await client.post(
        "/api/v1/admin/tasks/not_a_real_task",
        headers={"Authorization": f"Bearer {token}"},
        json={},
    )
    assert r.status_code == 404


# ── Background task: backfill_finished_predictions ──────────────────────────

@pytest.mark.asyncio
async def test_backfill_finished_predictions_stamps_scores(
    test_session_factory, world
):
    """Direct task invocation — seeded prediction gets ``actual_result`` stamped."""
    from football_ai.tasks.jobs import backfill_finished_predictions

    async with test_session_factory() as s:
        s.add(
            AIPrediction(
                match_id=world["finished_id"],
                model_version="t1",
                home_win_prob=0.4, draw_prob=0.3, away_win_prob=0.3,
                confidence_score=0.4,
            )
        )
        await s.commit()

    result = await backfill_finished_predictions(
        {"session_factory": test_session_factory},
        since_hours=24,
    )
    assert result["matches"] >= 1
    assert result["predictions_updated"] >= 1

    async with test_session_factory() as s:
        from sqlalchemy import select
        row = (
            await s.execute(
                select(AIPrediction).where(
                    AIPrediction.match_id == world["finished_id"]
                )
            )
        ).scalar_one()
        # 2-2 → draw
        assert row.actual_result == PredictedOutcome.draw
        assert row.was_correct is False  # we predicted home_win (0.4 max)
