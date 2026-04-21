"""
End-to-end coverage for the AI prediction + scouting endpoints.

What we mock and why
--------------------
* **Predictor model** — we stub ``MatchPredictor.instance()`` with a
  predictor whose ``_model`` is a tiny sklearn ``LogisticRegression``
  fit on synthetic data. This avoids dragging real XGBoost training
  into the test path while still exercising every layer (CRUD, deps,
  schemas, RBAC).
* **Anthropic SDK** — ``ScoutingService`` is constructed with a stub
  client whose ``messages.create`` returns a fixed payload. The 503
  path is exercised by overriding the dep with the default service
  (no key set in test env).
"""

from __future__ import annotations  # noqa: I001

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from sklearn.linear_model import LogisticRegression

from football_ai.api.main import app
from football_ai.api.v1.endpoints.players import get_scouting_service
from football_ai.ml.features.match_features import FEATURE_NAMES
from football_ai.ml.services.scouting_service import ScoutingService
from football_ai.ml.serving.match_predictor import MatchPredictor
from football_ai.models.match import Match, MatchStatus
from football_ai.models.player import Player
from football_ai.models.prediction import AIPrediction, PredictedOutcome
from football_ai.models.team import Team

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def world(test_session_factory):
    """Two teams + two players + one upcoming + one finished match."""
    arsenal = Team(name="Arsenal", country="England", league="Premier League")
    chelsea = Team(name="Chelsea", country="England", league="Premier League")
    async with test_session_factory() as s:
        s.add_all([arsenal, chelsea])
        await s.commit()
        await s.refresh(arsenal)
        await s.refresh(chelsea)

        saka = Player(name="Bukayo Saka", position="RW", current_team_id=arsenal.id)
        s.add(saka)
        await s.commit()
        await s.refresh(saka)

        now = datetime.now(UTC)
        upcoming = Match(
            home_team_id=arsenal.id,
            away_team_id=chelsea.id,
            league="Premier League",
            season="2024-25",
            match_date=now + timedelta(days=2),
            status=MatchStatus.scheduled,
        )
        finished = Match(
            home_team_id=arsenal.id,
            away_team_id=chelsea.id,
            league="Premier League",
            season="2024-25",
            match_date=now - timedelta(days=10),
            status=MatchStatus.finished,
            home_score=2,
            away_score=1,
        )
        s.add_all([upcoming, finished])
        await s.commit()
        await s.refresh(upcoming)
        await s.refresh(finished)

    return {
        "arsenal_id": arsenal.id,
        "chelsea_id": chelsea.id,
        "saka_id": saka.id,
        "upcoming_id": upcoming.id,
        "finished_id": finished.id,
    }


@pytest.fixture
def stub_predictor(monkeypatch):
    """Replace MatchPredictor.instance() with a predictor wrapping a tiny logreg."""
    MatchPredictor.reset_instance()

    rng = np.random.default_rng(7)
    n = len(FEATURE_NAMES)
    X = rng.normal(size=(60, n))
    # Bias the synthetic labels so the model produces non-trivial probs.
    y = ((X[:, 0] - X[:, 5] + 0.3 * rng.normal(size=60)) > 0).astype(int)
    y = np.where(y == 1, 0, 2)  # map to home_win / away_win classes
    # Inject a few draws so we have all 3 classes.
    y[:5] = 1

    model = LogisticRegression(max_iter=500).fit(X, y)

    predictor = MatchPredictor()
    predictor._model = model
    predictor._feature_cols = list(FEATURE_NAMES)
    predictor._model_version = "test-v1"
    monkeypatch.setattr(MatchPredictor, "_instance", predictor, raising=False)
    yield predictor
    MatchPredictor.reset_instance()


class _StubAnthropicClient:
    """Minimal stand-in for anthropic.Anthropic — only the bits we use."""

    def __init__(self) -> None:
        self.messages = self  # so client.messages.create works

    def create(self, **_: Any) -> Any:
        return SimpleNamespace(
            content=[SimpleNamespace(text="Saka is a high-output right winger.")],
            usage=SimpleNamespace(input_tokens=120, output_tokens=80),
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _register(client, email: str) -> str:
    await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "password-123"},
    )
    r = await client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": "password-123"},
    )
    return r.json()["access_token"]


async def _upgrade(client, token: str, role: str) -> None:
    r = await client.patch(
        "/api/v1/auth/me/role",
        headers={"Authorization": f"Bearer {token}"},
        json={"role": role},
    )
    assert r.status_code == 200, r.text


# ── Predictions ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_match_requires_pro_role(client, world, stub_predictor):
    token = await _register(client, "fan@example.com")
    r = await client.post(
        f"/api/v1/predictions/match/{world['upcoming_id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_predict_then_get_round_trip(client, world, stub_predictor):
    token = await _register(client, "scout@example.com")
    await _upgrade(client, token, "pro_analyst")
    headers = {"Authorization": f"Bearer {token}"}

    # POST runs inference and persists.
    r = await client.post(
        f"/api/v1/predictions/match/{world['upcoming_id']}", headers=headers
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["match_id"] == str(world["upcoming_id"])
    assert body["model_version"] == "test-v1"
    probs = (body["home_win_prob"], body["draw_prob"], body["away_win_prob"])
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    assert body["predicted_outcome"] in {"home_win", "draw", "away_win"}
    assert 0.0 <= body["confidence_score"] <= 1.0
    assert body["was_correct"] is None  # match not finished

    # GET reads what POST wrote (same model_version).
    r = await client.get(
        f"/api/v1/predictions/match/{world['upcoming_id']}", headers=headers
    )
    assert r.status_code == 200
    assert r.json()["model_version"] == "test-v1"


@pytest.mark.asyncio
async def test_predict_unknown_match_returns_404(client, world, stub_predictor):
    import uuid

    token = await _register(client, "analyst@example.com")
    await _upgrade(client, token, "pro_analyst")
    r = await client.post(
        f"/api/v1/predictions/match/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_history_and_accuracy_with_seeded_predictions(
    client, world, test_session_factory, stub_predictor
):
    # Seed two predictions directly, one already scored as correct.
    async with test_session_factory() as s:
        s.add_all([
            AIPrediction(
                match_id=world["finished_id"],
                model_version="test-v1",
                home_win_prob=0.6, draw_prob=0.2, away_win_prob=0.2,
                confidence_score=0.55,
                actual_result=PredictedOutcome.home_win,
                was_correct=True,
            ),
            AIPrediction(
                match_id=world["upcoming_id"],
                model_version="test-v1",
                home_win_prob=0.3, draw_prob=0.3, away_win_prob=0.4,
                confidence_score=0.30,
            ),
        ])
        await s.commit()

    token = await _register(client, "history@example.com")
    await _upgrade(client, token, "pro_analyst")
    headers = {"Authorization": f"Bearer {token}"}

    # History returns both rows.
    r = await client.get("/api/v1/predictions/history", headers=headers)
    assert r.status_code == 200
    assert r.json()["total"] == 2

    # Accuracy: only the scored one counts → 1/1 = 1.0.
    r = await client.get("/api/v1/predictions/accuracy", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["total_scored"] == 1
    assert body["total_correct"] == 1
    assert body["overall_accuracy"] == pytest.approx(1.0)
    assert any(b["league"] == "Premier League" for b in body["by_league"])


# ── Scouting ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scouting_requires_club_scout_role(client, world):
    token = await _register(client, "fan2@example.com")
    await _upgrade(client, token, "pro_analyst")  # not enough
    r = await client.get(
        f"/api/v1/players/catalog/{world['saka_id']}/scouting-report",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_scouting_returns_503_when_unconfigured(client, world):
    """Default test env has no ANTHROPIC_API_KEY → graceful 503."""
    token = await _register(client, "scout1@example.com")
    await _upgrade(client, token, "club_scout")
    r = await client.get(
        f"/api/v1/players/catalog/{world['saka_id']}/scouting-report",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_scouting_returns_prose_with_stub_client(client, world):
    """Inject a stub Anthropic client and verify the prose round-trip."""
    stub_service = ScoutingService(client=_StubAnthropicClient())
    app.dependency_overrides[get_scouting_service] = lambda: stub_service
    try:
        token = await _register(client, "scout2@example.com")
        await _upgrade(client, token, "club_scout")
        r = await client.get(
            f"/api/v1/players/catalog/{world['saka_id']}/scouting-report",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "right winger" in body["prose"]
        assert body["token_count"] == 200
        assert body["model"]  # whatever default model is set
    finally:
        app.dependency_overrides.pop(get_scouting_service, None)
