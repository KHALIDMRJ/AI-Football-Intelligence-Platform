"""
Unit tests for the match-features builder.

We seed a tiny in-memory DB, then assert:

* feature vector length + naming order match the predictor's contract
* cold-start (no history) falls back to documented priors
* H2H is scored from the home team's perspective
* form rates are computed correctly when matches DO exist
"""

from __future__ import annotations  # noqa: I001

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import football_ai.models  # noqa: F401 — register tables
from football_ai.db.base import Base
from football_ai.ml.features.match_features import (
    DEFAULT_FORM_PRIORS,
    DEFAULT_H2H_PRIORS,
    FEATURE_NAMES,
    build_features,
)
from football_ai.models.match import Match, MatchStatus
from football_ai.models.team import Team


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        future=True,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _team(name: str) -> Team:
    return Team(name=name, league="Premier League")


def _match(home: Team, away: Team, *, when: datetime, status, hs=None, as_=None) -> Match:
    return Match(
        home_team_id=home.id,
        away_team_id=away.id,
        league="Premier League",
        season="2024-25",
        match_date=when,
        status=status,
        home_score=hs,
        away_score=as_,
    )


@pytest.mark.asyncio
async def test_feature_names_count_is_15():
    assert len(FEATURE_NAMES) == 15


@pytest.mark.asyncio
async def test_cold_start_uses_priors(session):
    home, away = _team("Arsenal"), _team("Chelsea")
    session.add_all([home, away])
    await session.commit()
    await session.refresh(home)
    await session.refresh(away)

    fixture = _match(home, away, when=datetime(2025, 5, 1, tzinfo=UTC),
                     status=MatchStatus.scheduled)
    session.add(fixture)
    await session.commit()
    await session.refresh(fixture)

    f = await build_features(session, fixture)
    d = f.as_dict()

    # Every form rate / per-match figure should equal the prior.
    for key, prior in DEFAULT_FORM_PRIORS.items():
        assert d[f"home_{key}"] == pytest.approx(prior)
        assert d[f"away_{key}"] == pytest.approx(prior)
    # H2H falls back to priors as well.
    assert d["h2h_home_w_rate"] == pytest.approx(DEFAULT_H2H_PRIORS["home_w_rate"])
    assert d["h2h_d_rate"] == pytest.approx(DEFAULT_H2H_PRIORS["d_rate"])
    assert d["h2h_home_l_rate"] == pytest.approx(DEFAULT_H2H_PRIORS["home_l_rate"])
    # Days-since priors → 7
    assert d["home_days_since_last"] == pytest.approx(7.0)
    assert d["away_days_since_last"] == pytest.approx(7.0)
    assert f.completeness == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_form_rates_reflect_recent_results(session):
    home, away = _team("Arsenal"), _team("Chelsea")
    other = _team("Tottenham")
    session.add_all([home, away, other])
    await session.commit()
    for t in (home, away, other):
        await session.refresh(t)

    base = datetime(2025, 1, 1, tzinfo=UTC)

    # Home played 4 finished matches before the fixture: W, W, D, L.
    finished = [
        _match(home, other, when=base + timedelta(days=1),
               status=MatchStatus.finished, hs=2, as_=0),  # W
        _match(other, home, when=base + timedelta(days=8),
               status=MatchStatus.finished, hs=0, as_=1),  # W
        _match(home, other, when=base + timedelta(days=15),
               status=MatchStatus.finished, hs=1, as_=1),  # D
        _match(other, home, when=base + timedelta(days=22),
               status=MatchStatus.finished, hs=2, as_=1),  # L
    ]
    fixture = _match(home, away, when=base + timedelta(days=29),
                     status=MatchStatus.scheduled)
    session.add_all(finished + [fixture])
    await session.commit()
    await session.refresh(fixture)

    f = await build_features(session, fixture)
    d = f.as_dict()

    # 2 W / 1 D / 1 L over 4 → 0.5 / 0.25 / 0.25
    assert d["home_w_rate"] == pytest.approx(0.5)
    assert d["home_d_rate"] == pytest.approx(0.25)
    assert d["home_l_rate"] == pytest.approx(0.25)
    # GF: 2+1+1+1 = 5 / 4 ; GA: 0+0+1+2 = 3 / 4
    assert d["home_gf_pm"] == pytest.approx(1.25)
    assert d["home_ga_pm"] == pytest.approx(0.75)
    # Days-since-last: fixture is day 29, last home match is day 22 → 7.
    assert d["home_days_since_last"] == pytest.approx(7.0)
    assert f.completeness > 0.0


@pytest.mark.asyncio
async def test_h2h_is_from_home_perspective(session):
    home, away = _team("Arsenal"), _team("Chelsea")
    session.add_all([home, away])
    await session.commit()
    for t in (home, away):
        await session.refresh(t)

    base = datetime(2025, 1, 1, tzinfo=UTC)

    # 2 prior H2H, both won by 'home' (regardless of which side they were on).
    h2h = [
        _match(home, away, when=base + timedelta(days=1),
               status=MatchStatus.finished, hs=2, as_=0),
        _match(away, home, when=base + timedelta(days=10),
               status=MatchStatus.finished, hs=0, as_=3),
    ]
    fixture = _match(home, away, when=base + timedelta(days=20),
                     status=MatchStatus.scheduled)
    session.add_all(h2h + [fixture])
    await session.commit()
    await session.refresh(fixture)

    f = await build_features(session, fixture)
    d = f.as_dict()
    assert d["h2h_home_w_rate"] == pytest.approx(1.0)
    assert d["h2h_d_rate"] == pytest.approx(0.0)
    assert d["h2h_home_l_rate"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_factors_mention_form_when_history_present(session):
    home, away = _team("Arsenal"), _team("Chelsea")
    session.add_all([home, away])
    await session.commit()
    for t in (home, away):
        await session.refresh(t)
    base = datetime(2025, 2, 1, tzinfo=UTC)
    session.add_all([
        _match(home, away, when=base, status=MatchStatus.finished, hs=3, as_=1),
    ])
    fixture = _match(home, away, when=base + timedelta(days=10),
                     status=MatchStatus.scheduled)
    session.add(fixture)
    await session.commit()
    await session.refresh(fixture)

    f = await build_features(session, fixture)
    text = " | ".join(f.factors)
    assert "Home side" in text or "Away side" in text
