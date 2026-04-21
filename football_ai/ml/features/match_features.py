"""
Per-fixture feature vector builder for the match-outcome predictor.

Output shape (15 features, fixed order — anchored in ``FEATURE_NAMES``)
----------------------------------------------------------------------
Home form (5)   : home_w_rate, home_d_rate, home_l_rate, home_gf_pm, home_ga_pm
Away form (5)   : away_w_rate, away_d_rate, away_l_rate, away_gf_pm, away_ga_pm
Head-to-head (3): h2h_home_w_rate, h2h_d_rate, h2h_home_l_rate
Context (2)     : home_days_since_last, away_days_since_last

Football-domain notes
---------------------
* ``last_n=5`` for form is the conventional "recent form" window scouts
  use; longer windows wash out momentum, shorter ones are noisy.
* Home/away form is split because home-field effect is strong (~0.3 goal
  expectation gap in top-flight European football). Mixing the two would
  blur the signal.
* H2H is from the home team's perspective so a single rate triple
  captures the directional rivalry without doubling features.
* ``days_since_last`` is a fixture-congestion proxy — the literature
  consistently links short turnarounds to lower xG.

Cold-start handling
-------------------
A team with fewer than ``last_n`` matches uses the global median for
form rates / per-match goal counts, taken from the training data
medians stored on the model (defaults: 0.4 W, 0.25 D, 0.35 L, 1.4 GF,
1.3 GA). H2H with no history defaults to the prior (0.45, 0.25, 0.30).
``days_since_last`` with no prior match defaults to 7 (typical league rest).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.models.match import Match, MatchStatus

# Order matters — the trainer pickles this list and the predictor asserts
# the same order at inference time.
FEATURE_NAMES: list[str] = [
    "home_w_rate", "home_d_rate", "home_l_rate", "home_gf_pm", "home_ga_pm",
    "away_w_rate", "away_d_rate", "away_l_rate", "away_gf_pm", "away_ga_pm",
    "h2h_home_w_rate", "h2h_d_rate", "h2h_home_l_rate",
    "home_days_since_last", "away_days_since_last",
]

# Conservative population priors (rough Premier-League shape) used as
# cold-start defaults when a side has no history.
_PRIOR_W = 0.40
_PRIOR_D = 0.25
_PRIOR_L = 0.35
_PRIOR_GF = 1.40
_PRIOR_GA = 1.30
_PRIOR_DAYS = 7

DEFAULT_FORM_PRIORS: dict[str, float] = {
    "w_rate": _PRIOR_W,
    "d_rate": _PRIOR_D,
    "l_rate": _PRIOR_L,
    "gf_pm": _PRIOR_GF,
    "ga_pm": _PRIOR_GA,
}

DEFAULT_H2H_PRIORS: dict[str, float] = {
    "home_w_rate": 0.45,  # home advantage skews this slightly
    "d_rate": 0.25,
    "home_l_rate": 0.30,
}


@dataclass(frozen=True)
class MatchFeatures:
    """Feature vector + provenance for one fixture."""

    match_id: uuid.UUID
    values: list[float]
    completeness: float  # in [0, 1] — share of non-default features used
    factors: list[str]   # human-readable highlights, fed into AIPrediction

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.values))


# ── Internal helpers ─────────────────────────────────────────────────────────

async def _team_recent_finished(
    db: AsyncSession,
    team_id: uuid.UUID,
    *,
    before: datetime,
    last_n: int,
) -> list[Match]:
    """Most recent N finished matches involving ``team_id`` strictly before ``before``."""
    stmt = (
        select(Match)
        .where(
            Match.is_deleted.is_(False),
            Match.status == MatchStatus.finished,
            Match.match_date < before,
            Match.home_score.is_not(None),
            Match.away_score.is_not(None),
            or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
        )
        .order_by(Match.match_date.desc())
        .limit(last_n)
    )
    return list((await db.execute(stmt)).scalars().all())


def _form_from_matches(team_id: uuid.UUID, matches: list[Match]) -> dict[str, float]:
    if not matches:
        return dict(DEFAULT_FORM_PRIORS)

    wins = draws = losses = 0
    gf = ga = 0
    for m in matches:
        is_home = m.home_team_id == team_id
        team_score = m.home_score if is_home else m.away_score
        opp_score = m.away_score if is_home else m.home_score
        if team_score is None or opp_score is None:  # defensive
            continue
        gf += team_score
        ga += opp_score
        if team_score > opp_score:
            wins += 1
        elif team_score == opp_score:
            draws += 1
        else:
            losses += 1

    n = len(matches)
    return {
        "w_rate": wins / n,
        "d_rate": draws / n,
        "l_rate": losses / n,
        "gf_pm": gf / n,
        "ga_pm": ga / n,
    }


async def _last_match_date(
    db: AsyncSession, team_id: uuid.UUID, *, before: datetime
) -> datetime | None:
    stmt = (
        select(Match.match_date)
        .where(
            Match.is_deleted.is_(False),
            Match.match_date < before,
            or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
        )
        .order_by(Match.match_date.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def _h2h_from_home_perspective(
    db: AsyncSession,
    *,
    home_team_id: uuid.UUID,
    away_team_id: uuid.UUID,
    before: datetime,
    last_n: int,
) -> dict[str, float]:
    """Last-N finished H2H between these two clubs, scored from home_team_id POV."""
    stmt = (
        select(Match)
        .where(
            Match.is_deleted.is_(False),
            Match.status == MatchStatus.finished,
            Match.match_date < before,
            Match.home_score.is_not(None),
            Match.away_score.is_not(None),
            or_(
                and_(
                    Match.home_team_id == home_team_id,
                    Match.away_team_id == away_team_id,
                ),
                and_(
                    Match.home_team_id == away_team_id,
                    Match.away_team_id == home_team_id,
                ),
            ),
        )
        .order_by(Match.match_date.desc())
        .limit(last_n)
    )
    matches = list((await db.execute(stmt)).scalars().all())
    if not matches:
        return dict(DEFAULT_H2H_PRIORS)

    home_wins = draws = home_losses = 0
    for m in matches:
        # Score the result from the perspective of the *upcoming* home team.
        if m.home_team_id == home_team_id:
            our, theirs = m.home_score, m.away_score
        else:
            our, theirs = m.away_score, m.home_score
        if our > theirs:
            home_wins += 1
        elif our == theirs:
            draws += 1
        else:
            home_losses += 1
    n = len(matches)
    return {
        "home_w_rate": home_wins / n,
        "d_rate": draws / n,
        "home_l_rate": home_losses / n,
    }


# ── Public API ───────────────────────────────────────────────────────────────

async def build_features(
    db: AsyncSession,
    match: Match,
    *,
    last_n: int = 5,
) -> MatchFeatures:
    """Build the 15-feature vector for an upcoming/historical fixture.

    Caller is responsible for passing a ``Match`` row with both team IDs
    set; this function does NOT mutate state. ``match.match_date`` is
    treated as the as-of cut-off so historical training and live inference
    use identical look-back rules.
    """
    cutoff = match.match_date

    home_recent = await _team_recent_finished(
        db, match.home_team_id, before=cutoff, last_n=last_n
    )
    away_recent = await _team_recent_finished(
        db, match.away_team_id, before=cutoff, last_n=last_n
    )
    h2h = await _h2h_from_home_perspective(
        db,
        home_team_id=match.home_team_id,
        away_team_id=match.away_team_id,
        before=cutoff,
        last_n=last_n,
    )

    home_form = _form_from_matches(match.home_team_id, home_recent)
    away_form = _form_from_matches(match.away_team_id, away_recent)

    home_last = await _last_match_date(db, match.home_team_id, before=cutoff)
    away_last = await _last_match_date(db, match.away_team_id, before=cutoff)

    def _days_since(dt: datetime | None) -> float:
        if dt is None:
            return float(_PRIOR_DAYS)
        # Both dt and cutoff carry tz info from the model column; if a test
        # constructs a naive datetime, normalize it to UTC.
        a = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        b = cutoff if cutoff.tzinfo else cutoff.replace(tzinfo=UTC)
        return max(0.0, (b - a).total_seconds() / 86_400.0)

    values = [
        home_form["w_rate"], home_form["d_rate"], home_form["l_rate"],
        home_form["gf_pm"], home_form["ga_pm"],
        away_form["w_rate"], away_form["d_rate"], away_form["l_rate"],
        away_form["gf_pm"], away_form["ga_pm"],
        h2h["home_w_rate"], h2h["d_rate"], h2h["home_l_rate"],
        _days_since(home_last), _days_since(away_last),
    ]

    # Completeness = share of "real" inputs used (vs prior fallbacks).
    real_signals = (
        bool(home_recent), bool(away_recent),
        h2h != DEFAULT_H2H_PRIORS,
        home_last is not None, away_last is not None,
    )
    completeness = sum(real_signals) / len(real_signals)

    factors = _human_factors(home_form, away_form, h2h, home_recent, away_recent)

    return MatchFeatures(
        match_id=match.id,
        values=values,
        completeness=completeness,
        factors=factors,
    )


def _human_factors(
    home_form: dict[str, float],
    away_form: dict[str, float],
    h2h: dict[str, float],
    home_recent: list[Match],
    away_recent: list[Match],
) -> list[str]:
    """Generate up to 4 short narrative bullets from the feature inputs.

    Persisted on AIPrediction.key_factors so the UI can render "why" without
    a second LLM call.
    """
    out: list[str] = []
    if home_recent:
        out.append(
            f"Home side won {int(home_form['w_rate'] * len(home_recent))} of "
            f"last {len(home_recent)} matches "
            f"(GF/GA {home_form['gf_pm']:.2f}/{home_form['ga_pm']:.2f} per match)."
        )
    if away_recent:
        out.append(
            f"Away side won {int(away_form['w_rate'] * len(away_recent))} of "
            f"last {len(away_recent)} matches "
            f"(GF/GA {away_form['gf_pm']:.2f}/{away_form['ga_pm']:.2f} per match)."
        )
    if h2h != DEFAULT_H2H_PRIORS:
        if h2h["home_w_rate"] > h2h["home_l_rate"]:
            out.append(
                f"Home team leads recent head-to-head "
                f"({h2h['home_w_rate']:.0%} wins)."
            )
        elif h2h["home_l_rate"] > h2h["home_w_rate"]:
            out.append(
                f"Away team owns recent head-to-head "
                f"({h2h['home_l_rate']:.0%} home losses)."
            )
        else:
            out.append("Recent head-to-head is evenly matched.")
    return out[:4]
