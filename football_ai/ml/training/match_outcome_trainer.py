"""
Offline trainer for the match-outcome classifier (Phase 5).

Why offline-only
----------------
Training is not a request — it's a multi-second batch over historical
fixtures. The HTTP layer never invokes this; the CLI script
``scripts/train_match_predictor.py`` does. Inference (``MatchPredictor``)
loads the persisted artefact and is the only thing the API touches.

Algorithm
---------
XGBoost ``multi:softprob`` over the 15 features defined in
``ml.features.match_features.FEATURE_NAMES``. Class labels follow
``MatchPredictor.CLASS_ORDER`` (home_win=0, draw=1, away_win=2) so the
inference path can map argmax → ``PredictedOutcome`` without a separate
encoder.

Cold-start: if the historical match count is below ``min_matches``, we
skip training and raise ``RuntimeError`` rather than ship a model fit on
a handful of rows that would mislead downstream callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.logger import get_logger
from football_ai.ml.features.match_features import FEATURE_NAMES, build_features
from football_ai.ml.serving.match_predictor import outcome_from_score
from football_ai.ml.serving.model_registry import ModelRegistry
from football_ai.models.match import Match, MatchStatus
from football_ai.models.prediction import PredictedOutcome

logger = get_logger(__name__)

_LABEL_INDEX: dict[PredictedOutcome, int] = {
    PredictedOutcome.home_win: 0,
    PredictedOutcome.draw: 1,
    PredictedOutcome.away_win: 2,
}


@dataclass
class TrainingResult:
    n_samples: int
    train_logloss: float
    train_accuracy: float
    feature_cols: list[str]


async def _load_finished_matches(db: AsyncSession) -> list[Match]:
    stmt = (
        select(Match)
        .where(
            Match.is_deleted.is_(False),
            Match.status == MatchStatus.finished,
            Match.home_score.is_not(None),
            Match.away_score.is_not(None),
        )
        .order_by(Match.match_date.asc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def train_match_outcome_model(
    db: AsyncSession,
    *,
    min_matches: int = 50,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.08,
    registry: ModelRegistry | None = None,
) -> TrainingResult:
    """Fit the model on every finished match in the DB and persist it."""
    matches = await _load_finished_matches(db)
    if len(matches) < min_matches:
        raise RuntimeError(
            f"Only {len(matches)} finished matches available; need at least "
            f"{min_matches} to train a useful match-outcome model."
        )

    X_rows: list[list[float]] = []
    y: list[int] = []

    for m in matches:
        feats = await build_features(db, m)
        X_rows.append(feats.values)
        label = outcome_from_score(m.home_score or 0, m.away_score or 0)
        y.append(_LABEL_INDEX[label])

    X = np.asarray(X_rows, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    logger.info("Training on %d historical matches", len(y))

    # XGBoost is a project-wide dep; fall back to sklearn if it's missing
    # so the test suite still works in stripped-down environments.
    model = _build_xgb_classifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    model.fit(X, y_arr)

    proba = model.predict_proba(X)
    train_logloss = _logloss(y_arr, proba)
    train_accuracy = float((proba.argmax(axis=1) == y_arr).mean())
    logger.info(
        "Train logloss=%.4f accuracy=%.4f", train_logloss, train_accuracy
    )

    reg = registry or ModelRegistry()
    reg.save(
        "match_outcome",
        model,
        metrics={
            "n_samples": float(len(y)),
            "train_logloss": train_logloss,
            "train_accuracy": train_accuracy,
        },
        feature_cols=FEATURE_NAMES,
    )
    return TrainingResult(
        n_samples=len(y),
        train_logloss=train_logloss,
        train_accuracy=train_accuracy,
        feature_cols=list(FEATURE_NAMES),
    )


def _build_xgb_classifier(
    *, n_estimators: int, max_depth: int, learning_rate: float
) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError:  # pragma: no cover — tests pin xgboost
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )

    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=1,  # keep single-threaded so tests / CI stay deterministic
    )


def _logloss(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-15) -> float:
    n = len(y_true)
    p = np.clip(proba, eps, 1 - eps)
    return float(-np.mean(np.log(p[np.arange(n), y_true])))
