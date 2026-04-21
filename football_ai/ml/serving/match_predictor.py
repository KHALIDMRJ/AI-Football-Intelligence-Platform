"""
Match-outcome inference service.

Loads the persisted ``match_outcome`` model on first call and caches it
in-process. Each invocation:

1. Builds the 15-feature vector via ``ml.features.match_features.build_features``
2. Calls ``model.predict_proba`` → (P(home_win), P(draw), P(away_win))
3. Computes a confidence score (max prob × feature completeness)
4. Persists / upserts an ``AIPrediction`` row keyed by (match_id, model_version)
5. Returns the AIPrediction for the caller to serialize

Why upsert (not insert-only)
----------------------------
Re-prediction during a live match (Phase 6) needs to overwrite the prior
inference for the *same* model version. New model versions create a new
row (``UniqueConstraint(match_id, model_version)``) so historical accuracy
analytics survive model bumps.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.core.exceptions import PredictionError
from football_ai.logger import get_logger
from football_ai.ml.features.match_features import (
    FEATURE_NAMES,
    MatchFeatures,
    build_features,
)
from football_ai.ml.serving.model_registry import ModelRegistry
from football_ai.models.match import Match, MatchStatus
from football_ai.models.prediction import AIPrediction, PredictedOutcome

logger = get_logger(__name__)

_REGISTRY_NAME = "match_outcome"
# Class indices map to the trainer's label encoding — keep them in lockstep.
CLASS_ORDER: tuple[PredictedOutcome, ...] = (
    PredictedOutcome.home_win,
    PredictedOutcome.draw,
    PredictedOutcome.away_win,
)


class MatchPredictor:
    """Process-singleton wrapper around the persisted XGBoost model."""

    _instance_lock = threading.Lock()
    _instance: MatchPredictor | None = None

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry or ModelRegistry()
        self._model: Any = None
        self._model_version: str = "untrained"
        self._feature_cols: list[str] = []
        self._load_lock = threading.Lock()

    # ── Singleton accessor ────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> MatchPredictor:
        """Return a per-process singleton; cheap to call from request handlers."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Test hook — drops the cached singleton and any loaded model state."""
        with cls._instance_lock:
            cls._instance = None

    # ── Model loading ─────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if not self._registry.exists(_REGISTRY_NAME):
                raise PredictionError(
                    "Match-outcome model not trained yet. Run "
                    "`scripts/train_match_predictor.py` first.",
                )
            self._model = self._registry.load(_REGISTRY_NAME)
            meta = self._registry.get_metadata().get(_REGISTRY_NAME, {})
            self._feature_cols = meta.get("feature_cols", FEATURE_NAMES)
            self._model_version = meta.get("trained_at", "unknown")

            if self._feature_cols != FEATURE_NAMES:
                raise PredictionError(
                    "Feature schema drift: model trained on "
                    f"{self._feature_cols} but features module exports "
                    f"{FEATURE_NAMES}. Retrain required."
                )

    @property
    def model_version(self) -> str:
        self._ensure_loaded()
        return self._model_version

    # ── Inference ─────────────────────────────────────────────────────────────

    async def predict(
        self, db: AsyncSession, match: Match
    ) -> AIPrediction:
        """Run a prediction and upsert the AIPrediction row.

        Caller must commit the session afterwards (this method only flushes
        so the returned ORM object has its server-generated id available).
        """
        self._ensure_loaded()

        if match.status == MatchStatus.cancelled:
            raise PredictionError(
                "Cannot predict a cancelled match.",
            )

        features: MatchFeatures = await build_features(db, match)
        probs = self._predict_probs(features.values)

        home_p, draw_p, away_p = probs
        max_idx = max(range(3), key=lambda i: probs[i])
        confidence = round(probs[max_idx] * features.completeness, 4)

        existing = await self._fetch_existing(db, match.id, self._model_version)
        if existing is None:
            row = AIPrediction(
                match_id=match.id,
                model_version=self._model_version,
                home_win_prob=float(home_p),
                draw_prob=float(draw_p),
                away_win_prob=float(away_p),
                confidence_score=confidence,
                key_factors=features.factors,
            )
            db.add(row)
        else:
            existing.home_win_prob = float(home_p)
            existing.draw_prob = float(draw_p)
            existing.away_win_prob = float(away_p)
            existing.confidence_score = confidence
            existing.key_factors = features.factors
            row = existing

        await db.flush()
        return row

    def _predict_probs(self, values: list[float]) -> tuple[float, float, float]:
        """Run the underlying model — works for sklearn-compatible classifiers."""
        try:
            import numpy as np  # local import keeps test-startup fast
            X = np.asarray(values, dtype=float).reshape(1, -1)
            proba = self._model.predict_proba(X)[0]
        except Exception as exc:  # narrow ML-stack failures to a clean 422
            raise PredictionError(
                f"Model inference failed: {exc.__class__.__name__}: {exc}"
            ) from exc

        if len(proba) != 3:
            raise PredictionError(
                f"Model returned {len(proba)} class probabilities, expected 3."
            )
        # XGBoost / sklearn return numpy floats — coerce to plain Python.
        return (float(proba[0]), float(proba[1]), float(proba[2]))

    @staticmethod
    async def _fetch_existing(
        db: AsyncSession, match_id: uuid.UUID, model_version: str
    ) -> AIPrediction | None:
        stmt = select(AIPrediction).where(
            AIPrediction.match_id == match_id,
            AIPrediction.model_version == model_version,
        )
        return (await db.execute(stmt)).scalar_one_or_none()


def outcome_from_score(home: int, away: int) -> PredictedOutcome:
    """Map a final scoreline to the canonical class label."""
    if home > away:
        return PredictedOutcome.home_win
    if home < away:
        return PredictedOutcome.away_win
    return PredictedOutcome.draw


def predicted_class(p: AIPrediction) -> PredictedOutcome:
    """Return the argmax outcome for a stored prediction row."""
    pairs = (
        (p.home_win_prob, PredictedOutcome.home_win),
        (p.draw_prob, PredictedOutcome.draw),
        (p.away_win_prob, PredictedOutcome.away_win),
    )
    return max(pairs, key=lambda t: t[0])[1]
