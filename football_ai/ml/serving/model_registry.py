"""
Model registry.

Provides a single location for saving and loading all trained models.
Uses joblib for serialisation (handles sklearn, XGBoost, and numpy arrays).

Directory layout
----------------
models/
  xg_model.joblib
  xt_model.joblib
  p_scores_model.joblib
  p_concedes_model.joblib
  metadata.yaml            ← training date, feature count, metrics

Usage
-----
>>> registry = ModelRegistry()
>>> registry.save("xg", my_xg_model)
>>> model = registry.load("xg")
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import joblib
import yaml

from football_ai.config import settings
from football_ai.logger import get_logger
from football_ai.utils import ensure_dir

logger = get_logger(__name__)

_VALID_NAMES: frozenset[str] = frozenset(
    {"xg", "xt", "p_scores", "p_concedes", "match_outcome"}
)


class ModelRegistry:
    """
    Filesystem-backed model registry using joblib serialisation.

    All models are stored in the ``models/`` directory at the project root.
    A ``metadata.yaml`` file tracks training timestamps and evaluation metrics.
    """

    def __init__(self, models_dir: str | Path | None = None) -> None:
        self.models_dir = Path(models_dir or settings.abs(settings.paths.models))
        ensure_dir(self.models_dir)

    # ── Save / load ────────────────────────────────────────────────────────────

    def save(
        self,
        name: str,
        model: Any,
        metrics: dict[str, float] | None = None,
        feature_cols: list[str] | None = None,
    ) -> Path:
        """
        Serialise a model to disk and update metadata.

        Parameters
        ----------
        name : str
            One of ``{"xg", "xt", "p_scores", "p_concedes"}``.
        model : Any
            A fitted sklearn-compatible model or numpy array.
        metrics : dict[str, float], optional
            Evaluation metrics to store alongside the model.
        feature_cols : list[str], optional
            Ordered list of feature column names the model was trained on.

        Returns
        -------
        Path
            Path to the saved ``.joblib`` file.
        """
        self._validate_name(name)
        path = self._model_path(name)
        joblib.dump(model, path)
        logger.info("Saved model '%s' -> %s", name, path)
        self._update_metadata(name, metrics or {}, feature_cols or [])
        return path

    def load(self, name: str) -> Any:
        """
        Load a model from disk.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        self._validate_name(name)
        path = self._model_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Model '{name}' not found at {path}. "
                "Run the training pipeline first."
            )
        model = joblib.load(path)
        logger.info("Loaded model '%s' <- %s", name, path)
        return model

    def exists(self, name: str) -> bool:
        """Return True if a serialised model file exists for ``name``."""
        self._validate_name(name)
        return self._model_path(name).exists()

    def list_available(self) -> list[str]:
        """Return names of all models currently saved to disk."""
        return [
            name for name in _VALID_NAMES
            if self._model_path(name).exists()
        ]

    # ── Metadata ───────────────────────────────────────────────────────────────

    def get_metadata(self) -> dict[str, Any]:
        """Load and return the full metadata YAML as a dict."""
        meta_path = self.models_dir / "metadata.yaml"
        if not meta_path.exists():
            return {}
        with open(meta_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def get_feature_cols(self, name: str) -> list[str]:
        """Return the feature column list recorded at training time for ``name``."""
        meta = self.get_metadata()
        return meta.get(name, {}).get("feature_cols", [])

    # ── Private ────────────────────────────────────────────────────────────────

    def _model_path(self, name: str) -> Path:
        return self.models_dir / f"{name}_model.joblib"

    @staticmethod
    def _validate_name(name: str) -> None:
        if name not in _VALID_NAMES:
            raise ValueError(
                f"Unknown model name '{name}'. Valid names: {sorted(_VALID_NAMES)}"
            )

    def _update_metadata(
        self,
        name: str,
        metrics: dict[str, float],
        feature_cols: list[str],
    ) -> None:
        meta = self.get_metadata()
        meta[name] = {
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "feature_count": len(feature_cols),
            "feature_cols": feature_cols,
            "metrics": metrics,
        }
        meta_path = self.models_dir / "metadata.yaml"
        with open(meta_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(meta, fh, default_flow_style=False, sort_keys=True)
        logger.debug("Metadata updated for '%s'", name)
