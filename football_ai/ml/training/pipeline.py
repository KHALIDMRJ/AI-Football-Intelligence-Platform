"""
End-to-end ML training pipeline.

Orchestrates training of all four models (xG, xT, P_scores, P_concedes)
from the feature Parquet files produced by Phase 4.

The pipeline:
1. Loads all available feature Parquet files
2. Concatenates them into a single training DataFrame
3. Trains each requested model via the per-model trainer functions
4. Saves each model to the ModelRegistry
5. Returns a summary report dict

Usage
-----
>>> pipeline = MLTrainingPipeline()
>>> report = pipeline.run(models=["xg", "xt", "p_scores", "p_concedes"])
>>> print(report)
"""

from __future__ import annotations

import pandas as pd

from football_ai.ingestion.storage.parquet_store import ParquetStore
from football_ai.logger import get_logger
from football_ai.ml.serving.model_registry import ModelRegistry
from football_ai.ml.training.trainer import (
    train_p_concedes,
    train_p_scores,
    train_xg,
    train_xt,
)
from football_ai.utils import timer

logger = get_logger(__name__)

_ALL_MODELS: list[str] = ["xg", "xt", "p_scores", "p_concedes"]


class MLTrainingPipeline:
    """
    Coordinates training of all ML models from the Parquet feature store.

    Attributes
    ----------
    store : ParquetStore
        Data store used to load feature and SPADL Parquet files.
    registry : ModelRegistry
        Where trained models are saved.
    """

    def __init__(
        self,
        store: ParquetStore | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.store = store or ParquetStore()
        self.registry = registry or ModelRegistry()

    def run(
        self,
        models: list[str] | None = None,
        match_ids: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, dict]:
        """
        Run the full training pipeline.

        Parameters
        ----------
        models : list[str], optional
            Subset of ["xg", "xt", "p_scores", "p_concedes"] to train.
            Defaults to all four.
        match_ids : list[str], optional
            Specific match IDs to use for training.  Defaults to all available.
        force : bool
            Re-train even if a saved model already exists in the registry.

        Returns
        -------
        dict[str, dict]
            Keys are model names; values are evaluation metric dicts.
        """
        models_to_train = models or _ALL_MODELS
        invalid = [m for m in models_to_train if m not in _ALL_MODELS]
        if invalid:
            raise ValueError(f"Unknown model name(s): {invalid}. Valid: {_ALL_MODELS}")

        report: dict[str, dict] = {}

        # ── Skip models already trained (unless force=True) ────────────────────
        if not force:
            skip = [m for m in models_to_train if self.registry.exists(m)]
            if skip:
                logger.info(
                    "Skipping already-trained models: %s  (use force=True to retrain)",
                    skip,
                )
            models_to_train = [m for m in models_to_train if m not in skip]

        if not models_to_train:
            logger.info("All requested models already trained. Nothing to do.")
            return {m: self.registry.get_metadata().get(m, {}) for m in (models or _ALL_MODELS)}

        # ── Load training data ─────────────────────────────────────────────────
        with timer("Load feature data"):
            feature_df = self._load_features(match_ids)

        if feature_df.empty:
            raise RuntimeError(
                "No feature data found. Run the pipeline (run_pipeline.py) first."
            )

        logger.info(
            "Training data: %d rows × %d columns from %d matches",
            len(feature_df),
            feature_df.shape[1],
            feature_df["match_id"].nunique() if "match_id" in feature_df.columns else -1,
        )

        # xT needs the raw SPADL DataFrame (not features)
        spadl_df: pd.DataFrame | None = None
        if "xt" in models_to_train:
            with timer("Load SPADL data for xT"):
                spadl_df = self._load_spadl(match_ids)

        # ── Train each model ───────────────────────────────────────────────────
        for model_name in models_to_train:
            logger.info("=" * 50)
            logger.info("Training: %s", model_name)
            logger.info("=" * 50)

            try:
                with timer(f"Train {model_name}"):
                    if model_name == "xg":
                        _, metrics = train_xg(feature_df, self.registry)
                        report["xg"] = metrics

                    elif model_name == "xt":
                        if spadl_df is None or spadl_df.empty:
                            raise RuntimeError("SPADL data required for xT but not found.")
                        _, metrics = train_xt(spadl_df, self.registry)
                        report["xt"] = metrics

                    elif model_name == "p_scores":
                        _, metrics = train_p_scores(feature_df, self.registry)
                        report["p_scores"] = metrics

                    elif model_name == "p_concedes":
                        _, metrics = train_p_concedes(feature_df, self.registry)
                        report["p_concedes"] = metrics

            except Exception as exc:
                logger.error("Failed to train %s: %s", model_name, exc, exc_info=True)
                report[model_name] = {"error": str(exc)}

        logger.info("Training complete. Models saved to %s", self.registry.models_dir)
        return report

    # ── Data loading helpers ───────────────────────────────────────────────────

    def _load_features(self, match_ids: list[str] | None) -> pd.DataFrame:
        """Load and concatenate feature Parquet files."""
        available = self.store.list_feature_matches()
        if not available:
            return pd.DataFrame()

        ids = match_ids if match_ids else available
        missing = [m for m in ids if m not in available]
        if missing:
            logger.warning("Feature files not found for match IDs: %s", missing)
        ids = [m for m in ids if m in available]

        if not ids:
            return pd.DataFrame()

        dfs = [self.store.load_features(m) for m in ids]
        return pd.concat(dfs, ignore_index=True)

    def _load_spadl(self, match_ids: list[str] | None) -> pd.DataFrame:
        """Load and concatenate SPADL Parquet files."""
        available = self.store.list_processed_matches()
        if not available:
            return pd.DataFrame()

        ids = match_ids if match_ids else available
        ids = [m for m in ids if m in available]
        if not ids:
            return pd.DataFrame()

        dfs = [self.store.load_spadl_actions(m) for m in ids]
        return pd.concat(dfs, ignore_index=True)
