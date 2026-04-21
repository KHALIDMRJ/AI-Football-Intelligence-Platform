"""
Feature assembler.

Combines all feature groups (spatial, temporal, contextual) into a single
feature matrix ready for model training and inference.

The assembler:
1. Calls each feature module in order
2. Selects only the ``f_*`` feature columns (drops raw SPADL columns)
3. Fills NaN values with sensible defaults
4. Returns the full feature DataFrame with metadata columns kept for reference
5. Optionally saves the result to the Parquet feature store

The feature column inventory is logged on first assembly so you always
know exactly which features exist and in which order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_ai.features.contextual import compute_contextual_features
from football_ai.features.spatial import compute_spatial_features
from football_ai.features.temporal import compute_temporal_features
from football_ai.ingestion.storage.parquet_store import ParquetStore
from football_ai.logger import get_logger
from football_ai.utils import timer

logger = get_logger(__name__)

# Metadata columns kept alongside features (not used as model inputs)
_META_COLS: list[str] = [
    "match_id",
    "action_id",
    "index",
    "period",
    "timestamp",
    "minute",
    "second",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "position",
    "action_type",
    "body_part",
    "result",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "possession_id",
    "possession_team_id",
    "chain_id",
    "chain_index",
    "zone_id",
    "xg",
    # game-state labels (targets)
    "label_scores",
    "label_concedes",
    "score_home",
    "score_away",
    "score_diff",
]


class FeatureAssembler:
    """
    Builds the full feature matrix from a preprocessed SPADL actions DataFrame.

    Usage
    -----
    >>> assembler = FeatureAssembler()
    >>> feature_df = assembler.assemble(spadl_df, match_id="3813041")

    The returned DataFrame contains:
    - metadata columns (see _META_COLS)
    - feature columns prefixed with f_sp_, f_tm_, f_ctx_
    """

    def __init__(self, store: ParquetStore | None = None) -> None:
        self.store = store or ParquetStore()
        self._feature_cols: list[str] | None = None  # memoized on first run

    def assemble(
        self,
        df: pd.DataFrame,
        match_id: str,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Compute all features for a single match and return the feature DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed SPADL actions (output of PreprocessingPipeline.run()).
        match_id : str
            Used for Parquet storage keying.
        force : bool
            Recompute even if the feature Parquet already exists.

        Returns
        -------
        pd.DataFrame
            Feature matrix with metadata columns prepended.
        """
        if not force and self.store.has_features(match_id):
            logger.info("Features for match %s already exist -- loading from store.", match_id)
            return self.store.load_features(match_id)

        logger.info("Computing features for match %s (%d actions).", match_id, len(df))

        if df.empty:
            logger.warning("assemble() called with empty DataFrame -- returning empty.")
            self.store.save_features(df, match_id)
            return df

        with timer("spatial features"):
            df = compute_spatial_features(df)

        with timer("temporal features"):
            df = compute_temporal_features(df)

        with timer("contextual features"):
            df = compute_contextual_features(df)

        # Collect feature columns (all f_* columns)
        feature_cols = sorted([c for c in df.columns if c.startswith("f_")])
        if self._feature_cols is None:
            self._feature_cols = feature_cols
            logger.info(
                "Feature inventory: %d features. First 10: %s",
                len(feature_cols),
                feature_cols[:10],
            )

        # Fill remaining NaNs with 0 (safe for all feature types)
        df[feature_cols] = df[feature_cols].fillna(0.0)

        # Clamp float features to prevent inf/nan bleeding through
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0.0)

        # Keep only meta + feature columns
        keep_meta = [c for c in _META_COLS if c in df.columns]
        output_df = df[keep_meta + feature_cols].copy()

        # Save
        self.store.save_features(output_df, match_id)
        logger.info(
            "Features saved: %d rows × %d feature cols for match %s.",
            len(output_df),
            len(feature_cols),
            match_id,
        )
        return output_df

    @property
    def feature_columns(self) -> list[str]:
        """
        Return the sorted list of feature column names.
        Only available after at least one call to assemble().
        """
        if self._feature_cols is None:
            raise RuntimeError(
                "feature_columns is only available after calling assemble() once."
            )
        return self._feature_cols

    def get_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the model-input matrix X from a feature DataFrame.
        Drops all metadata and label columns, returns only f_* columns.
        """
        feature_cols = [c for c in df.columns if c.startswith("f_")]
        return df[feature_cols].astype(float)

    def get_labels(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Extract VAEP training labels from a feature DataFrame.

        Returns
        -------
        (y_scores, y_concedes) : tuple[pd.Series, pd.Series]
        """
        from football_ai.constants import Cols
        if Cols.LABEL_SCORES not in df.columns or Cols.LABEL_CONCEDES not in df.columns:
            raise KeyError(
                "label_scores / label_concedes columns not found. "
                "Run GameStateBuilder before FeatureAssembler."
            )
        return df[Cols.LABEL_SCORES].astype(int), df[Cols.LABEL_CONCEDES].astype(int)

    def feature_count(self, df: pd.DataFrame) -> int:
        """Return the number of f_* columns in a feature DataFrame."""
        return len([c for c in df.columns if c.startswith("f_")])
