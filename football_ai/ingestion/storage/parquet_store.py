"""
Parquet-based storage layer for ingested event data.

All raw and processed data is stored as Snappy-compressed Parquet files,
partitioned by match_id. This gives fast, predicate-pushdown reads
in later phases.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_ai.config import settings
from football_ai.logger import get_logger
from football_ai.schemas import RawEvent, SPADLAction
from football_ai.utils import ensure_dir, load_parquet, save_parquet

logger = get_logger(__name__)


class ParquetStore:
    """
    Manages read/write of raw events and processed SPADL actions
    in the Parquet-based data lake.

    Directory layout
    ----------------
    data/
      raw/
        match_{match_id}_raw.parquet
      processed/
        match_{match_id}_spadl.parquet
      features/
        match_{match_id}_features.parquet

    Usage
    -----
    >>> store = ParquetStore()
    >>> store.save_raw_events(events, match_id="3813041")
    >>> df = store.load_raw_events("3813041")
    """

    def __init__(
        self,
        raw_dir: str | Path | None = None,
        processed_dir: str | Path | None = None,
        features_dir: str | Path | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir or settings.abs(settings.paths.data_raw))
        self.processed_dir = Path(
            processed_dir or settings.abs(settings.paths.data_processed)
        )
        self.features_dir = Path(
            features_dir or settings.abs(settings.paths.data_features)
        )

        ensure_dir(self.raw_dir)
        ensure_dir(self.processed_dir)
        ensure_dir(self.features_dir)

    # ── Raw events ─────────────────────────────────────────────────────────────

    def save_raw_events(
        self, events: list[RawEvent], match_id: str
    ) -> Path:
        """Serialise a list of RawEvent objects to Parquet."""
        records = [e.model_dump() for e in events]
        df = pd.DataFrame(records)
        path = self.raw_dir / f"match_{match_id}_raw.parquet"
        return save_parquet(df, path)

    def load_raw_events(self, match_id: str) -> pd.DataFrame:
        """Load the raw event Parquet for a match."""
        path = self.raw_dir / f"match_{match_id}_raw.parquet"
        return load_parquet(path)

    # ── SPADL actions ─────────────────────────────────────────────────────────

    def save_spadl_actions(
        self, actions: list[SPADLAction] | pd.DataFrame, match_id: str
    ) -> Path:
        """Serialise normalised SPADL actions to Parquet."""
        if isinstance(actions, pd.DataFrame):
            df = actions
        else:
            df = pd.DataFrame([a.model_dump() for a in actions])
        path = self.processed_dir / f"match_{match_id}_spadl.parquet"
        return save_parquet(df, path)

    def load_spadl_actions(self, match_id: str) -> pd.DataFrame:
        """Load SPADL actions for a match."""
        path = self.processed_dir / f"match_{match_id}_spadl.parquet"
        return load_parquet(path)

    # ── Features ──────────────────────────────────────────────────────────────

    def save_features(self, df: pd.DataFrame, match_id: str) -> Path:
        """Save the feature matrix for a match."""
        path = self.features_dir / f"match_{match_id}_features.parquet"
        return save_parquet(df, path)

    def load_features(self, match_id: str) -> pd.DataFrame:
        """Load the feature matrix for a match."""
        path = self.features_dir / f"match_{match_id}_features.parquet"
        return load_parquet(path)

    # ── Discovery ─────────────────────────────────────────────────────────────

    def list_raw_matches(self) -> list[str]:
        """Return all match IDs for which raw data exists."""
        return self._extract_match_ids(self.raw_dir, suffix="_raw.parquet")

    def list_processed_matches(self) -> list[str]:
        """Return all match IDs that have been normalised to SPADL."""
        return self._extract_match_ids(
            self.processed_dir, suffix="_spadl.parquet"
        )

    def list_feature_matches(self) -> list[str]:
        """Return all match IDs for which features have been computed."""
        return self._extract_match_ids(
            self.features_dir, suffix="_features.parquet"
        )

    def has_raw(self, match_id: str) -> bool:
        return (self.raw_dir / f"match_{match_id}_raw.parquet").exists()

    def has_processed(self, match_id: str) -> bool:
        return (self.processed_dir / f"match_{match_id}_spadl.parquet").exists()

    def has_features(self, match_id: str) -> bool:
        return (self.features_dir / f"match_{match_id}_features.parquet").exists()

    # ── Combined load (all matches) ───────────────────────────────────────────

    def load_all_spadl(self) -> pd.DataFrame:
        """
        Load and concatenate SPADL actions across all processed matches.
        Useful for training models on the full dataset.
        """
        match_ids = self.list_processed_matches()
        if not match_ids:
            raise FileNotFoundError("No processed matches found. Run the pipeline first.")

        dfs = [self.load_spadl_actions(mid) for mid in match_ids]
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(
            "Loaded %d actions across %d matches", len(combined), len(match_ids)
        )
        return combined

    def load_all_features(self) -> pd.DataFrame:
        """Load and concatenate feature matrices across all matches."""
        match_ids = self.list_feature_matches()
        if not match_ids:
            raise FileNotFoundError(
                "No feature files found. Run the pipeline first."
            )

        dfs = [self.load_features(mid) for mid in match_ids]
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(
            "Loaded feature matrix: %d rows × %d cols across %d matches",
            len(combined),
            combined.shape[1],
            len(match_ids),
        )
        return combined

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_match_ids(directory: Path, suffix: str) -> list[str]:
        """Parse match IDs from filenames of form match_{id}{suffix}."""
        prefix = "match_"
        return [
            f.name[len(prefix): -len(suffix)]
            for f in directory.glob(f"{prefix}*{suffix}")
        ]
