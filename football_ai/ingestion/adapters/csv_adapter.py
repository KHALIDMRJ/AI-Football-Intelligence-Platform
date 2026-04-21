"""
CSV adapter for StatsBomb-format event data.

Handles the column mapping from StatsBomb's CSV export format
to the internal RawEvent schema.

Design decision: only ONE source column is mapped to each target column.
The adapter resolves conflicts (e.g. both event_type_name and type_name
exist in fus_FAR.csv) by priority: more-specific names win.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from football_ai.ingestion.adapters.base_adapter import BaseAdapter
from football_ai.logger import get_logger
from football_ai.schemas import RawEvent

logger = get_logger(__name__)


# Priority-ordered column candidates for each internal field.
# The first match found in the DataFrame wins; others are ignored.
_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "match_id":        ["match_id"],
    "index":           ["index"],
    "period":          ["period"],
    "timestamp":       ["timestamp"],
    "minute":          ["minute"],
    "second":          ["second"],
    "event_type":      ["event_type_name", "type_name"],   # priority order
    "team_id":         ["team_id"],
    "team_name":       ["team_name"],
    "possession_team_id":   ["possession_team_id"],
    "possession_team_name": ["possession_team_name"],
    "player_id":       ["player_id"],
    "player_name":     ["player_name"],
    "player_position": ["player_position_name"],
    "start_x":         ["location_x"],
    "start_y":         ["location_y"],
    "end_x":           ["end_location_x"],
    "end_y":           ["end_location_y"],
    "duration":        ["duration"],
    "under_pressure":  ["under_pressure"],
    "outcome":         ["outcome_name"],
    "body_part":       ["body_part_name"],
    "xg":              ["statsbomb_xg"],
}

_REQUIRED_INTERNAL: list[str] = ["match_id", "period", "minute", "second"]


class CSVAdapter(BaseAdapter):
    """
    Adapter for StatsBomb-format CSV event files.

    Supports both direct StatsBomb Open Data CSV exports and the
    fus_FAR.csv dataset format used in this project.

    Usage
    -----
    >>> adapter = CSVAdapter("data/raw/fus_FAR.csv")
    >>> events = adapter.ingest()
    """

    source_name = "StatsBomb CSV"

    def __init__(
        self,
        source_path: str | Path,
        match_id_override: str | None = None,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(source_path)
        self.match_id_override = match_id_override
        self.encoding = encoding

    # ── BaseAdapter interface ──────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Load CSV, apply priority-based column renaming, and coerce types."""
        df = pd.read_csv(self.source_path, encoding=self.encoding, low_memory=False)
        logger.debug("Raw CSV shape: %s -- columns: %s", df.shape, list(df.columns))
        df = self._resolve_columns(df)
        df = self._coerce_types(df)
        return df

    def to_raw_events(self, df: pd.DataFrame) -> list[RawEvent]:
        """
        Convert the normalised DataFrame into RawEvent objects.

        Deduplicates freeze-frame-expanded rows (StatsBomb repeats a row
        for each player in the freeze frame). Skips and logs invalid rows.
        """
        df = self._deduplicate(df)

        if self.match_id_override:
            df["match_id"] = self.match_id_override

        events: list[RawEvent] = []
        errors = 0

        for _, row in df.iterrows():
            try:
                event = RawEvent(**self._row_to_dict(row))
                events.append(event)
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    logger.debug(
                        "Skipping invalid row (index=%s): %s", row.get("index"), exc
                    )

        if errors:
            logger.warning("Skipped %d invalid rows out of %d total", errors, len(df))

        return events

    # ── Private helpers ────────────────────────────────────────────────────────

    def _resolve_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a clean DataFrame with one column per internal field.

        For each internal field, iterate through the candidate source columns
        in priority order and use the first one present. This guarantees
        no duplicate columns in the output.
        """
        resolved: dict[str, pd.Series] = {}
        available = set(df.columns)

        for internal_name, candidates in _COLUMN_CANDIDATES.items():
            for candidate in candidates:
                if candidate in available:
                    resolved[internal_name] = df[candidate]
                    break
            # If no candidate found, the internal column is simply absent

        # Keep non-candidate original columns (e.g. possession_id from StatsBomb)
        # but do NOT bring in duplicates of already-resolved fields
        extra_cols = [
            c for c in df.columns
            if c not in {cand for cands in _COLUMN_CANDIDATES.values() for cand in cands}
            and c not in resolved
        ]
        for col in extra_cols:
            resolved[col] = df[col]

        return pd.DataFrame(resolved, index=df.index)

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce each column to the correct type, filling missing values safely."""
        numeric_float = ["start_x", "start_y", "end_x", "end_y", "duration", "xg"]
        for col in numeric_float:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Clamp coordinates to pitch bounds
        for col, hi in [("start_x", 120.0), ("end_x", 120.0),
                         ("start_y",  80.0), ("end_y",  80.0)]:
            if col in df.columns:
                df[col] = df[col].clip(0.0, hi)

        # xG: clip to [0, 1]
        if "xg" in df.columns:
            df["xg"] = df["xg"].clip(0.0, 1.0)

        # Boolean
        if "under_pressure" in df.columns:
            df["under_pressure"] = (
                df["under_pressure"]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .fillna(False)
                .astype(bool)
            )

        # Integers
        for col in ["period", "minute", "second", "index"]:
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                )

        # Strings (replace literal "nan" / "None" with Python None)
        str_cols = [
            "match_id", "team_id", "team_name", "player_id", "player_name",
            "event_type", "outcome", "body_part", "player_position",
        ]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace({"nan": None, "None": None})

        # Timestamp: ensure string representation
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype(str).fillna("00:00:00.000")
        else:
            df["timestamp"] = "00:00:00.000"

        if "match_id" not in df.columns:
            df["match_id"] = "unknown"

        if "event_type" not in df.columns:
            df["event_type"] = "unknown"

        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        StatsBomb CSVs repeat rows for freeze-frame expanded shots.
        Keep only the first occurrence of each (match_id, index) pair.
        """
        key_cols = [c for c in ["match_id", "index"] if c in df.columns]
        if not key_cols:
            return df
        before = len(df)
        df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
        after = len(df)
        if before != after:
            logger.debug("Deduplicated: %d -> %d rows", before, after)
        return df

    def _row_to_dict(self, row: pd.Series) -> dict:
        """
        Convert a DataFrame row to a plain dict for RawEvent construction.

        Uses scalar extraction to guarantee no Series objects slip through
        (which would cause Pydantic validation to fail silently or crash).
        """
        def _scalar(key: str) -> object:
            """Extract a guaranteed scalar from the row; return None if absent."""
            val = row.get(key)
            if val is None:
                return None
            # Guard: if somehow a Series was stored, take first element
            if isinstance(val, pd.Series):
                val = val.iloc[0] if len(val) > 0 else None
            if isinstance(val, float) and np.isnan(val):
                return None
            if val in ("None", "nan", "NaN", ""):
                return None
            return val

        def _float(key: str, default: float = 0.0) -> float:
            v = _scalar(key)
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        def _int(key: str, default: int = 0) -> int:
            v = _scalar(key)
            try:
                return int(float(v)) if v is not None else default
            except (TypeError, ValueError):
                return default

        def _bool(key: str) -> bool:
            v = _scalar(key)
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            return str(v).lower() in ("true", "1", "yes") if v is not None else False

        def _str(key: str) -> str | None:
            v = _scalar(key)
            return str(v) if v is not None else None

        return {
            "match_id":        str(_scalar("match_id") or "unknown"),
            "index":           _int("index"),
            "period":          max(1, min(5, _int("period", 1))),
            "timestamp":       str(_scalar("timestamp") or "00:00:00.000"),
            "minute":          _int("minute"),
            "second":          max(0, min(59, _int("second"))),
            "event_type":      str(_scalar("event_type") or "unknown"),
            "team_id":         _str("team_id"),
            "team_name":       _str("team_name"),
            "player_id":       _str("player_id"),
            "player_name":     _str("player_name"),
            "player_position": _str("player_position"),
            "start_x":         _float("start_x"),
            "start_y":         _float("start_y"),
            "end_x":           _float("end_x"),
            "end_y":           _float("end_y"),
            "duration":        _float("duration"),
            "under_pressure":  _bool("under_pressure"),
            "outcome":         _str("outcome"),
            "body_part":       _str("body_part"),
            "xg":              min(1.0, max(0.0, _float("xg"))),
        }
