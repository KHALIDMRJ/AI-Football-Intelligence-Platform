"""
Dashboard data loader.

Wraps ParquetStore and tactical JSON access behind Streamlit's
``@st.cache_data`` decorator so expensive Parquet reads happen only
once per session.

All public functions return plain Python dicts/lists or pandas
DataFrames — never Pydantic models — so Streamlit can render them
without extra dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from football_ai.config import settings
from football_ai.constants import Cols
from football_ai.ingestion.storage.parquet_store import ParquetStore
from football_ai.logger import get_logger

logger = get_logger(__name__)

_STORE = ParquetStore()
_PROCESSED = _STORE.processed_dir


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parquet(path: Path) -> pd.DataFrame | None:
    """Return a DataFrame or None if the file doesn't exist."""
    if path.exists():
        return pd.read_parquet(path)
    return None


def _json_report(match_id: str) -> dict[str, Any] | None:
    path = _PROCESSED / f"match_{match_id}_tactical_report.json"
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


# ── Match discovery ───────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def list_match_ids() -> list[str]:
    """Return all match IDs that have a VAEP Parquet file."""
    files = sorted(_PROCESSED.glob("match_*_vaep.parquet"))
    ids = []
    for f in files:
        stem  = f.stem                  # match_{id}_vaep
        parts = stem.split("_")
        if len(parts) >= 3:
            ids.append("_".join(parts[1:-1]))
    return ids


# ── VAEP data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_vaep(match_id: str) -> pd.DataFrame | None:
    """Load per-action VAEP DataFrame for a match."""
    return _parquet(_PROCESSED / f"match_{match_id}_vaep.parquet")


@st.cache_data(ttl=300, show_spinner=False)
def load_player_summary(match_id: str) -> pd.DataFrame | None:
    """Load per-player VAEP summary for a match."""
    return _parquet(_PROCESSED / f"match_{match_id}_player_summary.parquet")


@st.cache_data(ttl=300, show_spinner=False)
def load_team_summary(match_id: str) -> pd.DataFrame | None:
    """Load per-team VAEP summary for a match."""
    return _parquet(_PROCESSED / f"match_{match_id}_team_summary.parquet")


@st.cache_data(ttl=300, show_spinner=False)
def load_tactical_report(match_id: str) -> dict[str, Any] | None:
    """Load the tactical JSON report for a match."""
    return _json_report(match_id)


@st.cache_data(ttl=300, show_spinner=False)
def load_weaknesses(match_id: str, team_id: str) -> pd.DataFrame | None:
    """Load weakness Parquet for a team in a match."""
    return _parquet(
        _PROCESSED / f"match_{match_id}_weaknesses_{team_id}.parquet"
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_pressure(match_id: str, team_id: str) -> pd.DataFrame | None:
    """Load pressure (opponent threat) Parquet for a team in a match."""
    return _parquet(
        _PROCESSED / f"match_{match_id}_pressure_{team_id}.parquet"
    )


# ── Derived helpers ───────────────────────────────────────────────────────────

def get_team_ids(match_id: str) -> list[str]:
    """Return the two team IDs present in a match."""
    vaep = load_vaep(match_id)
    if vaep is None or Cols.TEAM_ID not in vaep.columns:
        return []
    return [str(t) for t in vaep[Cols.TEAM_ID].unique().tolist()]


def get_team_name(match_id: str, team_id: str) -> str:
    """Look up a team's display name from the VAEP DataFrame."""
    vaep = load_vaep(match_id)
    if vaep is None:
        return team_id
    rows = vaep[vaep[Cols.TEAM_ID].astype(str) == str(team_id)]
    if not rows.empty and Cols.TEAM_NAME in rows.columns:
        return str(rows[Cols.TEAM_NAME].iloc[0])
    return team_id


def match_label(match_id: str) -> str:
    """Human-readable match label: 'Home FC vs Away FC (match_id)'."""
    team_ids = get_team_ids(match_id)
    if len(team_ids) >= 2:
        h = get_team_name(match_id, team_ids[0])
        a = get_team_name(match_id, team_ids[1])
        return f"{h} vs {a}  [{match_id}]"
    return match_id


def pipeline_status() -> dict[str, Any]:
    """
    Return a dict describing which pipeline phases have been completed
    based on which artefact files exist on disk.
    """
    match_ids = list_match_ids()
    if not match_ids:
        return {
            "matches_processed": 0,
            "vaep_computed":     False,
            "tactical_computed": False,
            "models_trained":    False,
        }
    mid = match_ids[0]
    tactical_path = _PROCESSED / f"match_{mid}_tactical_report.json"
    model_dir     = settings.abs(settings.paths.models)
    models_ready  = (model_dir / "p_scores_model.joblib").exists()

    return {
        "matches_processed": len(match_ids),
        "vaep_computed":     (_PROCESSED / f"match_{mid}_vaep.parquet").exists(),
        "tactical_computed": tactical_path.exists(),
        "models_trained":    models_ready,
    }
