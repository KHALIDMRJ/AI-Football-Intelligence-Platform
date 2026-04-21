"""
Match service.

All business logic for match-related endpoints lives here.
Routers call services; services call the Parquet store and tactical pipeline.
No pandas DataFrames are returned — only typed Pydantic-compatible dicts
or lists that routers can serialise directly.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from football_ai.api.schemas.responses import (
    ActionItem,
    MatchDetailResponse,
    MatchItem,
    TeamVAEPItem,
)
from football_ai.constants import Cols
from football_ai.ingestion.storage.parquet_store import ParquetStore
from football_ai.logger import get_logger

logger = get_logger(__name__)


class MatchService:
    """
    Provides match-level data by reading processed Parquet artefacts.

    All methods raise ``KeyError`` when a match_id is not found so that
    routers can convert it to an HTTP 404.
    """

    def __init__(self, store: ParquetStore | None = None) -> None:
        self.store = store or ParquetStore()

    # ── Discovery ──────────────────────────────────────────────────────────────

    def list_match_ids(self) -> list[str]:
        """Return all match IDs that have VAEP computed (Phase 6 complete)."""
        processed = self.store.processed_dir
        parquets = list(processed.glob("match_*_vaep.parquet"))
        ids = []
        for p in parquets:
            # Filename: match_{id}_vaep.parquet
            stem = p.stem  # match_{id}_vaep
            parts = stem.split("_")
            # Extract ID between "match_" and "_vaep"
            if len(parts) >= 3:
                ids.append("_".join(parts[1:-1]))
        return sorted(ids)

    # ── Match list ─────────────────────────────────────────────────────────────

    def get_match_list(self) -> list[MatchItem]:
        """Return a compact summary for every processed match."""
        match_ids = self.list_match_ids()
        result = []
        for mid in match_ids:
            try:
                detail = self.get_match_detail(mid)
                result.append(MatchItem(
                    match_id=detail.match_id,
                    home_team_id=detail.home_team_id,
                    away_team_id=detail.away_team_id,
                    home_team_name=detail.home_team_name,
                    away_team_name=detail.away_team_name,
                    total_actions=detail.total_actions,
                    xg_home=detail.xg_home,
                    xg_away=detail.xg_away,
                    vaep_home=detail.vaep_home,
                    vaep_away=detail.vaep_away,
                ))
            except Exception as exc:
                logger.warning("Could not load match %s: %s", mid, exc)
        return result

    # ── Match detail ───────────────────────────────────────────────────────────

    def get_match_detail(self, match_id: str) -> MatchDetailResponse:
        """
        Return full match analysis for a single match.

        Raises
        ------
        KeyError
            If VAEP Parquet does not exist for this match_id.
        """
        vaep_df = self._load_vaep(match_id)

        team_ids   = vaep_df[Cols.TEAM_ID].unique().tolist()
        home_id    = str(team_ids[0]) if team_ids else "unknown"
        away_id    = str(team_ids[1]) if len(team_ids) > 1 else "unknown"
        home_name  = self._team_name(vaep_df, home_id)
        away_name  = self._team_name(vaep_df, away_id)

        # xG per team
        xg_home = self._team_xg(vaep_df, home_id)
        xg_away = self._team_xg(vaep_df, away_id)

        # VAEP per team (from team summary)
        team_summary = self._load_team_summary(match_id)
        vaep_home = self._team_vaep(team_summary, home_id)
        vaep_away = self._team_vaep(team_summary, away_id)

        # Formation from tactical report JSON
        home_formation, away_formation = self._formations(match_id, home_id, away_id)

        # Team stats
        team_stats = self._build_team_stats(team_summary)

        # Most valuable actions (top 5)
        most_valuable = self._top_actions(vaep_df, n=5)

        return MatchDetailResponse(
            match_id=match_id,
            home_team_id=home_id,
            away_team_id=away_id,
            home_team_name=home_name,
            away_team_name=away_name,
            total_actions=len(vaep_df),
            xg_home=round(xg_home, 4),
            xg_away=round(xg_away, 4),
            vaep_home=round(vaep_home, 4),
            vaep_away=round(vaep_away, 4),
            home_formation=home_formation,
            away_formation=away_formation,
            team_stats=team_stats,
            most_valuable_actions=most_valuable,
        )

    def get_match_report(self, match_id: str) -> dict[str, Any]:
        """
        Return the raw tactical JSON report for a match.

        Raises
        ------
        KeyError
            If the tactical report JSON does not exist.
        """
        path = self.store.processed_dir / f"match_{match_id}_tactical_report.json"
        if not path.exists():
            raise KeyError(
                f"Tactical report not found for match {match_id}. "
                "Run Phase 7 (tactical pipeline) first."
            )
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_vaep(self, match_id: str) -> pd.DataFrame:
        path = self.store.processed_dir / f"match_{match_id}_vaep.parquet"
        if not path.exists():
            raise KeyError(
                f"Match '{match_id}' not found. "
                "Run the pipeline (run_pipeline.py) first."
            )
        return pd.read_parquet(path)

    def _load_team_summary(self, match_id: str) -> pd.DataFrame:
        path = self.store.processed_dir / f"match_{match_id}_team_summary.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    @staticmethod
    def _team_name(df: pd.DataFrame, team_id: str) -> str:
        if Cols.TEAM_NAME in df.columns:
            rows = df[df[Cols.TEAM_ID].astype(str) == str(team_id)]
            if not rows.empty:
                return str(rows[Cols.TEAM_NAME].iloc[0])
        return team_id

    @staticmethod
    def _team_xg(df: pd.DataFrame, team_id: str) -> float:
        if Cols.XG not in df.columns:
            return 0.0
        return float(df[df[Cols.TEAM_ID].astype(str) == str(team_id)][Cols.XG].sum())

    @staticmethod
    def _team_vaep(team_summary: pd.DataFrame, team_id: str) -> float:
        if team_summary.empty or "vaep_total" not in team_summary.columns:
            return 0.0
        row = team_summary[team_summary[Cols.TEAM_ID].astype(str) == str(team_id)]
        return float(row["vaep_total"].iloc[0]) if not row.empty else 0.0

    def _formations(
        self, match_id: str, home_id: str, away_id: str
    ) -> tuple[str, str]:
        try:
            report = self.get_match_report(match_id)
            return (
                report.get("home_formation", "unknown"),
                report.get("away_formation", "unknown"),
            )
        except KeyError:
            return "unknown", "unknown"

    @staticmethod
    def _build_team_stats(team_summary: pd.DataFrame) -> list[TeamVAEPItem]:
        if team_summary.empty:
            return []
        items = []
        for _, row in team_summary.iterrows():
            items.append(TeamVAEPItem(
                team_id=str(row.get(Cols.TEAM_ID, "")),
                team_name=str(row.get(Cols.TEAM_NAME, "")),
                vaep_total=float(row.get("vaep_total", 0.0)),
                vaep_offensive=float(row.get("vaep_offensive", 0.0)),
                vaep_defensive=float(row.get("vaep_defensive", 0.0)),
                xg_total=float(row.get("xg_total", 0.0)),
            ))
        return items

    @staticmethod
    def _top_actions(df: pd.DataFrame, n: int = 5) -> list[ActionItem]:
        if df.empty or Cols.VAEP_VALUE not in df.columns:
            return []
        top = df.nlargest(n, Cols.VAEP_VALUE)
        items = []
        for _, row in top.iterrows():
            items.append(ActionItem(
                action_id=str(row.get(Cols.ACTION_ID, "")),
                action_type=str(row.get(Cols.ACTION_TYPE, "")),
                player_name=str(row.get(Cols.PLAYER_NAME, "")),
                team_name=str(row.get(Cols.TEAM_NAME, "")),
                minute=int(row.get(Cols.MINUTE, 0)),
                vaep_value=round(float(row.get(Cols.VAEP_VALUE, 0.0)), 4),
                start_x=round(float(row.get(Cols.START_X, 0.0)), 1),
                start_y=round(float(row.get(Cols.START_Y, 0.0)), 1),
            ))
        return items
