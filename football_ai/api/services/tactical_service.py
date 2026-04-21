"""
Tactical service.

Business logic for team and tactical endpoints.
Reads weakness Parquet files, pressure Parquet files, player summaries,
and the tactical report JSON to build team profiles and tactical responses.
"""

from __future__ import annotations

import pandas as pd

from football_ai.api.schemas.responses import (
    FormationInfo,
    PressureZoneItem,
    TacticalReportResponse,
    TeamDetailResponse,
    ZoneWeaknessItem,
)
from football_ai.api.services.match_service import MatchService
from football_ai.api.services.player_service import PlayerService
from football_ai.constants import Cols
from football_ai.ingestion.storage.parquet_store import ParquetStore
from football_ai.logger import get_logger

logger = get_logger(__name__)


class TacticalService:
    """
    Provides team weakness, formation, and pressure-map data.
    """

    def __init__(self, store: ParquetStore | None = None) -> None:
        self.store          = store or ParquetStore()
        self.match_service  = MatchService(store=self.store)
        self.player_service = PlayerService(store=self.store)

    # ── Teams list ─────────────────────────────────────────────────────────────

    def get_teams(self, match_id: str) -> list[TeamDetailResponse]:
        """Return all team profiles for a match."""
        vaep_path = self.store.processed_dir / f"match_{match_id}_vaep.parquet"
        if not vaep_path.exists():
            raise KeyError(
                f"Match '{match_id}' not found. Run the pipeline first."
            )
        vaep_df   = pd.read_parquet(vaep_path)
        team_ids  = vaep_df[Cols.TEAM_ID].unique().tolist()
        return [self.get_team(match_id, str(tid)) for tid in team_ids]

    # ── Single team ────────────────────────────────────────────────────────────

    def get_team(self, match_id: str, team_id: str) -> TeamDetailResponse:
        """
        Return a team profile for a single team in a match.

        Raises
        ------
        KeyError
            If the match or team is not found.
        """
        team_summary = self._load_team_summary(match_id)

        row_df = team_summary[
            team_summary[Cols.TEAM_ID].astype(str) == str(team_id)
        ] if not team_summary.empty else pd.DataFrame()

        if row_df.empty:
            raise KeyError(
                f"Team '{team_id}' not found in match '{match_id}'."
            )

        row = row_df.iloc[0]

        # Weaknesses
        weaknesses = self._load_weaknesses(match_id, team_id)

        # Formation from tactical report
        formation, conf = self._team_formation(match_id, team_id)

        # Critical zone count
        n_critical = (
            int((weaknesses["risk_level"] == "critical").sum())
            if not weaknesses.empty and "risk_level" in weaknesses.columns
            else 0
        )

        weak_items = self._weakness_items(weaknesses, limit=20)

        return TeamDetailResponse(
            team_id=str(team_id),
            team_name=str(row.get(Cols.TEAM_NAME, team_id)),
            match_id=match_id,
            action_count=int(row.get("action_count", 0)),
            vaep_total=float(row.get("vaep_total", 0.0)),
            vaep_offensive=float(row.get("vaep_offensive", 0.0)),
            vaep_defensive=float(row.get("vaep_defensive", 0.0)),
            xg_total=float(row.get("xg_total", 0.0)),
            formation=formation,
            n_critical_zones=n_critical,
            weaknesses=weak_items,
        )

    # ── Tactical report ────────────────────────────────────────────────────────

    def get_tactical_report(self, match_id: str) -> TacticalReportResponse:
        """
        Assemble the full tactical intelligence response for a match.

        Raises
        ------
        KeyError
            If no VAEP data exists for the match.
        """
        # Load match detail for team IDs and high-level stats
        match_detail = self.match_service.get_match_detail(match_id)
        home_id = match_detail.home_team_id
        away_id = match_detail.away_team_id

        # Formations
        formations = self._all_formations(match_id)

        # Weaknesses
        home_weak = self._weakness_items(
            self._load_weaknesses(match_id, home_id), limit=10
        )
        away_weak = self._weakness_items(
            self._load_weaknesses(match_id, away_id), limit=10
        )

        # Pressure zones (opponent threat map)
        home_pressure = self._pressure_items(match_id, home_id, limit=10)
        away_pressure = self._pressure_items(match_id, away_id, limit=10)

        # Top players
        top_players_raw, _ = self.player_service.get_rankings(
            match_id=match_id, metric="vaep_total", limit=10
        )

        return TacticalReportResponse(
            match_id=match_id,
            home_team_name=match_detail.home_team_name,
            away_team_name=match_detail.away_team_name,
            xg_home=match_detail.xg_home,
            xg_away=match_detail.xg_away,
            vaep_home=match_detail.vaep_home,
            vaep_away=match_detail.vaep_away,
            formations=formations,
            home_weaknesses=home_weak,
            away_weaknesses=away_weak,
            home_pressure_zones=home_pressure,
            away_pressure_zones=away_pressure,
            top_players=top_players_raw,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_team_summary(self, match_id: str) -> pd.DataFrame:
        path = self.store.processed_dir / f"match_{match_id}_team_summary.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def _load_weaknesses(self, match_id: str, team_id: str) -> pd.DataFrame:
        path = (
            self.store.processed_dir
            / f"match_{match_id}_weaknesses_{team_id}.parquet"
        )
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def _load_pressure(self, match_id: str, team_id: str) -> pd.DataFrame:
        path = (
            self.store.processed_dir
            / f"match_{match_id}_pressure_{team_id}.parquet"
        )
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def _team_formation(
        self, match_id: str, team_id: str
    ) -> tuple[str, float]:
        try:
            report = self.match_service.get_match_report(match_id)
            home_id = report.get("home_team_id", "")
            if str(team_id) == str(home_id):
                return report.get("home_formation", "unknown"), 0.0
            return report.get("away_formation", "unknown"), 0.0
        except (KeyError, Exception):
            return "unknown", 0.0

    def _all_formations(self, match_id: str) -> list[FormationInfo]:
        try:
            report = self.match_service.get_match_report(match_id)
            items = []
            for side, id_key, name_key, form_key in [
                ("home", "home_team_id", "home_team_name", "home_formation"),
                ("away", "away_team_id", "away_team_name", "away_formation"),
            ]:
                items.append(FormationInfo(
                    team_id=str(report.get(id_key, "")),
                    team_name=str(report.get(name_key, "")),
                    formation=str(report.get(form_key, "unknown")),
                    confidence=0.0,
                ))
            return items
        except (KeyError, Exception):
            return []

    @staticmethod
    def _weakness_items(
        df: pd.DataFrame, limit: int = 20
    ) -> list[ZoneWeaknessItem]:
        if df.empty:
            return []
        required = ["zone_id", "zone_x", "zone_y",
                    "mean_opponent_vaep", "action_count", "risk_level"]
        if not all(c in df.columns for c in required):
            return []
        top = df.head(limit)
        return [
            ZoneWeaknessItem(
                zone_id=int(row["zone_id"]),
                zone_x=float(row["zone_x"]),
                zone_y=float(row["zone_y"]),
                mean_opponent_vaep=float(row["mean_opponent_vaep"]),
                action_count=int(row["action_count"]),
                risk_level=str(row["risk_level"]),
            )
            for _, row in top.iterrows()
        ]

    def _pressure_items(
        self, match_id: str, team_id: str, limit: int = 10
    ) -> list[PressureZoneItem]:
        df = self._load_pressure(match_id, team_id)
        if df.empty or "threat" not in df.columns:
            return []
        top = df.nlargest(limit, "threat")
        return [
            PressureZoneItem(
                zone_id=int(row.get("zone_id", 0)),
                zone_x=float(row.get("zone_x", 0.0)),
                zone_y=float(row.get("zone_y", 0.0)),
                threat=float(row.get("threat", 0.0)),
            )
            for _, row in top.iterrows()
        ]
