"""
Tactical report builder.

Assembles structured, serialisable reports from all tactical module outputs.
Three report types are produced:

MatchReport
    Top-level match summary: teams, xG, VAEP totals, formations,
    most dangerous possessions, most valuable actions.

PlayerReport
    Individual player profile: VAEP breakdown, ranking, key actions.

TeamReport
    Team-level profile: formation, pressure map summary,
    weakest zones, top players.

All reports are plain Python dataclasses that serialise to dict/JSON
with ``.to_dict()``, making them ready for the FastAPI response layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from football_ai.constants import Cols
from football_ai.logger import get_logger
from football_ai.tactical.formation_analyser import FormationResult
from football_ai.tactical.pressure_map import PressureMapResult

logger = get_logger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ActionSummary:
    """Compact representation of a single notable action."""
    action_id:   str
    action_type: str
    player_name: str
    team_name:   str
    minute:      int
    vaep_value:  float
    start_x:     float
    start_y:     float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlayerReport:
    """Individual player tactical report."""

    player_id:        str
    player_name:      str
    team_id:          str
    team_name:        str
    position:         str
    action_count:     int
    vaep_total:       float
    vaep_offensive:   float
    vaep_defensive:   float
    vaep_per_90:      float | None
    xg_total:         float
    xt_total:         float
    overall_rank:     int
    offensive_rank:   int
    defensive_rank:   int
    tier:             str
    top_actions:      list[ActionSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["top_actions"] = [a.to_dict() for a in self.top_actions]
        return d


@dataclass
class TeamReport:
    """Team-level tactical report."""

    team_id:            str
    team_name:          str
    action_count:       int
    vaep_total:         float
    vaep_offensive:     float
    vaep_defensive:     float
    xg_total:           float
    formation:          str
    formation_conf:     float
    n_high_risk_zones:  int
    n_critical_zones:   int
    weakest_zone_x:     float | None
    weakest_zone_y:     float | None
    top_players:        list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatchReport:
    """Top-level match tactical report."""

    match_id:           str
    home_team_id:       str
    away_team_id:       str
    home_team_name:     str
    away_team_name:     str
    total_actions:      int
    xg_home:            float
    xg_away:            float
    vaep_home:          float
    vaep_away:          float
    home_formation:     str
    away_formation:     str
    home_report:        TeamReport | None = None
    away_report:        TeamReport | None = None
    most_valuable_actions: list[ActionSummary] = field(default_factory=list)
    top_players:        list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.home_report:
            d["home_report"] = self.home_report.to_dict()
        if self.away_report:
            d["away_report"] = self.away_report.to_dict()
        d["most_valuable_actions"] = [a.to_dict() for a in self.most_valuable_actions]
        return d


# ── Builder ───────────────────────────────────────────────────────────────────

class ReportBuilder:
    """
    Assembles MatchReport, PlayerReport, and TeamReport from module outputs.

    Parameters
    ----------
    top_actions_n : int
        Number of most-valuable actions to include in reports. Default 5.
    top_players_n : int
        Number of top players to include in team/match reports. Default 5.

    Usage
    -----
    >>> builder = ReportBuilder()
    >>> match_report = builder.build_match_report(
    ...     vaep_df, player_summary, team_summary,
    ...     home_team_id="2761", away_team_id="2762",
    ...     formations={"2761": formation_result_home, ...},
    ...     weaknesses={"2761": weakness_df, ...},
    ... )
    >>> print(match_report.to_dict())
    """

    def __init__(
        self,
        top_actions_n: int = 5,
        top_players_n: int = 5,
    ) -> None:
        self.top_actions_n = top_actions_n
        self.top_players_n = top_players_n

    # ── Match report ───────────────────────────────────────────────────────────

    def build_match_report(
        self,
        vaep_df: pd.DataFrame,
        player_summary: pd.DataFrame,
        team_summary: pd.DataFrame,
        home_team_id: str,
        away_team_id: str,
        formations: dict[str, FormationResult] | None = None,
        weaknesses: dict[str, pd.DataFrame] | None = None,
        pressure_maps: dict[str, PressureMapResult] | None = None,
    ) -> MatchReport:
        """
        Build a complete match-level tactical report.
        """
        match_id = (
            str(vaep_df[Cols.MATCH_ID].iloc[0])
            if Cols.MATCH_ID in vaep_df.columns and not vaep_df.empty
            else "unknown"
        )

        home_name = self._team_name(vaep_df, home_team_id)
        away_name = self._team_name(vaep_df, away_team_id)

        xg_home = self._team_xg(vaep_df, home_team_id)
        xg_away = self._team_xg(vaep_df, away_team_id)

        vaep_home = self._team_vaep(team_summary, home_team_id)
        vaep_away = self._team_vaep(team_summary, away_team_id)

        home_formation = (
            formations[home_team_id].formation_str
            if formations and home_team_id in formations
            else "unknown"
        )
        away_formation = (
            formations[away_team_id].formation_str
            if formations and away_team_id in formations
            else "unknown"
        )

        # Most valuable actions
        most_valuable = self._top_actions(vaep_df, n=self.top_actions_n)

        # Top players from summary
        top_players = self._top_players_dicts(player_summary, n=self.top_players_n)

        # Per-team reports
        home_report = self._build_team_report(
            vaep_df, team_summary, player_summary, home_team_id,
            formations=formations, weaknesses=weaknesses,
        )
        away_report = self._build_team_report(
            vaep_df, team_summary, player_summary, away_team_id,
            formations=formations, weaknesses=weaknesses,
        )

        report = MatchReport(
            match_id=match_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_team_name=home_name,
            away_team_name=away_name,
            total_actions=len(vaep_df),
            xg_home=round(xg_home, 4),
            xg_away=round(xg_away, 4),
            vaep_home=round(vaep_home, 4),
            vaep_away=round(vaep_away, 4),
            home_formation=home_formation,
            away_formation=away_formation,
            home_report=home_report,
            away_report=away_report,
            most_valuable_actions=most_valuable,
            top_players=top_players,
        )

        logger.info(
            "MatchReport built: %s vs %s | xG %.2f–%.2f | VAEP %.2f–%.2f",
            home_name, away_name, xg_home, xg_away, vaep_home, vaep_away,
        )

        return report

    # ── Player report ──────────────────────────────────────────────────────────

    def build_player_report(
        self,
        vaep_df: pd.DataFrame,
        player_summary: pd.DataFrame,
        player_id: str,
    ) -> PlayerReport | None:
        """
        Build an individual player report.

        Returns None if the player is not found in the summary.
        """
        if player_summary.empty:
            return None

        row = player_summary[
            player_summary[Cols.PLAYER_ID].astype(str) == str(player_id)
        ]
        if row.empty:
            logger.warning("Player %s not found in player_summary.", player_id)
            return None

        r = row.iloc[0]

        # Rank lookups (1-based position in the sorted summary)
        overall_rank  = self._rank_of(player_summary, player_id, "vaep_total")
        off_rank      = self._rank_of(player_summary, player_id, "vaep_offensive")
        def_rank      = self._rank_of(player_summary, player_id, "vaep_defensive")

        # Top actions by this player
        player_actions = vaep_df[
            vaep_df[Cols.PLAYER_ID].astype(str) == str(player_id)
        ]
        top_actions = self._top_actions(player_actions, n=self.top_actions_n)

        return PlayerReport(
            player_id=str(player_id),
            player_name=str(r.get("player_name", player_id)),
            team_id=str(r.get(Cols.TEAM_ID, "")),
            team_name=str(r.get(Cols.TEAM_NAME, "")),
            position=str(r.get("position", "")),
            action_count=int(r.get("action_count", 0)),
            vaep_total=float(r.get("vaep_total", 0.0)),
            vaep_offensive=float(r.get("vaep_offensive", 0.0)),
            vaep_defensive=float(r.get("vaep_defensive", 0.0)),
            vaep_per_90=float(r["vaep_per_90"]) if pd.notna(r.get("vaep_per_90")) else None,
            xg_total=float(r.get("xg_total", 0.0)),
            xt_total=float(r.get("xt_total", 0.0)),
            overall_rank=overall_rank,
            offensive_rank=off_rank,
            defensive_rank=def_rank,
            tier=str(r.get("tier", "unknown")),
            top_actions=top_actions,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_team_report(
        self,
        vaep_df: pd.DataFrame,
        team_summary: pd.DataFrame,
        player_summary: pd.DataFrame,
        team_id: str,
        formations: dict | None = None,
        weaknesses: dict | None = None,
    ) -> TeamReport | None:
        row = (
            team_summary[team_summary[Cols.TEAM_ID].astype(str) == str(team_id)]
            if not team_summary.empty and Cols.TEAM_ID in team_summary.columns
            else pd.DataFrame()
        )
        if row.empty:
            return None

        r = row.iloc[0]
        has_formation = bool(formations and team_id in formations)
        formation = formations[team_id].formation_str if has_formation else "unknown"
        form_conf = formations[team_id].confidence    if has_formation else 0.0

        weak_df = (
            weaknesses.get(team_id, pd.DataFrame()) if weaknesses else pd.DataFrame()
        )
        has_risk = not weak_df.empty and "risk_level" in weak_df.columns
        n_high = (
            int((weak_df["risk_level"] == "high").sum()) if has_risk else 0
        )
        n_critical = (
            int((weak_df["risk_level"] == "critical").sum()) if has_risk else 0
        )
        has_zone_x = not weak_df.empty and "zone_x" in weak_df.columns
        has_zone_y = not weak_df.empty and "zone_y" in weak_df.columns
        worst_zone_x = float(weak_df["zone_x"].iloc[0]) if has_zone_x else None
        worst_zone_y = float(weak_df["zone_y"].iloc[0]) if has_zone_y else None

        team_players = (
            player_summary[player_summary[Cols.TEAM_ID].astype(str) == str(team_id)]
            if not player_summary.empty and Cols.TEAM_ID in player_summary.columns
            else pd.DataFrame()
        )
        top_players = self._top_players_dicts(team_players, n=self.top_players_n)

        return TeamReport(
            team_id=str(team_id),
            team_name=str(r.get(Cols.TEAM_NAME, team_id)),
            action_count=int(r.get("action_count", 0)),
            vaep_total=float(r.get("vaep_total", 0.0)),
            vaep_offensive=float(r.get("vaep_offensive", 0.0)),
            vaep_defensive=float(r.get("vaep_defensive", 0.0)),
            xg_total=float(r.get("xg_total", 0.0)),
            formation=formation,
            formation_conf=round(form_conf, 3),
            n_high_risk_zones=n_high,
            n_critical_zones=n_critical,
            weakest_zone_x=worst_zone_x,
            weakest_zone_y=worst_zone_y,
            top_players=top_players,
        )

    def _top_actions(
        self,
        vaep_df: pd.DataFrame,
        n: int,
    ) -> list[ActionSummary]:
        """Return top-n actions by VAEP value as ActionSummary objects."""
        if vaep_df.empty or Cols.VAEP_VALUE not in vaep_df.columns:
            return []
        top = vaep_df.nlargest(n, Cols.VAEP_VALUE)
        result = []
        for _, row in top.iterrows():
            result.append(ActionSummary(
                action_id=str(row.get(Cols.ACTION_ID, "")),
                action_type=str(row.get(Cols.ACTION_TYPE, "")),
                player_name=str(row.get(Cols.PLAYER_NAME, "")),
                team_name=str(row.get(Cols.TEAM_NAME, "")),
                minute=int(row.get(Cols.MINUTE, 0)),
                vaep_value=round(float(row.get(Cols.VAEP_VALUE, 0.0)), 4),
                start_x=round(float(row.get(Cols.START_X, 0.0)), 1),
                start_y=round(float(row.get(Cols.START_Y, 0.0)), 1),
            ))
        return result

    def _top_players_dicts(
        self,
        player_summary: pd.DataFrame,
        n: int,
    ) -> list[dict[str, Any]]:
        if player_summary.empty:
            return []
        top = player_summary.head(n)
        cols = [
            Cols.PLAYER_ID, "player_name", Cols.TEAM_NAME,
            "vaep_total", "vaep_offensive", "vaep_defensive",
        ]
        cols = [c for c in cols if c in top.columns]
        return top[cols].to_dict(orient="records")

    @staticmethod
    def _team_name(vaep_df: pd.DataFrame, team_id: str) -> str:
        if Cols.TEAM_NAME in vaep_df.columns:
            rows = vaep_df[vaep_df[Cols.TEAM_ID].astype(str) == str(team_id)]
            if not rows.empty:
                return str(rows[Cols.TEAM_NAME].iloc[0])
        return str(team_id)

    @staticmethod
    def _team_xg(vaep_df: pd.DataFrame, team_id: str) -> float:
        if Cols.XG not in vaep_df.columns or Cols.TEAM_ID not in vaep_df.columns:
            return 0.0
        return float(
            vaep_df[vaep_df[Cols.TEAM_ID].astype(str) == str(team_id)][Cols.XG].sum()
        )

    @staticmethod
    def _team_vaep(team_summary: pd.DataFrame, team_id: str) -> float:
        if team_summary.empty or "vaep_total" not in team_summary.columns:
            return 0.0
        row = team_summary[team_summary[Cols.TEAM_ID].astype(str) == str(team_id)]
        return float(row["vaep_total"].iloc[0]) if not row.empty else 0.0

    @staticmethod
    def _rank_of(
        player_summary: pd.DataFrame,
        player_id: str,
        metric: str,
    ) -> int:
        if player_summary.empty or metric not in player_summary.columns:
            return -1
        sorted_df = player_summary.sort_values(metric, ascending=False).reset_index(drop=True)
        matches   = sorted_df[sorted_df[Cols.PLAYER_ID].astype(str) == str(player_id)]
        return int(matches.index[0] + 1) if not matches.empty else -1
