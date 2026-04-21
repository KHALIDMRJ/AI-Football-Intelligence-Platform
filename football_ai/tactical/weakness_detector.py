"""
Team weakness detector.

Identifies pitch zones where a team's defensive VAEP is negative —
meaning opponents consistently created value by moving the ball into
or through those zones against this team.

Algorithm
---------
For a given ``team_id``:
    1. Select all actions where the *opponent* was in possession
       (possession_team_id != team_id).
    2. Group those actions by ``zone_id`` (18 × 12 grid from Phase 3).
    3. Compute mean VAEP value per zone for the opponent's actions
       — high positive VAEP for the opponent = dangerous zone for the team.
    4. Flag zones above a risk threshold as weaknesses.
    5. Map zones back to (x, y) pitch coordinates for visualisation.

Risk levels
-----------
    critical  — top 10 % by opponent VAEP
    high      — 10–25 %
    medium    — 25–50 %
    low       — bottom 50 %

Output
------
pd.DataFrame with columns:
    zone_id, zone_x, zone_y, mean_opponent_vaep, action_count, risk_level
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_ai.config import settings
from football_ai.constants import Cols
from football_ai.logger import get_logger
from football_ai.utils import zone_to_coords

logger = get_logger(__name__)

_RISK_LEVELS = {
    "critical": 0.90,
    "high":     0.75,
    "medium":   0.50,
    "low":      0.0,
}


class WeaknessDetector:
    """
    Detects team tactical weaknesses from opponent VAEP distribution
    across pitch zones.

    Parameters
    ----------
    min_actions_per_zone : int
        Zones with fewer than this many actions are excluded (noise).
        Default 3.

    Usage
    -----
    >>> detector = WeaknessDetector()
    >>> weaknesses = detector.detect(vaep_df, team_id="2761")
    >>> critical = weaknesses[weaknesses["risk_level"] == "critical"]
    """

    def __init__(self, min_actions_per_zone: int = 3) -> None:
        self.min_actions_per_zone = min_actions_per_zone

    def detect(
        self,
        vaep_df: pd.DataFrame,
        team_id: str,
    ) -> pd.DataFrame:
        """
        Detect zones where opponents gain high VAEP against ``team_id``.

        Parameters
        ----------
        vaep_df : pd.DataFrame
            VAEP-scored action DataFrame with zone_id, possession_team_id,
            vaep_value, player_id columns.
        team_id : str
            The team whose weaknesses we are analysing.

        Returns
        -------
        pd.DataFrame
            One row per risky zone, sorted by mean_opponent_vaep descending.
            Columns: zone_id, zone_x, zone_y, mean_opponent_vaep,
                     action_count, risk_level.
        """
        required = [Cols.VAEP_VALUE, Cols.ZONE_ID]
        missing  = [c for c in required if c not in vaep_df.columns]
        if missing:
            raise ValueError(
                f"WeaknessDetector requires {required}. Missing: {missing}."
            )

        # ── Isolate opponent actions ───────────────────────────────────────────
        poss_col = (
            Cols.POSSESSION_TEAM_ID
            if Cols.POSSESSION_TEAM_ID in vaep_df.columns
            else Cols.TEAM_ID
        )
        opponent_mask = vaep_df[poss_col].astype(str) != str(team_id)
        opp_df = vaep_df.loc[opponent_mask].copy()

        if opp_df.empty:
            logger.warning(
                "No opponent actions found for team_id=%s. "
                "Check possession_team_id column.", team_id,
            )
            return pd.DataFrame(columns=[
                "zone_id", "zone_x", "zone_y",
                "mean_opponent_vaep", "action_count", "risk_level",
            ])

        # ── Aggregate VAEP by zone ────────────────────────────────────────────
        zone_stats = (
            opp_df.groupby(Cols.ZONE_ID)[Cols.VAEP_VALUE]
            .agg(mean_opponent_vaep="mean", action_count="count")
            .reset_index()
            .rename(columns={Cols.ZONE_ID: "zone_id"})
        )

        # ── Filter noise ──────────────────────────────────────────────────────
        zone_stats = zone_stats[
            zone_stats["action_count"] >= self.min_actions_per_zone
        ].copy()

        if zone_stats.empty:
            return zone_stats

        # ── Map zone IDs to pitch coordinates ─────────────────────────────────
        coords = zone_stats["zone_id"].apply(
            lambda z: zone_to_coords(
                z,
                settings.pitch.zones_x,
                settings.pitch.zones_y,
                settings.pitch.length,
                settings.pitch.width,
            )
        )
        zone_stats["zone_x"] = coords.apply(lambda c: round(c[0], 1))
        zone_stats["zone_y"] = coords.apply(lambda c: round(c[1], 1))

        # ── Assign risk level by percentile ───────────────────────────────────
        zone_stats["risk_level"] = self._assign_risk(
            zone_stats["mean_opponent_vaep"]
        )

        # ── Sort ──────────────────────────────────────────────────────────────
        zone_stats = (
            zone_stats
            .sort_values("mean_opponent_vaep", ascending=False)
            .reset_index(drop=True)
        )
        zone_stats["mean_opponent_vaep"] = zone_stats["mean_opponent_vaep"].round(5)

        critical_count = (zone_stats["risk_level"] == "critical").sum()
        high_count     = (zone_stats["risk_level"] == "high").sum()
        logger.info(
            "Weakness detection for team %s: %d zones analysed, "
            "%d critical, %d high-risk.",
            team_id, len(zone_stats), critical_count, high_count,
        )

        return zone_stats

    def detect_all_teams(self, vaep_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Run ``detect()`` for every team present in ``vaep_df``.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys are team_id strings.
        """
        teams = vaep_df[Cols.TEAM_ID].unique().tolist()
        return {str(tid): self.detect(vaep_df, str(tid)) for tid in teams}

    @staticmethod
    def _assign_risk(series: pd.Series) -> pd.Series:
        """Assign risk level based on within-series percentile rank."""
        pct = series.rank(pct=True, ascending=True)
        conditions = [pct >= 0.90, pct >= 0.75, pct >= 0.50]
        choices    = ["critical", "high", "medium"]
        return pd.Series(
            np.select(conditions, choices, default="low"),
            index=series.index,
        )
