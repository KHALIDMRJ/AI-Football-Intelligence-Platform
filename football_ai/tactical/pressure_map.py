"""
Pressure map.

Identifies pitch zones with high opponent pressure, high negative VAEP
(dangerous for the defending team), and high action density.

Three maps are built per team:

pressure_intensity
    Proportion of actions that occurred under pressure per zone.
    High values = zones where the team is consistently pressed.

opponent_threat
    Mean VAEP of opponent actions per zone (mirrors WeaknessDetector
    but returns a grid/heatmap rather than a filtered table).

action_density
    Raw action count per zone, normalised to [0, 1].
    Shows where the game was played most intensely.

Output
------
PressureMap dataclass:
    team_id            str
    team_name          str
    pressure_intensity pd.DataFrame  (zone_id, zone_x, zone_y, intensity)
    opponent_threat    pd.DataFrame  (zone_id, zone_x, zone_y, threat)
    action_density     pd.DataFrame  (zone_id, zone_x, zone_y, density)
    high_risk_zones    pd.DataFrame  zones where all three metrics are elevated
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from football_ai.config import settings
from football_ai.constants import Cols
from football_ai.logger import get_logger
from football_ai.utils import zone_to_coords

logger = get_logger(__name__)


@dataclass
class PressureMapResult:
    """Container for pressure map outputs for a single team."""

    team_id:             str
    team_name:           str
    pressure_intensity:  pd.DataFrame = field(default_factory=pd.DataFrame)
    opponent_threat:     pd.DataFrame = field(default_factory=pd.DataFrame)
    action_density:      pd.DataFrame = field(default_factory=pd.DataFrame)
    high_risk_zones:     pd.DataFrame = field(default_factory=pd.DataFrame)


class PressureMap:
    """
    Builds three zone-level pressure maps for a team.

    Parameters
    ----------
    min_zone_actions : int
        Minimum actions per zone to include in maps. Default 2.
    high_risk_threshold : float
        Percentile (0–1) above which a zone is flagged high-risk
        on ALL three maps simultaneously. Default 0.75.

    Usage
    -----
    >>> pm = PressureMap()
    >>> result = pm.build(vaep_df, team_id="2761")
    >>> result.high_risk_zones
    """

    def __init__(
        self,
        min_zone_actions: int = 2,
        high_risk_threshold: float = 0.75,
    ) -> None:
        self.min_zone_actions    = min_zone_actions
        self.high_risk_threshold = high_risk_threshold

    def build(
        self,
        vaep_df: pd.DataFrame,
        team_id: str,
    ) -> PressureMapResult:
        """
        Build all three pressure maps for ``team_id``.

        Parameters
        ----------
        vaep_df : pd.DataFrame
            VAEP-scored action DataFrame.
        team_id : str

        Returns
        -------
        PressureMapResult
        """
        team_str = str(team_id)

        team_name = self._team_name(vaep_df, team_str)

        # Own actions (the team's own ball-carrying actions)
        own_mask = vaep_df[Cols.TEAM_ID].astype(str) == team_str
        own_df   = vaep_df.loc[own_mask].copy()

        # Opponent actions (threat to this team)
        poss_col = (
            Cols.POSSESSION_TEAM_ID
            if Cols.POSSESSION_TEAM_ID in vaep_df.columns
            else Cols.TEAM_ID
        )
        opp_mask = vaep_df[poss_col].astype(str) != team_str
        opp_df   = vaep_df.loc[opp_mask].copy()

        pressure_df   = self._pressure_intensity(own_df)
        threat_df     = self._opponent_threat(opp_df)
        density_df    = self._action_density(own_df)
        high_risk_df  = self._high_risk_zones(pressure_df, threat_df, density_df)

        logger.info(
            "Pressure map [%s]: %d high-risk zones.",
            team_name, len(high_risk_df),
        )

        return PressureMapResult(
            team_id=team_str,
            team_name=team_name,
            pressure_intensity=pressure_df,
            opponent_threat=threat_df,
            action_density=density_df,
            high_risk_zones=high_risk_df,
        )

    def build_all_teams(
        self, vaep_df: pd.DataFrame
    ) -> dict[str, PressureMapResult]:
        """Run ``build()`` for every team in ``vaep_df``."""
        teams = vaep_df[Cols.TEAM_ID].unique().tolist()
        return {str(tid): self.build(vaep_df, str(tid)) for tid in teams}

    # ── Map builders ──────────────────────────────────────────────────────────

    def _pressure_intensity(self, own_df: pd.DataFrame) -> pd.DataFrame:
        """Proportion of own actions that occurred under pressure, per zone."""
        if own_df.empty or Cols.ZONE_ID not in own_df.columns:
            return pd.DataFrame(columns=["zone_id", "zone_x", "zone_y", "intensity"])

        has_pressure = Cols.UNDER_PRESSURE in own_df.columns
        if has_pressure:
            zone_agg = (
                own_df.groupby(Cols.ZONE_ID)
                .agg(
                    total=(Cols.ZONE_ID, "count"),
                    pressed=(Cols.UNDER_PRESSURE, "sum"),
                )
                .reset_index()
                .rename(columns={Cols.ZONE_ID: "zone_id"})
            )
            zone_agg["intensity"] = (
                zone_agg["pressed"] / zone_agg["total"].clip(lower=1)
            ).round(4)
        else:
            zone_agg = (
                own_df.groupby(Cols.ZONE_ID)
                .size()
                .reset_index(name="total")
                .rename(columns={Cols.ZONE_ID: "zone_id"})
            )
            zone_agg["intensity"] = 0.0

        zone_agg = zone_agg[zone_agg["total"] >= self.min_zone_actions].copy()
        return self._add_coords(zone_agg[["zone_id", "intensity"]])

    def _opponent_threat(self, opp_df: pd.DataFrame) -> pd.DataFrame:
        """Mean opponent VAEP per zone — raw threat score."""
        if opp_df.empty or Cols.ZONE_ID not in opp_df.columns:
            return pd.DataFrame(columns=["zone_id", "zone_x", "zone_y", "threat"])

        zone_agg = (
            opp_df.groupby(Cols.ZONE_ID)[Cols.VAEP_VALUE]
            .agg(threat="mean", count="count")
            .reset_index()
            .rename(columns={Cols.ZONE_ID: "zone_id"})
        )
        zone_agg = zone_agg[zone_agg["count"] >= self.min_zone_actions].copy()
        zone_agg["threat"] = zone_agg["threat"].clip(lower=0.0).round(5)
        return self._add_coords(zone_agg[["zone_id", "threat"]])

    def _action_density(self, own_df: pd.DataFrame) -> pd.DataFrame:
        """Normalised action count per zone in [0, 1]."""
        if own_df.empty or Cols.ZONE_ID not in own_df.columns:
            return pd.DataFrame(columns=["zone_id", "zone_x", "zone_y", "density"])

        zone_counts = (
            own_df.groupby(Cols.ZONE_ID)
            .size()
            .reset_index(name="count")
            .rename(columns={Cols.ZONE_ID: "zone_id"})
        )
        zone_counts = zone_counts[zone_counts["count"] >= self.min_zone_actions].copy()
        max_count = zone_counts["count"].max()
        zone_counts["density"] = (
            (zone_counts["count"] / max(max_count, 1)).round(4)
        )
        return self._add_coords(zone_counts[["zone_id", "density"]])

    def _high_risk_zones(
        self,
        pressure_df: pd.DataFrame,
        threat_df: pd.DataFrame,
        density_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Zones that appear in the top quartile of ALL three maps."""
        if pressure_df.empty or threat_df.empty or density_df.empty:
            return pd.DataFrame(columns=["zone_id", "zone_x", "zone_y"])

        thr = self.high_risk_threshold

        high_pressure = set(
            pressure_df[
                pressure_df["intensity"] >= pressure_df["intensity"].quantile(thr)
            ]["zone_id"]
        )
        high_threat = set(
            threat_df[
                threat_df["threat"] >= threat_df["threat"].quantile(thr)
            ]["zone_id"]
        )
        high_density = set(
            density_df[
                density_df["density"] >= density_df["density"].quantile(thr)
            ]["zone_id"]
        )

        risk_zones = high_pressure & high_threat & high_density

        if not risk_zones:
            # Fall back: zones in top quartile of threat alone
            risk_zones = high_threat

        df = threat_df[threat_df["zone_id"].isin(risk_zones)].copy()
        df = df.sort_values("threat", ascending=False).reset_index(drop=True)
        return df

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _add_coords(df: pd.DataFrame) -> pd.DataFrame:
        """Append zone_x and zone_y columns to a zone-level DataFrame."""
        if df.empty:
            df["zone_x"] = pd.Series(dtype=float)
            df["zone_y"] = pd.Series(dtype=float)
            return df
        coords = df["zone_id"].apply(
            lambda z: zone_to_coords(
                int(z),
                settings.pitch.zones_x,
                settings.pitch.zones_y,
                settings.pitch.length,
                settings.pitch.width,
            )
        )
        df = df.copy()
        df["zone_x"] = coords.apply(lambda c: round(c[0], 1))
        df["zone_y"] = coords.apply(lambda c: round(c[1], 1))
        return df[["zone_id", "zone_x", "zone_y"] + [
            c for c in df.columns if c not in ("zone_id", "zone_x", "zone_y")
        ]]

    @staticmethod
    def _team_name(vaep_df: pd.DataFrame, team_id: str) -> str:
        if Cols.TEAM_NAME in vaep_df.columns:
            row = vaep_df[vaep_df[Cols.TEAM_ID].astype(str) == team_id]
            if not row.empty:
                return str(row[Cols.TEAM_NAME].iloc[0])
        return team_id
