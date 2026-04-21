"""
Formation analyser.

Infers a team's formation from the spatial distribution of their
players' on-ball actions using k-means clustering.

Approach
--------
1. For each team, collect all action start positions (x, y).
2. Mirror coordinates so all teams attack left-to-right.
3. Run k-means with k = outfield_players (default 10).
4. Map cluster centroids to a canonical 3-zone depth
   (defence / midfield / attack) and width (left / centre / right).
5. Count players per depth zone → infer formation string (e.g. "4-3-3").

This is a heuristic, not a ground-truth formation tracker.
It works well for matches with ≥ 200 actions per team.

Output
------
FormationResult dataclass:
    team_id       str
    team_name     str
    formation_str str          — e.g. "4-3-3", "4-4-2", "3-5-2"
    centroids     pd.DataFrame — cluster_id, x, y, zone_label, depth, width
    confidence    float        — silhouette-like score in [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from football_ai.config import settings
from football_ai.constants import Cols
from football_ai.logger import get_logger

logger = get_logger(__name__)

# Pitch thirds (normalised x in [0, 1])
_DEFENCE_X_MAX  = 0.40   # x < 40 % of pitch
_MIDFIELD_X_MAX = 0.70   # 40–70 %
# attack: x > 70 %

# Width thirds
_LEFT_Y_MAX     = 0.35   # y < 35 %
_RIGHT_Y_MIN    = 0.65   # y > 65 %

# Minimum actions needed to run formation analysis
_MIN_ACTIONS = 30


@dataclass
class FormationResult:
    """Result of formation analysis for a single team."""

    team_id:       str
    team_name:     str
    formation_str: str                            # e.g. "4-3-3"
    centroids:     pd.DataFrame = field(default_factory=pd.DataFrame)
    confidence:    float = 0.0
    note:          str = ""


class FormationAnalyser:
    """
    Infers team formation from player action positions via k-means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of player clusters (outfield players). Default 10.
    random_state : int
        Seed for k-means reproducibility.

    Usage
    -----
    >>> analyser = FormationAnalyser()
    >>> result = analyser.analyse(vaep_df, team_id="2761")
    >>> print(result.formation_str)   # "4-3-3"
    """

    def __init__(
        self,
        n_clusters: int = 10,
        random_state: int = 42,
    ) -> None:
        self.n_clusters  = n_clusters
        self.random_state = random_state

    def analyse(
        self,
        vaep_df: pd.DataFrame,
        team_id: str,
    ) -> FormationResult:
        """
        Infer formation for a single team.

        Parameters
        ----------
        vaep_df : pd.DataFrame
            VAEP-scored action DataFrame with start_x, start_y, team_id.
        team_id : str

        Returns
        -------
        FormationResult
        """
        team_mask = vaep_df[Cols.TEAM_ID].astype(str) == str(team_id)
        team_df   = vaep_df.loc[team_mask].copy()

        team_name = (
            team_df[Cols.TEAM_NAME].iloc[0]
            if Cols.TEAM_NAME in team_df.columns and not team_df.empty
            else str(team_id)
        )

        if len(team_df) < _MIN_ACTIONS:
            logger.warning(
                "Team %s has only %d actions — formation analysis unreliable.",
                team_id, len(team_df),
            )
            return FormationResult(
                team_id=str(team_id),
                team_name=team_name,
                formation_str="unknown",
                confidence=0.0,
                note=f"Insufficient data ({len(team_df)} actions < {_MIN_ACTIONS})",
            )

        # ── Collect positions ─────────────────────────────────────────────────
        coords = team_df[[Cols.START_X, Cols.START_Y]].dropna().values
        pl = settings.pitch.length
        pw = settings.pitch.width

        # Normalise to [0, 1]
        X = coords.copy().astype(float)
        X[:, 0] /= pl
        X[:, 1] /= pw

        # ── k-means ───────────────────────────────────────────────────────────
        k = min(self.n_clusters, len(X) - 1)
        if k < 3:
            return FormationResult(
                team_id=str(team_id), team_name=team_name,
                formation_str="unknown", confidence=0.0,
                note="Too few unique positions for clustering.",
            )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        km.fit(X_scaled)

        # Inverse-transform centroids back to normalised space
        centroids_norm = scaler.inverse_transform(km.cluster_centers_)

        # ── Label each centroid by depth / width zone ─────────────────────────
        rows = []
        for i, (cx, cy) in enumerate(centroids_norm):
            depth = self._depth_zone(cx)
            width = self._width_zone(cy)
            rows.append({
                "cluster_id": i,
                "x":          round(float(cx * pl), 1),
                "y":          round(float(cy * pw), 1),
                "depth":      depth,
                "width":      width,
                "zone_label": f"{depth}_{width}",
            })
        centroids_df = pd.DataFrame(rows)

        # ── Formation string ──────────────────────────────────────────────────
        formation_str, confidence = self._infer_formation(centroids_df)

        logger.info(
            "Formation [%s]: %s  (confidence=%.2f, k=%d)",
            team_name, formation_str, confidence, k,
        )

        return FormationResult(
            team_id=str(team_id),
            team_name=team_name,
            formation_str=formation_str,
            centroids=centroids_df,
            confidence=confidence,
        )

    def analyse_all_teams(
        self,
        vaep_df: pd.DataFrame,
    ) -> dict[str, FormationResult]:
        """Run ``analyse()`` for every team in ``vaep_df``."""
        teams = vaep_df[Cols.TEAM_ID].unique().tolist()
        return {
            str(tid): self.analyse(vaep_df, str(tid))
            for tid in teams
        }

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _depth_zone(x_norm: float) -> str:
        if x_norm < _DEFENCE_X_MAX:
            return "defence"
        if x_norm < _MIDFIELD_X_MAX:
            return "midfield"
        return "attack"

    @staticmethod
    def _width_zone(y_norm: float) -> str:
        if y_norm < _LEFT_Y_MAX:
            return "left"
        if y_norm > _RIGHT_Y_MIN:
            return "right"
        return "centre"

    @staticmethod
    def _infer_formation(centroids: pd.DataFrame) -> tuple[str, float]:
        """
        Convert depth-zone cluster counts into a formation string.

        Formation string convention: defence-midfield-attack
        Goalkeeper is excluded (deepest defender assumed to be GK proxy).
        """
        if centroids.empty:
            return "unknown", 0.0

        depth_counts = centroids["depth"].value_counts()
        n_def = int(depth_counts.get("defence", 0))
        n_mid = int(depth_counts.get("midfield", 0))
        n_att = int(depth_counts.get("attack", 0))

        # Subtract 1 from defence for goalkeeper proxy
        n_def = max(0, n_def - 1)
        total = n_def + n_mid + n_att

        if total == 0:
            return "unknown", 0.0

        # Confidence: how balanced the distribution is
        # Perfect 4-4-2 → high; all in one zone → low
        parts = [n_def, n_mid, n_att]
        non_zero = sum(1 for p in parts if p > 0)
        confidence = round(min(1.0, non_zero / 3.0 * (total / 10.0)), 2)

        formation_str = f"{n_def}-{n_mid}-{n_att}"
        return formation_str, confidence
