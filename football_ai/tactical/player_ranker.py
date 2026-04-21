"""
Player ranker.

Produces ranked player tables from a VAEP-scored action DataFrame.
Extends the VAEPAggregator summaries with per-role breakdowns and
normalized per-90 rankings for fair cross-player comparison.

Ranking tiers
-------------
    overall      — by vaep_total
    offensive    — by vaep_offensive
    defensive    — by vaep_defensive
    per_90       — by vaep_per_90 (requires ≥ min_minutes)
    efficiency   — vaep_total / action_count (quality per touch)

Output
------
Each method returns a pd.DataFrame with a 1-based rank index and
the same schema as VAEPAggregator.player_summary(), extended with:
    rank         int    — 1-based position in this ranking
    tier         str    — "elite" / "strong" / "average" / "below_average"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_ai.logger import get_logger
from football_ai.vaep.aggregator import VAEPAggregator

logger = get_logger(__name__)

# Thresholds for tier labels (percentile-based at runtime)
_TIER_LABELS = ["elite", "strong", "average", "below_average"]


def _assign_tier(series: pd.Series) -> pd.Series:
    """
    Assign a tier label based on percentile rank within the series.

    Top 10 % → elite, 10-40 % → strong,
    40-75 % → average, bottom 25 % → below_average.
    """
    pct = series.rank(pct=True, ascending=True)
    conditions = [
        pct >= 0.90,
        pct >= 0.60,
        pct >= 0.25,
    ]
    choices = ["elite", "strong", "average"]
    return pd.Series(
        np.select(conditions, choices, default="below_average"),
        index=series.index,
    )


class PlayerRanker:
    """
    Ranks players across multiple metrics using VAEP-scored action data.

    Parameters
    ----------
    min_actions : int
        Minimum number of on-ball actions required for inclusion.
        Default 5 — filters single-appearance substitutes.
    min_minutes : float
        Minimum minutes played for inclusion in per-90 rankings.
        Default 10.0.

    Usage
    -----
    >>> ranker = PlayerRanker()
    >>> table = ranker.rank_overall(vaep_df)
    >>> top10 = ranker.top_n(vaep_df, n=10, metric="vaep_total")
    """

    def __init__(
        self,
        min_actions: int = 5,
        min_minutes: float = 10.0,
    ) -> None:
        self.min_actions = min_actions
        self.min_minutes = min_minutes
        self._agg = VAEPAggregator(min_actions=min_actions)

    # ── Public ranking methods ─────────────────────────────────────────────────

    def rank_overall(self, vaep_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank all eligible players by total VAEP.

        Returns
        -------
        pd.DataFrame
            Columns: rank, player_id, player_name, team_name,
                     action_count, vaep_total, vaep_offensive,
                     vaep_defensive, vaep_per_90, tier.
        """
        return self._build_ranking(vaep_df, sort_col="vaep_total")

    def rank_offensive(self, vaep_df: pd.DataFrame) -> pd.DataFrame:
        """Rank players by offensive VAEP (positive scoring contributions)."""
        return self._build_ranking(vaep_df, sort_col="vaep_offensive")

    def rank_defensive(self, vaep_df: pd.DataFrame) -> pd.DataFrame:
        """Rank players by defensive VAEP (concede-risk reduction)."""
        return self._build_ranking(vaep_df, sort_col="vaep_defensive")

    def rank_per_90(self, vaep_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank players by VAEP per 90 minutes.

        Players with fewer than ``min_minutes`` played are excluded.
        """
        df = self._build_ranking(vaep_df, sort_col="vaep_per_90")
        return df[df["minutes_played"] >= self.min_minutes].copy()

    def rank_efficiency(self, vaep_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank players by VAEP per action — measures quality per touch.
        Rewards players who create value with fewer touches (e.g. finishers).
        """
        summary = self._agg.player_summary(vaep_df)
        if summary.empty:
            return summary
        summary["vaep_per_action"] = (
            summary["vaep_total"] / summary["action_count"].clip(lower=1)
        ).round(5)
        return self._finalize(summary, sort_col="vaep_per_action")

    def top_n(
        self,
        vaep_df: pd.DataFrame,
        n: int = 10,
        metric: str = "vaep_total",
    ) -> pd.DataFrame:
        """
        Return the top-n players by the chosen metric.

        Parameters
        ----------
        vaep_df : pd.DataFrame
            VAEP-scored action DataFrame.
        n : int
        metric : str
            One of "vaep_total", "vaep_offensive", "vaep_defensive",
            "vaep_per_90", "efficiency".

        Returns
        -------
        pd.DataFrame
            Top-n rows from the appropriate ranking method.
        """
        dispatch = {
            "vaep_total":     self.rank_overall,
            "vaep_offensive": self.rank_offensive,
            "vaep_defensive": self.rank_defensive,
            "vaep_per_90":    self.rank_per_90,
            "efficiency":     self.rank_efficiency,
        }
        if metric not in dispatch:
            raise ValueError(
                f"Unknown metric '{metric}'. "
                f"Choose from {sorted(dispatch)}."
            )
        ranked = dispatch[metric](vaep_df)
        return ranked.head(n).reset_index(drop=True)

    def all_rankings(self, vaep_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Compute all ranking tables in one call.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys: "overall", "offensive", "defensive", "per_90", "efficiency".
        """
        return {
            "overall":    self.rank_overall(vaep_df),
            "offensive":  self.rank_offensive(vaep_df),
            "defensive":  self.rank_defensive(vaep_df),
            "per_90":     self.rank_per_90(vaep_df),
            "efficiency": self.rank_efficiency(vaep_df),
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_ranking(self, vaep_df: pd.DataFrame, sort_col: str) -> pd.DataFrame:
        summary = self._agg.player_summary(vaep_df)
        if summary.empty:
            return summary
        return self._finalize(summary, sort_col=sort_col)

    @staticmethod
    def _finalize(summary: pd.DataFrame, sort_col: str) -> pd.DataFrame:
        """Sort, assign tier, reset rank index."""
        if sort_col not in summary.columns:
            return summary
        df = summary.sort_values(sort_col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1        # 1-based rank
        df.index.name = "rank"
        df["tier"] = _assign_tier(df[sort_col])
        return df
