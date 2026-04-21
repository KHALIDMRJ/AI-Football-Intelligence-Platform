"""
VAEP aggregation.

Aggregates per-action VAEP values into player-level and team-level summaries.

Player-level outputs
--------------------
    player_id         str
    player_name       str
    team_id           str
    team_name         str
    minutes_played    float   — derived from action timestamps
    action_count      int
    vaep_total        float   — Σ V(aᵢ)
    vaep_offensive    float   — Σ max(0, Δ_scores)
    vaep_defensive    float   — Σ max(0, −Δ_concedes)
    vaep_per_90       float   — vaep_total / minutes_played * 90
    xg_total          float   — Σ xg (shot events only)
    xt_total          float   — Σ xt_delta (pass/carry events)

Team-level outputs
------------------
    team_id           str
    team_name         str
    action_count      int
    vaep_total        float
    vaep_offensive    float
    vaep_defensive    float
    xg_total          float
    xt_total          float

Both DataFrames are sorted by vaep_total descending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_ai.constants import Cols
from football_ai.logger import get_logger

logger = get_logger(__name__)

_MIN_MINUTES_FOR_PER90 = 1.0    # avoid division by near-zero minutes
_SECONDS_PER_90 = 5_400.0       # 90 × 60


def _safe_minutes(df: pd.DataFrame, player_id: str) -> float:
    """
    Estimate minutes played for a player from their action timestamps.

    Uses (max_timestamp − min_timestamp) in seconds / 60.
    Falls back to 0 if timestamp column is absent.
    """
    if Cols.TIMESTAMP not in df.columns or Cols.PLAYER_ID not in df.columns:
        return 0.0
    player_rows = df[df[Cols.PLAYER_ID].astype(str) == player_id]
    if player_rows.empty:
        return 0.0
    ts = player_rows[Cols.TIMESTAMP].astype(float)
    return float((ts.max() - ts.min()) / 60.0)


class VAEPAggregator:
    """
    Aggregates per-action VAEP values to player and team level.

    Parameters
    ----------
    min_actions : int
        Players with fewer actions than this are excluded from rankings.
        Default 5 — avoids single-action outliers dominating.

    Usage
    -----
    >>> agg = VAEPAggregator()
    >>> player_df = agg.player_summary(vaep_df)
    >>> team_df   = agg.team_summary(vaep_df)
    """

    def __init__(self, min_actions: int = 5) -> None:
        self.min_actions = min_actions

    # ── Player summary ─────────────────────────────────────────────────────────

    def player_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate VAEP to a per-player summary DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Action-level DataFrame with vaep_value, vaep_offensive,
            vaep_defensive, xg, xt_delta columns (output of the VAEP pipeline).

        Returns
        -------
        pd.DataFrame
            One row per player, sorted by vaep_total descending.
            Columns: player_id, player_name, team_id, team_name,
                     action_count, minutes_played,
                     vaep_total, vaep_offensive, vaep_defensive, vaep_per_90,
                     xg_total, xt_total.
        """
        self._validate(df)

        if df.empty:
            return pd.DataFrame()

        # ── Build aggregation dict ─────────────────────────────────────────────
        agg_cols: dict[str, object] = {
            Cols.PLAYER_NAME:   (Cols.PLAYER_NAME, "first"),
            Cols.TEAM_ID:       (Cols.TEAM_ID, "first"),
            Cols.TEAM_NAME:     (Cols.TEAM_NAME, "first"),
            "action_count":     (Cols.VAEP_VALUE, "count"),
            "vaep_total":       (Cols.VAEP_VALUE, "sum"),
            "vaep_offensive":   (Cols.VAEP_OFFENSIVE, "sum"),
            "vaep_defensive":   (Cols.VAEP_DEFENSIVE, "sum"),
        }

        # Optional: xg and xt_delta if present
        if Cols.XG in df.columns:
            agg_cols["xg_total"] = (Cols.XG, "sum")
        if Cols.XT_DELTA in df.columns:
            agg_cols["xt_total"] = (Cols.XT_DELTA, "sum")

        player_df = (
            df.groupby(Cols.PLAYER_ID, as_index=True)
            .agg(**agg_cols)
            .reset_index()
        )

        # ── Minutes played ────────────────────────────────────────────────────
        player_df["minutes_played"] = player_df[Cols.PLAYER_ID].apply(
            lambda pid: _safe_minutes(df, pid)
        )

        # ── VAEP / 90 ─────────────────────────────────────────────────────────
        player_df["vaep_per_90"] = np.where(
            player_df["minutes_played"] >= _MIN_MINUTES_FOR_PER90,
            player_df["vaep_total"] / player_df["minutes_played"] * 90.0,
            np.nan,
        )

        # ── Filter minimum actions ────────────────────────────────────────────
        before = len(player_df)
        player_df = player_df[
            player_df["action_count"] >= self.min_actions
        ].copy()
        filtered = before - len(player_df)
        if filtered:
            logger.debug(
                "Filtered %d players with fewer than %d actions.",
                filtered, self.min_actions,
            )

        # ── Sort ──────────────────────────────────────────────────────────────
        player_df = player_df.sort_values("vaep_total", ascending=False)
        player_df = player_df.reset_index(drop=True)
        player_df.index += 1   # 1-based rank

        # ── Round floats ──────────────────────────────────────────────────────
        float_cols = [
            "vaep_total", "vaep_offensive", "vaep_defensive",
            "vaep_per_90", "minutes_played",
        ]
        if "xg_total" in player_df.columns:
            float_cols.append("xg_total")
        if "xt_total" in player_df.columns:
            float_cols.append("xt_total")
        player_df[float_cols] = player_df[float_cols].round(4)

        logger.info(
            "Player summary: %d players.  "
            "Top VAEP: %s (%.4f)",
            len(player_df),
            player_df[Cols.PLAYER_NAME].iloc[0] if not player_df.empty else "—",
            player_df["vaep_total"].iloc[0] if not player_df.empty else 0.0,
        )

        return player_df

    # ── Team summary ───────────────────────────────────────────────────────────

    def team_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate VAEP to a per-team summary DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Same action-level DataFrame used in ``player_summary()``.

        Returns
        -------
        pd.DataFrame
            One row per team, sorted by vaep_total descending.
            Columns: team_id, team_name, action_count,
                     vaep_total, vaep_offensive, vaep_defensive,
                     xg_total, xt_total.
        """
        self._validate(df)

        if df.empty or Cols.TEAM_ID not in df.columns:
            return pd.DataFrame()

        agg_cols: dict[str, object] = {
            Cols.TEAM_NAME:   (Cols.TEAM_NAME, "first"),
            "action_count":   (Cols.VAEP_VALUE, "count"),
            "vaep_total":     (Cols.VAEP_VALUE, "sum"),
            "vaep_offensive": (Cols.VAEP_OFFENSIVE, "sum"),
            "vaep_defensive": (Cols.VAEP_DEFENSIVE, "sum"),
        }
        if Cols.XG in df.columns:
            agg_cols["xg_total"] = (Cols.XG, "sum")
        if Cols.XT_DELTA in df.columns:
            agg_cols["xt_total"] = (Cols.XT_DELTA, "sum")

        team_df = (
            df.groupby(Cols.TEAM_ID, as_index=True)
            .agg(**agg_cols)
            .reset_index()
            .sort_values("vaep_total", ascending=False)
            .reset_index(drop=True)
        )

        float_cols = ["vaep_total", "vaep_offensive", "vaep_defensive"]
        if "xg_total" in team_df.columns:
            float_cols.append("xg_total")
        if "xt_total" in team_df.columns:
            float_cols.append("xt_total")
        team_df[float_cols] = team_df[float_cols].round(4)

        logger.info(
            "Team summary: %d teams.  Top VAEP: %s (%.4f)",
            len(team_df),
            team_df[Cols.TEAM_NAME].iloc[0] if not team_df.empty else "—",
            team_df["vaep_total"].iloc[0] if not team_df.empty else 0.0,
        )

        return team_df

    # ── Top-N helpers ──────────────────────────────────────────────────────────

    def top_players(
        self,
        df: pd.DataFrame,
        n: int = 10,
        metric: str = "vaep_total",
    ) -> pd.DataFrame:
        """
        Return the top-n players by the given metric.

        Parameters
        ----------
        df : pd.DataFrame
            Action-level DataFrame (not already aggregated).
        n : int
            Number of players to return.
        metric : str
            Column to rank by. One of:
            ``vaep_total``, ``vaep_per_90``, ``vaep_offensive``,
            ``vaep_defensive``, ``xg_total``, ``xt_total``.

        Returns
        -------
        pd.DataFrame
            Top-n rows from ``player_summary()``, sorted by ``metric``.
        """
        summary = self.player_summary(df)
        if summary.empty or metric not in summary.columns:
            return summary
        return (
            summary
            .sort_values(metric, ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        required = [Cols.PLAYER_ID, Cols.VAEP_VALUE, Cols.VAEP_OFFENSIVE,
                    Cols.VAEP_DEFENSIVE]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"VAEPAggregator requires columns {required}. "
                f"Missing: {missing}. "
                f"Run ActionValueComputer.compute() first."
            )
