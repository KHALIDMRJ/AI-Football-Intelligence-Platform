"""
Player service.

Business logic for player-related endpoints.
Reads player_summary Parquet files and VAEP action files to build
player profiles and ranked lists.
"""

from __future__ import annotations

import pandas as pd

from football_ai.api.schemas.responses import ActionItem, PlayerDetailResponse, PlayerItem
from football_ai.constants import Cols
from football_ai.ingestion.storage.parquet_store import ParquetStore
from football_ai.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MATCH_LIMIT = 200   # guard against returning enormous lists


class PlayerService:
    """
    Provides player-level data from processed Parquet artefacts.

    Parameters
    ----------
    store : ParquetStore, optional
    """

    def __init__(self, store: ParquetStore | None = None) -> None:
        self.store = store or ParquetStore()

    # ── List all players ───────────────────────────────────────────────────────

    def get_players(
        self,
        match_id: str | None = None,
        team_id: str | None = None,
        limit: int = 50,
    ) -> tuple[list[PlayerItem], str | None]:
        """
        Return a list of players from one match or across all matches.

        Parameters
        ----------
        match_id : str, optional
            If given, restrict to that match.
        team_id : str, optional
            If given, filter to that team.
        limit : int
            Maximum number of players returned.

        Returns
        -------
        (list[PlayerItem], match_id_used)
        """
        summary, used_match_id = self._load_player_summary(match_id)

        if summary.empty:
            return [], used_match_id

        if team_id:
            summary = summary[
                summary[Cols.TEAM_ID].astype(str) == str(team_id)
            ]

        # Sort by vaep_total descending by default
        if "vaep_total" in summary.columns:
            summary = summary.sort_values("vaep_total", ascending=False)

        summary = summary.head(limit)
        return [self._row_to_item(row) for _, row in summary.iterrows()], used_match_id

    # ── Player rankings ────────────────────────────────────────────────────────

    def get_rankings(
        self,
        match_id: str | None = None,
        metric: str = "vaep_total",
        limit: int = 20,
    ) -> tuple[list[PlayerItem], str | None]:
        """
        Return player rankings by a given metric.

        Parameters
        ----------
        metric : str
            Column to sort by. One of vaep_total, vaep_offensive,
            vaep_defensive, vaep_per_90, xg_total, xt_total.
        """
        valid_metrics = {
            "vaep_total", "vaep_offensive", "vaep_defensive",
            "vaep_per_90", "xg_total", "xt_total",
        }
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric '{metric}'. "
                f"Choose from {sorted(valid_metrics)}."
            )

        summary, used_match_id = self._load_player_summary(match_id)
        if summary.empty or metric not in summary.columns:
            return [], used_match_id

        ranked = (
            summary
            .dropna(subset=[metric])
            .sort_values(metric, ascending=False)
            .head(limit)
        )
        return [self._row_to_item(row) for _, row in ranked.iterrows()], used_match_id

    # ── Player detail ──────────────────────────────────────────────────────────

    def get_player(
        self,
        player_id: str,
        match_id: str | None = None,
    ) -> PlayerDetailResponse:
        """
        Return a full player profile.

        Raises
        ------
        KeyError
            If the player is not found.
        """
        summary, used_match_id = self._load_player_summary(match_id)

        if summary.empty:
            raise KeyError(f"Player {player_id!r} not found.")

        row_df = summary[
            summary[Cols.PLAYER_ID].astype(str) == str(player_id)
        ]
        if row_df.empty:
            raise KeyError(
                f"Player {player_id!r} not found in "
                f"match {used_match_id or 'any match'}."
            )

        row = row_df.iloc[0]
        top_actions = self._player_top_actions(player_id, used_match_id)

        return PlayerDetailResponse(
            player_id=str(player_id),
            player_name=str(row.get("player_name", player_id)),
            team_id=str(row.get(Cols.TEAM_ID, "")),
            team_name=str(row.get(Cols.TEAM_NAME, "")),
            action_count=int(row.get("action_count", 0)),
            vaep_total=float(row.get("vaep_total", 0.0)),
            vaep_offensive=float(row.get("vaep_offensive", 0.0)),
            vaep_defensive=float(row.get("vaep_defensive", 0.0)),
            vaep_per_90=(
                float(row["vaep_per_90"])
                if pd.notna(row.get("vaep_per_90"))
                else None
            ),
            xg_total=float(row.get("xg_total", 0.0)),
            xt_total=float(row.get("xt_total", 0.0)),
            top_actions=top_actions,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_player_summary(
        self,
        match_id: str | None,
    ) -> tuple[pd.DataFrame, str | None]:
        """
        Load player_summary for a specific match or the first available match.
        Returns (dataframe, match_id_actually_used).
        """
        if match_id:
            path = (
                self.store.processed_dir
                / f"match_{match_id}_player_summary.parquet"
            )
            if not path.exists():
                raise KeyError(
                    f"Player summary not found for match {match_id!r}. "
                    "Run the pipeline first."
                )
            return pd.read_parquet(path), match_id

        # No specific match — use first available
        files = sorted(
            self.store.processed_dir.glob("match_*_player_summary.parquet")
        )
        if not files:
            return pd.DataFrame(), None

        dfs = [pd.read_parquet(f) for f in files[:_DEFAULT_MATCH_LIMIT]]
        combined = pd.concat(dfs, ignore_index=True)
        return combined, None

    def _player_top_actions(
        self,
        player_id: str,
        match_id: str | None,
        n: int = 5,
    ) -> list[ActionItem]:
        """Load the top-n actions by VAEP for this player."""
        try:
            if match_id:
                vaep_path = (
                    self.store.processed_dir
                    / f"match_{match_id}_vaep.parquet"
                )
                if not vaep_path.exists():
                    return []
                vaep_df = pd.read_parquet(vaep_path)
            else:
                files = sorted(
                    self.store.processed_dir.glob("match_*_vaep.parquet")
                )
                if not files:
                    return []
                vaep_df = pd.read_parquet(files[0])

            player_df = vaep_df[
                vaep_df[Cols.PLAYER_ID].astype(str) == str(player_id)
            ]
            if player_df.empty or Cols.VAEP_VALUE not in player_df.columns:
                return []

            top = player_df.nlargest(n, Cols.VAEP_VALUE)
            return [
                ActionItem(
                    action_id=str(r.get(Cols.ACTION_ID, "")),
                    action_type=str(r.get(Cols.ACTION_TYPE, "")),
                    player_name=str(r.get(Cols.PLAYER_NAME, "")),
                    team_name=str(r.get(Cols.TEAM_NAME, "")),
                    minute=int(r.get(Cols.MINUTE, 0)),
                    vaep_value=round(float(r.get(Cols.VAEP_VALUE, 0.0)), 4),
                    start_x=round(float(r.get(Cols.START_X, 0.0)), 1),
                    start_y=round(float(r.get(Cols.START_Y, 0.0)), 1),
                )
                for _, r in top.iterrows()
            ]
        except Exception as exc:
            logger.warning("Could not load top actions for player %s: %s", player_id, exc)
            return []

    @staticmethod
    def _row_to_item(row: pd.Series) -> PlayerItem:
        return PlayerItem(
            player_id=str(row.get(Cols.PLAYER_ID, "")),
            player_name=str(row.get("player_name", "")),
            team_id=str(row.get(Cols.TEAM_ID, "")),
            team_name=str(row.get(Cols.TEAM_NAME, "")),
            action_count=int(row.get("action_count", 0)),
            vaep_total=float(row.get("vaep_total", 0.0)),
            vaep_offensive=float(row.get("vaep_offensive", 0.0)),
            vaep_defensive=float(row.get("vaep_defensive", 0.0)),
            vaep_per_90=(
                float(row["vaep_per_90"])
                if pd.notna(row.get("vaep_per_90"))
                else None
            ),
            xg_total=float(row.get("xg_total", 0.0)),
            xt_total=float(row.get("xt_total", 0.0)),
        )
