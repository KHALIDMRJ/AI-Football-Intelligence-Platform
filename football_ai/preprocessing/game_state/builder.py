"""
Game state builder.

For each action in a match, computes the binary labels used to train
the P_scores and P_concedes models:

    label_scores[i]   = 1 if the team in possession scores within
                        the next k actions,  else 0
    label_concedes[i] = 1 if the team in possession concedes within
                        the next k actions,  else 0

Also tracks the running score and goal difference.
"""

from __future__ import annotations

import pandas as pd

from football_ai.config import settings
from football_ai.constants import ActionResult, ActionType, Cols
from football_ai.logger import get_logger

logger = get_logger(__name__)


class GameStateBuilder:
    """
    Builds VAEP training labels and game-context columns for a match DataFrame.

    Added columns
    -------------
    - ``label_scores``   : int (0/1) — scored in next k actions?
    - ``label_concedes`` : int (0/1) — conceded in next k actions?
    - ``score_home``     : int — running home score at this moment
    - ``score_away``     : int — running away score at this moment
    - ``score_diff``     : int — home minus away at this moment

    Usage
    -----
    >>> builder = GameStateBuilder()
    >>> df = builder.build(actions_df, home_team_id="2761")
    """

    def __init__(self, k_actions: int | None = None) -> None:
        self.k = k_actions or settings.vaep.k_actions

    def build(self, df: pd.DataFrame, home_team_id: str) -> pd.DataFrame:
        """
        Compute game-state labels for all actions in ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Sorted SPADL actions for a single match (with possession columns).
        home_team_id : str
            The ID of the home team (used for score tracking).

        Returns
        -------
        pd.DataFrame
            Same DataFrame with label and score columns added.
        """
        df = df.copy().reset_index(drop=True)
        n = len(df)

        # ── Running score ──────────────────────────────────────────────────────
        home_score = [0] * n
        away_score = [0] * n
        current_home = 0
        current_away = 0

        for i, row in df.iterrows():
            if self._is_goal(row):
                if str(row.get(Cols.TEAM_ID, "")) == str(home_team_id):
                    current_home += 1
                else:
                    current_away += 1
            home_score[i] = current_home
            away_score[i] = current_away

        df["score_home"] = home_score
        df["score_away"] = away_score
        df["score_diff"] = df["score_home"] - df["score_away"]

        # ── VAEP labels ────────────────────────────────────────────────────────
        # For each action i, look at the next k actions and check if the
        # team in possession scores (label_scores) or concedes (label_concedes).

        team_col = df[Cols.POSSESSION_TEAM_ID].values if Cols.POSSESSION_TEAM_ID in df.columns \
            else df[Cols.TEAM_ID].values

        label_scores = [0] * n
        label_concedes = [0] * n

        # Pre-compute goal flags for speed
        is_goal = df.apply(self._is_goal, axis=1).values
        goal_team = df[Cols.TEAM_ID].values

        for i in range(n):
            poss_team = team_col[i]
            future_end = min(i + self.k + 1, n)

            for j in range(i + 1, future_end):
                if is_goal[j]:
                    if goal_team[j] == poss_team:
                        label_scores[i] = 1
                    else:
                        label_concedes[i] = 1

        df[Cols.LABEL_SCORES] = label_scores
        df[Cols.LABEL_CONCEDES] = label_concedes

        scored_pct = sum(label_scores) / n * 100 if n else 0
        conceded_pct = sum(label_concedes) / n * 100 if n else 0

        logger.info(
            "Game state labels (k=%d): %.1f%% score, %.1f%% concede",
            self.k,
            scored_pct,
            conceded_pct,
        )

        return df

    @staticmethod
    def _is_goal(row: pd.Series) -> bool:
        """Return True if this action is a goal."""
        result = str(row.get(Cols.RESULT, "")).lower()
        atype = str(row.get(Cols.ACTION_TYPE, "")).lower()
        return result == ActionResult.GOAL.value or (
            atype == ActionType.SHOT.value and result == ActionResult.GOAL.value
        )
