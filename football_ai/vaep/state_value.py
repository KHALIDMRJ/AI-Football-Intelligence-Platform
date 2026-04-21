"""
Game state value computation.

Computes V(Sᵢ) for every action in a feature DataFrame:

    V(Sᵢ) = P^k_scores(Sᵢ) − P^k_concedes(Sᵢ)

where:
    P^k_scores(Sᵢ)   = probability the team in possession scores
                        within the next k actions, given state Sᵢ
    P^k_concedes(Sᵢ) = probability the team in possession concedes
                        within the next k actions, given state Sᵢ

Both probabilities are outputs of the fitted ML models from Phase 5.
The result is a DataFrame with three new columns appended:

    p_scores   — P^k_scores  ∈ [0, 1]
    p_concedes — P^k_concedes ∈ [0, 1]
    state_value — V(Sᵢ) = p_scores − p_concedes ∈ [-1, 1]
"""

from __future__ import annotations

import pandas as pd

from football_ai.constants import Cols
from football_ai.logger import get_logger
from football_ai.ml.models.p_concedes_model import PConcedesModel
from football_ai.ml.models.p_scores_model import PScoresModel

logger = get_logger(__name__)

_STATE_VALUE_COL = "state_value"


class StateValueComputer:
    """
    Computes V(Sᵢ) = P_scores(Sᵢ) − P_concedes(Sᵢ) for every action.

    Requires fitted PScoresModel and PConcedesModel instances.  Both models
    are called independently; their probabilities are clipped to [0, 1]
    before the subtraction.

    Parameters
    ----------
    p_scores_model : PScoresModel
        Fitted scoring probability model.
    p_concedes_model : PConcedesModel
        Fitted conceding probability model.

    Usage
    -----
    >>> computer = StateValueComputer(p_scores_model, p_concedes_model)
    >>> df = computer.compute(feature_df)
    >>> df[["p_scores", "p_concedes", "state_value"]].head()
    """

    def __init__(
        self,
        p_scores_model: PScoresModel,
        p_concedes_model: PConcedesModel,
    ) -> None:
        self.p_scores_model = p_scores_model
        self.p_concedes_model = p_concedes_model

    def compute(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Append p_scores, p_concedes, and state_value columns to the DataFrame.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature matrix produced by FeatureAssembler.  Must contain
            the ``f_*`` feature columns that the models were trained on.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with three new columns:
            ``p_scores``, ``p_concedes``, ``state_value``.
        """
        if feature_df.empty:
            logger.warning("StateValueComputer.compute() called on empty DataFrame.")
            feature_df = feature_df.copy()
            feature_df[Cols.P_SCORES]    = pd.Series(dtype=float)
            feature_df[Cols.P_CONCEDES]  = pd.Series(dtype=float)
            feature_df[_STATE_VALUE_COL] = pd.Series(dtype=float)
            return feature_df

        df = feature_df.copy()

        logger.info(
            "Computing state values for %d actions.", len(df)
        )

        # ── P_scores ──────────────────────────────────────────────────────────
        p_scores_series = self.p_scores_model.score_actions(df)
        df[Cols.P_SCORES] = (
            p_scores_series
            .clip(0.0, 1.0)
            .fillna(0.0)
        )

        # ── P_concedes ────────────────────────────────────────────────────────
        p_concedes_series = self.p_concedes_model.score_actions(df)
        df[Cols.P_CONCEDES] = (
            p_concedes_series
            .clip(0.0, 1.0)
            .fillna(0.0)
        )

        # ── V(Sᵢ) = P_scores − P_concedes ────────────────────────────────────
        df[_STATE_VALUE_COL] = (
            df[Cols.P_SCORES] - df[Cols.P_CONCEDES]
        ).clip(-1.0, 1.0)

        logger.info(
            "State values computed.  "
            "V(S) range: [%.4f, %.4f]  mean=%.4f",
            float(df[_STATE_VALUE_COL].min()),
            float(df[_STATE_VALUE_COL].max()),
            float(df[_STATE_VALUE_COL].mean()),
        )

        return df

    def state_value_series(self, feature_df: pd.DataFrame) -> pd.Series:
        """
        Return only the V(Sᵢ) Series without modifying the input DataFrame.

        Convenience wrapper around ``compute()`` for use in pipelines that
        manage their own column assignments.
        """
        result = self.compute(feature_df)
        return result[_STATE_VALUE_COL]
