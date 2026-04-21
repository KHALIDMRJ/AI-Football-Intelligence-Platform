"""
Action value computation.

Computes V(aᵢ) for every action using the state-value deltas:

    V(aᵢ) = V(Sᵢ) − V(Sᵢ₋₁)

Because V(Sᵢ) = P_scores(Sᵢ) − P_concedes(Sᵢ), the action value
decomposes into two interpretable components:

    Δ_scores(aᵢ)   = P_scores(Sᵢ)   − P_scores(Sᵢ₋₁)
    Δ_concedes(aᵢ) = P_concedes(Sᵢ) − P_concedes(Sᵢ₋₁)

    V(aᵢ) = Δ_scores(aᵢ) − Δ_concedes(aᵢ)

Offensive and defensive decomposition
--------------------------------------
    vaep_offensive(aᵢ) = max(0,  Δ_scores(aᵢ))
                       + max(0, −Δ_concedes(aᵢ))   (reduced concede risk)

    vaep_defensive(aᵢ) = max(0, −Δ_scores(aᵢ))    (prevented scoring)
                       + max(0,  Δ_concedes(aᵢ))   (increased concede risk)
                       ... sign-flipped so defensive contribution is positive

Simpler and more common decomposition used here (matches VAEP paper):
    vaep_offensive(aᵢ) = max(0,  Δ_scores(aᵢ))
    vaep_defensive(aᵢ) = max(0, −Δ_concedes(aᵢ))

Both are always ≥ 0. Their difference is V(aᵢ) when both are positive,
but they can be computed independently for analytics purposes.

Columns added
-------------
    vaep_value        V(aᵢ)                  ∈ [−1, 1]
    vaep_offensive    max(0,  Δ_scores)       ∈ [0, 1]
    vaep_defensive    max(0, −Δ_concedes)     ∈ [0, 1]
    delta_p_scores    Δ_scores(aᵢ)           ∈ [−1, 1]
    delta_p_concedes  Δ_concedes(aᵢ)         ∈ [−1, 1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_ai.constants import Cols
from football_ai.logger import get_logger

logger = get_logger(__name__)

# Output column names
_VAEP_VALUE      = Cols.VAEP_VALUE
_VAEP_OFFENSIVE  = Cols.VAEP_OFFENSIVE
_VAEP_DEFENSIVE  = Cols.VAEP_DEFENSIVE
_DELTA_P_SCORES  = "delta_p_scores"
_DELTA_P_CONCEDES = "delta_p_concedes"
_STATE_VALUE_COL = "state_value"


class ActionValueComputer:
    """
    Computes per-action VAEP values from a DataFrame that already contains
    ``p_scores``, ``p_concedes``, and ``state_value`` columns (output of
    ``StateValueComputer.compute()``).

    The first action of every possession has no "previous" state, so
    V(aᵢ) is set to 0.0 for those rows (no information gain).

    Usage
    -----
    >>> computer = ActionValueComputer()
    >>> df = computer.compute(df_with_state_values)
    >>> df[["vaep_value", "vaep_offensive", "vaep_defensive"]].describe()
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-action VAEP columns in-place on a copy.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that already has ``p_scores``, ``p_concedes``,
            ``state_value``, ``possession_id`` columns.  This is the
            output of ``StateValueComputer.compute()``.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with five new columns appended:
            ``delta_p_scores``, ``delta_p_concedes``,
            ``vaep_value``, ``vaep_offensive``, ``vaep_defensive``.
        """
        self._validate(df)

        if df.empty:
            return self._add_empty_cols(df.copy())

        df = df.copy().reset_index(drop=True)
        n = len(df)

        p_scores   = df[Cols.P_SCORES].to_numpy(dtype=float)
        p_concedes = df[Cols.P_CONCEDES].to_numpy(dtype=float)
        state_val  = df[_STATE_VALUE_COL].to_numpy(dtype=float)

        poss_ids = (
            df[Cols.POSSESSION_ID].to_numpy()
            if Cols.POSSESSION_ID in df.columns
            else np.zeros(n, dtype=int)
        )

        delta_scores   = np.zeros(n, dtype=float)
        delta_concedes = np.zeros(n, dtype=float)
        vaep_value     = np.zeros(n, dtype=float)

        for i in range(1, n):
            # Only compute delta within the same possession.
            # Across possessions the state change is not attributable
            # to the current action.
            if poss_ids[i] == poss_ids[i - 1]:
                delta_scores[i]   = p_scores[i]   - p_scores[i - 1]
                delta_concedes[i] = p_concedes[i]  - p_concedes[i - 1]
                vaep_value[i]     = state_val[i]   - state_val[i - 1]

        # ── Decomposition ─────────────────────────────────────────────────────
        # Offensive: positive scoring delta (increased threat)
        vaep_offensive = np.maximum(0.0, delta_scores)

        # Defensive: negative conceding delta (reduced concede risk)
        # We flip the sign so defensive contributions are positive numbers.
        vaep_defensive = np.maximum(0.0, -delta_concedes)

        # ── Clip to reasonable bounds ─────────────────────────────────────────
        vaep_value     = np.clip(vaep_value,    -1.0, 1.0)
        vaep_offensive = np.clip(vaep_offensive,  0.0, 1.0)
        vaep_defensive = np.clip(vaep_defensive,  0.0, 1.0)

        df[_DELTA_P_SCORES]   = delta_scores
        df[_DELTA_P_CONCEDES] = delta_concedes
        df[_VAEP_VALUE]       = vaep_value
        df[_VAEP_OFFENSIVE]   = vaep_offensive
        df[_VAEP_DEFENSIVE]   = vaep_defensive

        logger.info(
            "Action values computed.  "
            "VAEP range: [%.4f, %.4f]  mean=%.4f  "
            "Off mean=%.4f  Def mean=%.4f",
            float(vaep_value.min()),
            float(vaep_value.max()),
            float(vaep_value.mean()),
            float(vaep_offensive.mean()),
            float(vaep_defensive.mean()),
        )

        return df

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        required = [Cols.P_SCORES, Cols.P_CONCEDES, _STATE_VALUE_COL]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"ActionValueComputer requires columns {required}. "
                f"Missing: {missing}. "
                f"Run StateValueComputer.compute() first."
            )

    @staticmethod
    def _add_empty_cols(df: pd.DataFrame) -> pd.DataFrame:
        for col in [_DELTA_P_SCORES, _DELTA_P_CONCEDES,
                    _VAEP_VALUE, _VAEP_OFFENSIVE, _VAEP_DEFENSIVE]:
            df[col] = pd.Series(dtype=float)
        return df
