"""
Possession detector and action-chain builder.

Two concepts:
- Possession: continuous sequence of actions by the same team, broken
  when the ball changes team or there is a gap > max_gap_seconds.
- Action chain: sub-sequence of 2–5 pass-type actions within a possession
  that we use for training VAEP labels.
"""

from __future__ import annotations

import uuid

import pandas as pd

from football_ai.config import settings
from football_ai.constants import ActionType, Cols
from football_ai.logger import get_logger

logger = get_logger(__name__)

# Action types that can be part of a chain (ball-progressing actions)
_CHAIN_ACTION_TYPES: frozenset[str] = frozenset({
    ActionType.PASS.value,
    ActionType.CROSS.value,
    ActionType.CARRY.value,
    ActionType.DRIBBLE.value,
    ActionType.SHOT.value,
    ActionType.HEADER.value,
    ActionType.FREE_KICK.value,
    ActionType.CORNER.value,
})


class PossessionDetector:
    """
    Assigns possession_id and chain_id to each action in a match.

    A new possession starts when:
    - The team in possession changes
    - The time gap between consecutive actions exceeds ``max_gap_seconds``

    A new chain starts when:
    - A new possession starts
    - A defensive action (clearance, interception, tackle) occurs
    - The chain has reached ``max_chain_length`` actions

    Usage
    -----
    >>> detector = PossessionDetector()
    >>> df = detector.detect(actions_df)
    """

    def __init__(
        self,
        max_gap_seconds: float | None = None,
        min_chain_length: int | None = None,
        max_chain_length: int | None = None,
    ) -> None:
        self.max_gap = max_gap_seconds or settings.possession.max_gap_seconds
        self.min_chain = min_chain_length or settings.possession.min_chain_length
        self.max_chain = max_chain_length or settings.possession.max_chain_length

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ``possession_id``, ``possession_team_id``, ``chain_id``,
        and ``chain_index`` columns to the actions DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Sorted SPADL actions for a single match.

        Returns
        -------
        pd.DataFrame
            Same DataFrame with possession/chain columns added.
        """
        df = df.copy().reset_index(drop=True)

        possession_ids: list[int] = []
        possession_team_ids: list[str] = []
        chain_ids: list[str] = []
        chain_indices: list[int] = []

        current_possession = 0
        current_team = None
        current_chain = str(uuid.uuid4())[:8]
        chain_idx = 0
        prev_timestamp: float | None = None

        for i, row in df.iterrows():
            team = row.get(Cols.TEAM_ID, "")
            ts = row.get(Cols.TIMESTAMP, 0.0)
            atype = row.get(Cols.ACTION_TYPE, "")

            # Detect possession break
            gap = (ts - prev_timestamp) if prev_timestamp is not None else 0.0
            team_changed = current_team is not None and team != current_team
            gap_exceeded = gap > self.max_gap

            if team_changed or gap_exceeded:
                current_possession += 1
                current_chain = str(uuid.uuid4())[:8]
                chain_idx = 0
                current_team = team

            # Detect chain break (within same possession)
            elif self._is_chain_break(atype) or chain_idx >= self.max_chain:
                current_chain = str(uuid.uuid4())[:8]
                chain_idx = 0
            else:
                chain_idx += 1

            current_team = team
            prev_timestamp = ts

            possession_ids.append(current_possession)
            possession_team_ids.append(str(team))
            chain_ids.append(current_chain)
            chain_indices.append(chain_idx)

        df[Cols.POSSESSION_ID] = possession_ids
        df[Cols.POSSESSION_TEAM_ID] = possession_team_ids
        df[Cols.CHAIN_ID] = chain_ids
        df[Cols.CHAIN_INDEX] = chain_indices

        n_possessions = df[Cols.POSSESSION_ID].nunique()
        n_chains = df[Cols.CHAIN_ID].nunique()
        logger.info(
            "Possession detection: %d possessions, %d action chains",
            n_possessions,
            n_chains,
        )

        return df

    def _is_chain_break(self, action_type: str) -> bool:
        """Return True if this action type should start a new chain."""
        return action_type not in _CHAIN_ACTION_TYPES
