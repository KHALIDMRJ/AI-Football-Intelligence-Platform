"""
Temporal and sequential features.

For each action, looks at the preceding actions in the same possession/chain
to capture speed-of-play, sequence patterns, and ball-movement history.

All new columns are prefixed with ``f_tm_``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_ai.constants import ActionType, Cols
from football_ai.logger import get_logger

logger = get_logger(__name__)

# Action types we one-hot encode in the "last k actions" block
_SEQUENCE_ACTION_TYPES: list[str] = [
    ActionType.PASS.value,
    ActionType.SHOT.value,
    ActionType.DRIBBLE.value,
    ActionType.CROSS.value,
    ActionType.CARRY.value,
    ActionType.CLEARANCE.value,
    ActionType.INTERCEPTION.value,
    ActionType.TACKLE.value,
    "other",
]

_K_PREV: int = 3   # number of preceding actions to encode


def _type_bucket(action_type: str) -> str:
    """Map an action type to one of the _SEQUENCE_ACTION_TYPES buckets."""
    if action_type in _SEQUENCE_ACTION_TYPES:
        return action_type
    return "other"


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal and sequential feature columns to a SPADL actions DataFrame.

    Expects columns: timestamp, period, team_id, possession_id,
    chain_id, action_type, start_x, start_y, end_x, end_y, duration.

    Parameters
    ----------
    df : pd.DataFrame
        SPADL actions for a single match, sorted by (period, timestamp).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with temporal feature columns appended.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)

    ts   = df[Cols.TIMESTAMP].to_numpy(dtype=float)
    sx   = df[Cols.START_X].to_numpy(dtype=float)
    sy   = df[Cols.START_Y].to_numpy(dtype=float)
    ex   = df[Cols.END_X].to_numpy(dtype=float)
    ey   = df[Cols.END_Y].to_numpy(dtype=float)
    atypes = df[Cols.ACTION_TYPE].to_numpy(dtype=str)

    poss_ids = (
        df[Cols.POSSESSION_ID].to_numpy()
        if Cols.POSSESSION_ID in df.columns
        else np.zeros(n)
    )
    chain_idx = (
        df[Cols.CHAIN_INDEX].to_numpy(dtype=int)
        if Cols.CHAIN_INDEX in df.columns
        else np.zeros(n, dtype=int)
    )

    # ── Time gap to previous action ───────────────────────────────────────────
    gap = np.zeros(n, dtype=float)
    for i in range(1, n):
        same_poss = poss_ids[i] == poss_ids[i - 1]
        gap[i] = ts[i] - ts[i - 1] if same_poss else 0.0

    df["f_tm_time_gap_prev"]    = gap
    df["f_tm_time_gap_prev_sq"] = gap ** 2   # non-linear signal

    # ── Possession duration (time from possession start to this action) ───────
    poss_start_ts: dict[object, float] = {}
    poss_dur = np.zeros(n, dtype=float)
    for i in range(n):
        pid = poss_ids[i]
        if pid not in poss_start_ts:
            poss_start_ts[pid] = ts[i]
        poss_dur[i] = ts[i] - poss_start_ts[pid]

    df["f_tm_possession_duration"] = poss_dur

    # ── Chain index (position within the 2–5 action chain) ───────────────────
    df["f_tm_chain_index"] = chain_idx.astype(int)

    # ── Speed of play (distance / time_gap) ──────────────────────────────────
    dist = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
    _speed = np.zeros(n, dtype=float)
    np.divide(dist, gap, out=_speed, where=(gap > 0.01))
    df["f_tm_speed_of_play"] = np.clip(_speed, 0.0, 50.0)

    # ── Action duration ───────────────────────────────────────────────────────
    dur_col = "duration" if "duration" in df.columns else None
    if dur_col:
        df["f_tm_action_duration"] = df[dur_col].fillna(0.0)
    else:
        df["f_tm_action_duration"] = 0.0

    # ── Period minute (0-based within the period) ─────────────────────────────
    # Approximates elapsed time in the current period
    if Cols.MINUTE in df.columns:
        df["f_tm_minute"]        = df[Cols.MINUTE].astype(int)
        df["f_tm_minute_norm"]   = df[Cols.MINUTE].astype(float) / 90.0
        df["f_tm_late_game"]     = (df[Cols.MINUTE] >= 75).astype(int)
        df["f_tm_extra_time"]    = (df[Cols.MINUTE] >= 90).astype(int)
    else:
        df["f_tm_minute"]      = 0
        df["f_tm_minute_norm"] = 0.0
        df["f_tm_late_game"]   = 0
        df["f_tm_extra_time"]  = 0

    # ── Period ────────────────────────────────────────────────────────────────
    if Cols.PERIOD in df.columns:
        df["f_tm_period"] = df[Cols.PERIOD].astype(int)
    else:
        df["f_tm_period"] = 1

    # ── Last-k action types: one-hot encoded for each of the k prior actions ──
    # For each of k=1..3 steps back, encode the action type as a one-hot vector.
    for k in range(1, _K_PREV + 1):
        prev_type = ["unknown"] * n
        for i in range(k, n):
            if poss_ids[i] == poss_ids[i - k]:
                prev_type[i] = _type_bucket(atypes[i - k])

        for atype in _SEQUENCE_ACTION_TYPES:
            col = f"f_tm_prev{k}_{atype}"
            df[col] = [int(pt == atype) for pt in prev_type]

    # ── Previous action end coords (spatial momentum) ─────────────────────────
    prev_ex = np.zeros(n, dtype=float)
    prev_ey = np.zeros(n, dtype=float)
    for i in range(1, n):
        if poss_ids[i] == poss_ids[i - 1]:
            prev_ex[i] = ex[i - 1]
            prev_ey[i] = ey[i - 1]

    df["f_tm_prev_end_x"] = prev_ex
    df["f_tm_prev_end_y"] = prev_ey

    # Displacement between previous end and current start
    df["f_tm_displacement_x"] = sx - prev_ex
    df["f_tm_displacement_y"] = sy - prev_ey

    logger.debug(
        "Temporal features added: %d columns",
        len([c for c in df.columns if c.startswith("f_tm_")]),
    )
    return df
