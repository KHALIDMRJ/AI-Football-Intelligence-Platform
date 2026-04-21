"""
Contextual features.

Captures game-state context at the moment each action occurs:
- current score difference
- action type, body part, result (one-hot)
- under pressure flag
- team possession flag
- match phase (open play, set piece, corner, free kick)

All new columns are prefixed with ``f_ctx_``.
"""

from __future__ import annotations

import pandas as pd

from football_ai.constants import ActionResult, ActionType, BodyPart, Cols
from football_ai.logger import get_logger

logger = get_logger(__name__)

# Canonical lists for one-hot encoding — order must be stable across runs
_ACTION_TYPES: list[str] = [
    ActionType.PASS.value,
    ActionType.SHOT.value,
    ActionType.DRIBBLE.value,
    ActionType.CROSS.value,
    ActionType.CARRY.value,
    ActionType.CLEARANCE.value,
    ActionType.INTERCEPTION.value,
    ActionType.TACKLE.value,
    ActionType.CORNER.value,
    ActionType.FREE_KICK.value,
    ActionType.THROW_IN.value,
    ActionType.GOAL_KICK.value,
    ActionType.KEEPER_SAVE.value,
    ActionType.HEADER.value,
    "other",
]

_BODY_PARTS: list[str] = [
    BodyPart.RIGHT_FOOT.value,
    BodyPart.LEFT_FOOT.value,
    BodyPart.HEAD.value,
    BodyPart.OTHER.value,
    BodyPart.UNKNOWN.value,
]

_RESULTS: list[str] = [
    ActionResult.SUCCESS.value,
    ActionResult.FAIL.value,
    ActionResult.GOAL.value,
    ActionResult.OUT.value,
    ActionResult.OFFSIDE.value,
    ActionResult.UNKNOWN.value,
]

_SET_PIECE_TYPES: frozenset[str] = frozenset({
    ActionType.CORNER.value,
    ActionType.FREE_KICK.value,
    ActionType.THROW_IN.value,
    ActionType.GOAL_KICK.value,
})


def _type_bucket(val: str, buckets: list[str]) -> str:
    return val if val in buckets else "other"


def compute_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add contextual feature columns to a SPADL actions DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        SPADL actions with game-state columns present.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with contextual feature columns appended.
    """
    df = df.copy()
    n = len(df)

    # ── Score difference ──────────────────────────────────────────────────────
    if "score_diff" in df.columns:
        df["f_ctx_score_diff"]      = df["score_diff"].astype(int)
        df["f_ctx_winning"]         = (df["score_diff"] > 0).astype(int)
        df["f_ctx_losing"]          = (df["score_diff"] < 0).astype(int)
        df["f_ctx_drawing"]         = (df["score_diff"] == 0).astype(int)
        df["f_ctx_score_diff_abs"]  = df["score_diff"].abs().astype(int)
    else:
        for col in ["f_ctx_score_diff", "f_ctx_winning", "f_ctx_losing",
                    "f_ctx_drawing", "f_ctx_score_diff_abs"]:
            df[col] = 0

    # ── Under pressure ────────────────────────────────────────────────────────
    df["f_ctx_under_pressure"] = (
        df[Cols.UNDER_PRESSURE].astype(bool).astype(int)
        if Cols.UNDER_PRESSURE in df.columns
        else 0
    )

    # ── Action type one-hot ───────────────────────────────────────────────────
    action_types = df[Cols.ACTION_TYPE].astype(str).tolist()
    for atype in _ACTION_TYPES:
        df[f"f_ctx_type_{atype}"] = [
            int(_type_bucket(at, _ACTION_TYPES) == atype) for at in action_types
        ]

    # ── Body part one-hot ─────────────────────────────────────────────────────
    body_parts = (
        df[Cols.BODY_PART].astype(str).tolist()
        if Cols.BODY_PART in df.columns
        else ["unknown"] * n
    )
    for bp in _BODY_PARTS:
        df[f"f_ctx_body_{bp}"] = [int(b == bp) for b in body_parts]

    # ── Result one-hot ────────────────────────────────────────────────────────
    results = df[Cols.RESULT].astype(str).tolist() if Cols.RESULT in df.columns else ["unknown"] * n
    for res in _RESULTS:
        df[f"f_ctx_result_{res}"] = [int(r == res) for r in results]

    # ── Set piece flag ────────────────────────────────────────────────────────
    df["f_ctx_set_piece"] = [int(at in _SET_PIECE_TYPES) for at in action_types]

    # ── Counter-attack proxy: possession started in own half and reached
    #    attacking third within <= 3 actions
    if Cols.POSSESSION_ID in df.columns and "f_sp_third_start" in df.columns:
        # Simple proxy: action in attacking third after possession starting in own third
        df["f_ctx_counter_attack_proxy"] = (
            (df["f_sp_third_start"] == 2) & (df["f_tm_possession_duration"] < 10.0)
            if "f_tm_possession_duration" in df.columns
            else pd.Series(0, index=df.index)
        ).astype(int)
    else:
        df["f_ctx_counter_attack_proxy"] = 0

    # ── Possession team same as action team ───────────────────────────────────
    if Cols.POSSESSION_TEAM_ID in df.columns and Cols.TEAM_ID in df.columns:
        df["f_ctx_in_possession"] = (
            df[Cols.TEAM_ID].astype(str) == df[Cols.POSSESSION_TEAM_ID].astype(str)
        ).astype(int)
    else:
        df["f_ctx_in_possession"] = 1

    logger.debug(
        "Contextual features added: %d columns",
        len([c for c in df.columns if c.startswith("f_ctx_")]),
    )
    return df
