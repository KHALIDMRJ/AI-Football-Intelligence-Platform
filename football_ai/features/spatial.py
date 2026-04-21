"""
Spatial and geometric features.

Computes all position-based features for a SPADL action DataFrame:
- distance and angle to goal (start and end)
- zone ID at start and end
- progressive pass / carry flag
- half-space entry flag
- penalty-area entry flag
- six-yard-box flag
- distance covered by the action
- x-component progress (meters gained toward goal)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_ai.config import settings
from football_ai.constants import Cols
from football_ai.logger import get_logger
from football_ai.utils import angle_to_goal, distance_to_goal, zone_id

logger = get_logger(__name__)

# Penalty area bounds (StatsBomb standard)
_PENALTY_X_MIN: float = 102.0
_PENALTY_Y_MIN: float = 18.0
_PENALTY_Y_MAX: float = 62.0

# Six-yard box
_SIX_YARD_X_MIN: float = 114.0
_SIX_YARD_Y_MIN: float = 30.0
_SIX_YARD_Y_MAX: float = 50.0

# Half-space columns (x zones 4-5 and 12-13 out of 18 columns)
_HALF_SPACE_Y_RANGES: list[tuple[float, float]] = [
    (18.0, 30.0),   # left half-space
    (50.0, 62.0),   # right half-space
]


def _in_penalty_area(x: float, y: float) -> bool:
    return x >= _PENALTY_X_MIN and _PENALTY_Y_MIN <= y <= _PENALTY_Y_MAX


def _in_six_yard_box(x: float, y: float) -> bool:
    return x >= _SIX_YARD_X_MIN and _SIX_YARD_Y_MIN <= y <= _SIX_YARD_Y_MAX


def _in_half_space(y: float) -> bool:
    return any(lo <= y <= hi for lo, hi in _HALF_SPACE_Y_RANGES)


def compute_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add spatial feature columns to a SPADL actions DataFrame.

    Expects columns: start_x, start_y, end_x, end_y.
    All new columns are prefixed with ``f_sp_`` to avoid collisions.

    Parameters
    ----------
    df : pd.DataFrame
        SPADL actions with coordinate columns present.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with spatial feature columns appended.
    """
    df = df.copy()

    sx = df[Cols.START_X].to_numpy(dtype=float)
    sy = df[Cols.START_Y].to_numpy(dtype=float)
    ex = df[Cols.END_X].to_numpy(dtype=float)
    ey = df[Cols.END_Y].to_numpy(dtype=float)

    n = len(df)

    # ── Distance to goal ──────────────────────────────────────────────────────
    dist_start = np.array([distance_to_goal(sx[i], sy[i]) for i in range(n)])
    dist_end   = np.array([distance_to_goal(ex[i], ey[i]) for i in range(n)])

    df["f_sp_dist_to_goal_start"] = dist_start
    df["f_sp_dist_to_goal_end"]   = dist_end
    df["f_sp_dist_to_goal_delta"] = dist_start - dist_end  # positive = closer to goal

    # ── Angle to goal ─────────────────────────────────────────────────────────
    df["f_sp_angle_to_goal_start"] = [angle_to_goal(sx[i], sy[i]) for i in range(n)]
    df["f_sp_angle_to_goal_end"]   = [angle_to_goal(ex[i], ey[i]) for i in range(n)]

    # ── Zone IDs ──────────────────────────────────────────────────────────────
    zones_x = settings.pitch.zones_x
    zones_y = settings.pitch.zones_y
    pitch_l = settings.pitch.length
    pitch_w = settings.pitch.width

    df["f_sp_zone_start"] = [
        zone_id(sx[i], sy[i], zones_x, zones_y, pitch_l, pitch_w) for i in range(n)
    ]
    df["f_sp_zone_end"] = [
        zone_id(ex[i], ey[i], zones_x, zones_y, pitch_l, pitch_w) for i in range(n)
    ]
    df["f_sp_zone_same"]  = (df["f_sp_zone_start"] == df["f_sp_zone_end"]).astype(int)

    # ── Distance covered (vector length of the action) ────────────────────────
    df["f_sp_distance_covered"] = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)

    # ── X-progress: meters gained toward the opponent goal ───────────────────
    df["f_sp_x_progress"] = ex - sx   # positive = moving toward goal (x=120)

    # ── Pitch thirds (encoded as 0/1/2: own / mid / attacking) ───────────────
    df["f_sp_third_start"] = pd.cut(
        df[Cols.START_X],
        bins=[0, 40, 80, 120],
        labels=[0, 1, 2],
        include_lowest=True,
    ).astype(int)

    # ── Binary location flags ─────────────────────────────────────────────────
    df["f_sp_start_in_pen_area"]    = [int(_in_penalty_area(sx[i], sy[i])) for i in range(n)]
    df["f_sp_end_in_pen_area"]      = [int(_in_penalty_area(ex[i], ey[i])) for i in range(n)]
    df["f_sp_pen_area_entry"]       = (
        (df["f_sp_start_in_pen_area"] == 0) & (df["f_sp_end_in_pen_area"] == 1)
    ).astype(int)

    df["f_sp_start_in_six_yard"]    = [int(_in_six_yard_box(sx[i], sy[i])) for i in range(n)]
    df["f_sp_end_in_six_yard"]      = [int(_in_six_yard_box(ex[i], ey[i])) for i in range(n)]

    df["f_sp_start_in_half_space"]  = [int(_in_half_space(sy[i])) for i in range(n)]
    df["f_sp_end_in_half_space"]    = [int(_in_half_space(ey[i])) for i in range(n)]
    df["f_sp_half_space_entry"]     = (
        (df["f_sp_start_in_half_space"] == 0) & (df["f_sp_end_in_half_space"] == 1)
    ).astype(int)

    # ── Progressive action: end location at least 10 m closer to goal ────────
    df["f_sp_progressive"] = (df["f_sp_dist_to_goal_delta"] >= 10.0).astype(int)

    # ── Start location raw (useful for models that learn pitch regions) ───────
    df["f_sp_start_x"] = sx
    df["f_sp_start_y"] = sy
    df["f_sp_end_x"]   = ex
    df["f_sp_end_y"]   = ey

    # ── Normalised coordinates [0, 1] ─────────────────────────────────────────
    df["f_sp_start_x_norm"] = sx / pitch_l
    df["f_sp_start_y_norm"] = sy / pitch_w
    df["f_sp_end_x_norm"]   = ex / pitch_l
    df["f_sp_end_y_norm"]   = ey / pitch_w

    n_added = sum(1 for c in df.columns if c.startswith("f_sp_"))
    logger.debug("Spatial features added: %d columns", n_added)
    return df
