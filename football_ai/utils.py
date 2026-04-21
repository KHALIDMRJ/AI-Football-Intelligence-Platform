"""
Shared utility functions used across all modules.

Keep this file lean — only genuinely reusable, domain-agnostic helpers.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from football_ai.constants import GOAL_POST_BOTTOM, GOAL_POST_TOP, GOAL_X, GOAL_Y
from football_ai.logger import get_logger

logger = get_logger(__name__)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def distance_to_goal(x: float, y: float) -> float:
    """Euclidean distance from position (x, y) to goal centre."""
    return math.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)


def angle_to_goal(x: float, y: float) -> float:
    """
    Angle (radians) subtended at position (x, y) by the goal opening.
    Uses the law of cosines on the triangle formed by the position
    and the two goal posts.
    Returns 0 if position is behind the goal line.
    """
    if x >= GOAL_X:
        return 0.0

    # Distances to each post
    a = math.sqrt((GOAL_X - x) ** 2 + (GOAL_POST_TOP - y) ** 2)
    b = math.sqrt((GOAL_X - x) ** 2 + (GOAL_POST_BOTTOM - y) ** 2)
    c = GOAL_POST_TOP - GOAL_POST_BOTTOM  # goal width = 8 m

    # Cosine rule: c² = a² + b² - 2ab·cos(θ)  →  θ = arccos(...)
    cos_theta = (a**2 + b**2 - c**2) / (2 * a * b + 1e-9)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


def zone_id(
    x: float,
    y: float,
    zones_x: int = 18,
    zones_y: int = 12,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
) -> int:
    """
    Map pitch coordinates to a flat zone index [0, zones_x * zones_y).
    Zone 0 is bottom-left; zones fill left-to-right, bottom-to-top.
    """
    col = int(min(x / pitch_length * zones_x, zones_x - 1))
    row = int(min(y / pitch_width * zones_y, zones_y - 1))
    return row * zones_x + col


def zone_to_coords(
    zone: int,
    zones_x: int = 18,
    zones_y: int = 12,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
) -> tuple[float, float]:
    """Return the centre (x, y) of a zone given its flat index."""
    row = zone // zones_x
    col = zone % zones_x
    x = (col + 0.5) / zones_x * pitch_length
    y = (row + 0.5) / zones_y * pitch_width
    return x, y


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def safe_cast(series: pd.Series, dtype: type, default: object = None) -> pd.Series:
    """Cast a pandas Series to dtype, coercing errors to ``default``."""
    try:
        return series.astype(dtype)
    except (ValueError, TypeError):
        return pd.to_numeric(series, errors="coerce").fillna(default)


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required column is missing from df."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist. Return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, path: str | Path, *, verbose: bool = True) -> Path:
    """Save a DataFrame to Parquet and return the final path."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False, compression="snappy")
    if verbose:
        logger.info("Saved %d rows -> %s", len(df), p)
    return p


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a Parquet file and return a DataFrame. Raises FileNotFoundError."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet file not found: {p}")
    df = pd.read_parquet(p)
    logger.debug("Loaded %d rows <- %s", len(df), p)
    return df


# ── ID / hashing helpers ──────────────────────────────────────────────────────

def stable_hash(*parts: object) -> str:
    """Return an 8-char stable hex hash from an arbitrary set of values."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()[:8]


# ── Timing helpers ────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str = "block") -> Generator[None, None, None]:
    """Context manager that logs the elapsed time of a code block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("[timer] %s -- %.3f s", label, elapsed)


# ── Math helpers ──────────────────────────────────────────────────────────────

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns ``default`` when denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def clip_probability(p: float) -> float:
    """Clip a probability value to [0, 1]."""
    return float(np.clip(p, 0.0, 1.0))


def minutes_to_seconds(minutes: float) -> float:
    return minutes * 60.0


def seconds_to_minutes(seconds: float) -> float:
    return seconds / 60.0
