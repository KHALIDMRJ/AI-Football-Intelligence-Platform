"""
Domain constants for the Football AI Platform.

All magic numbers live here so the rest of the codebase stays clean.
"""

from __future__ import annotations

from enum import Enum, StrEnum

# ── Pitch geometry (StatsBomb standard, metres) ───────────────────────────────

PITCH_LENGTH: float = 120.0
PITCH_WIDTH: float = 80.0
GOAL_X: float = 120.0
GOAL_Y: float = 40.0
GOAL_POST_TOP: float = 44.0
GOAL_POST_BOTTOM: float = 36.0
GOAL_WIDTH: float = GOAL_POST_TOP - GOAL_POST_BOTTOM  # 8.0 m


# ── xT grid dimensions ────────────────────────────────────────────────────────

XT_ZONES_X: int = 18
XT_ZONES_Y: int = 12
XT_TOTAL_ZONES: int = XT_ZONES_X * XT_ZONES_Y  # 216


# ── VAEP parameters ───────────────────────────────────────────────────────────

VAEP_K_ACTIONS: int = 10          # look-ahead window
VAEP_MIN_MINUTES: int = 20        # minimum for VAEP/90 computation
SECONDS_PER_90: float = 5_400.0   # 90 * 60


# ── Possession detection ──────────────────────────────────────────────────────

POSSESSION_MAX_GAP_SECONDS: float = 5.0
POSSESSION_MIN_CHAIN: int = 2
POSSESSION_MAX_CHAIN: int = 5


# ── Action types (canonical internal names) ───────────────────────────────────

class ActionType(StrEnum):
    PASS = "pass"
    SHOT = "shot"
    DRIBBLE = "dribble"
    CROSS = "cross"
    CARRY = "carry"
    CLEARANCE = "clearance"
    INTERCEPTION = "interception"
    TACKLE = "tackle"
    FOUL = "foul"
    CORNER = "corner"
    FREE_KICK = "free_kick"
    THROW_IN = "throw_in"
    GOAL_KICK = "goal_kick"
    KEEPER_SAVE = "keeper_save"
    KEEPER_CLAIM = "keeper_claim"
    BLOCK = "block"
    HEADER = "header"
    UNKNOWN = "unknown"


# ── Body parts ────────────────────────────────────────────────────────────────

class BodyPart(StrEnum):
    RIGHT_FOOT = "right_foot"
    LEFT_FOOT = "left_foot"
    HEAD = "head"
    OTHER = "other"
    UNKNOWN = "unknown"


# ── Action results ────────────────────────────────────────────────────────────

class ActionResult(StrEnum):
    SUCCESS = "success"
    FAIL = "fail"
    GOAL = "goal"
    OUT = "out"
    OFFSIDE = "offside"
    UNKNOWN = "unknown"


# ── Period IDs ────────────────────────────────────────────────────────────────

class Period(int, Enum):
    FIRST_HALF = 1
    SECOND_HALF = 2
    EXTRA_TIME_FIRST = 3
    EXTRA_TIME_SECOND = 4
    PENALTY_SHOOTOUT = 5


# ── Pitch zones (named zones for tactical analysis) ───────────────────────────

class PitchZone(StrEnum):
    OWN_THIRD_LEFT = "own_third_left"
    OWN_THIRD_CENTRE = "own_third_centre"
    OWN_THIRD_RIGHT = "own_third_right"
    MID_THIRD_LEFT = "mid_third_left"
    MID_THIRD_CENTRE = "mid_third_centre"
    MID_THIRD_RIGHT = "mid_third_right"
    ATT_THIRD_LEFT = "att_third_left"
    ATT_THIRD_CENTRE = "att_third_centre"
    ATT_THIRD_RIGHT = "att_third_right"
    PENALTY_AREA = "penalty_area"
    SIX_YARD_BOX = "six_yard_box"


# ── Column name constants (prevent typos in string-based column access) ────────

class Cols:
    """Canonical DataFrame column names."""

    MATCH_ID = "match_id"
    ACTION_ID = "action_id"
    INDEX = "index"
    PERIOD = "period"
    TIMESTAMP = "timestamp"
    MINUTE = "minute"
    SECOND = "second"
    TEAM_ID = "team_id"
    TEAM_NAME = "team_name"
    PLAYER_ID = "player_id"
    PLAYER_NAME = "player_name"
    POSITION = "position"
    ACTION_TYPE = "action_type"
    BODY_PART = "body_part"
    RESULT = "result"
    START_X = "start_x"
    START_Y = "start_y"
    END_X = "end_x"
    END_Y = "end_y"
    DURATION = "duration"
    UNDER_PRESSURE = "under_pressure"
    POSSESSION_ID = "possession_id"
    POSSESSION_TEAM_ID = "possession_team_id"
    CHAIN_ID = "chain_id"
    CHAIN_INDEX = "chain_index"
    ZONE_ID = "zone_id"
    XG = "xg"
    XT_START = "xt_start"
    XT_END = "xt_end"
    XT_DELTA = "xt_delta"
    P_SCORES = "p_scores"
    P_CONCEDES = "p_concedes"
    VAEP_VALUE = "vaep_value"
    VAEP_OFFENSIVE = "vaep_offensive"
    VAEP_DEFENSIVE = "vaep_defensive"
    LABEL_SCORES = "label_scores"
    LABEL_CONCEDES = "label_concedes"
