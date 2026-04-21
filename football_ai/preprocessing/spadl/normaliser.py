"""
SPADL normaliser.

Converts a list of RawEvent objects into SPADLAction objects — the canonical
8-attribute action format used by all downstream modules.

The 8 SPADL attributes are:
    1. time (timestamp in seconds)
    2. start location (x, y)
    3. end location (x, y)
    4. involved player
    5. team
    6. action type
    7. body part
    8. result
"""

from __future__ import annotations

import hashlib

from football_ai.constants import ActionResult, ActionType, BodyPart
from football_ai.logger import get_logger
from football_ai.schemas import RawEvent, SPADLAction

logger = get_logger(__name__)


# ── Event-type mapping: vendor string → ActionType ────────────────────────────

_TYPE_MAP: dict[str, ActionType] = {
    # Passes
    "pass": ActionType.PASS,
    "passes": ActionType.PASS,
    "cross": ActionType.CROSS,
    "corner": ActionType.CORNER,
    "free kick": ActionType.FREE_KICK,
    "free_kick": ActionType.FREE_KICK,
    "throw-in": ActionType.THROW_IN,
    "throw_in": ActionType.THROW_IN,
    "goal kick": ActionType.GOAL_KICK,
    "goal_kick": ActionType.GOAL_KICK,
    # Shots
    "shot": ActionType.SHOT,
    "shots": ActionType.SHOT,
    "penalty": ActionType.SHOT,
    # Ball control
    "dribble": ActionType.DRIBBLE,
    "carries": ActionType.CARRY,
    "carry": ActionType.CARRY,
    "ball receipt*": ActionType.PASS,      # treat receipt as part of a pass sequence
    "ball_receipt": ActionType.PASS,
    # Defending
    "clearance": ActionType.CLEARANCE,
    "interception": ActionType.INTERCEPTION,
    "block": ActionType.BLOCK,
    "tackle": ActionType.TACKLE,
    "duel": ActionType.TACKLE,
    "pressure": ActionType.TACKLE,
    "ball recovery": ActionType.INTERCEPTION,
    "ball_recovery": ActionType.INTERCEPTION,
    # Fouls
    "foul committed": ActionType.FOUL,
    "foul_committed": ActionType.FOUL,
    "foul won": ActionType.FOUL,
    "foul_won": ActionType.FOUL,
    # Goalkeeper
    "goal keeper": ActionType.KEEPER_SAVE,
    "goalkeeper": ActionType.KEEPER_SAVE,
    "keeper_save": ActionType.KEEPER_SAVE,
    "keeper_claim": ActionType.KEEPER_CLAIM,
    # Headers
    "header": ActionType.HEADER,
    # Misc
    "dispossessed": ActionType.UNKNOWN,
    "miscontrol": ActionType.UNKNOWN,
    "offside": ActionType.UNKNOWN,
    "substitution": ActionType.UNKNOWN,
    "half start": ActionType.UNKNOWN,
    "half_start": ActionType.UNKNOWN,
    "half end": ActionType.UNKNOWN,
    "half_end": ActionType.UNKNOWN,
    "starting xi": ActionType.UNKNOWN,
    "tactical shift": ActionType.UNKNOWN,
    "player on": ActionType.UNKNOWN,
    "player off": ActionType.UNKNOWN,
    "bad behaviour": ActionType.UNKNOWN,
    "injury stoppage": ActionType.UNKNOWN,
}

# ── Body part mapping ─────────────────────────────────────────────────────────

_BODY_PART_MAP: dict[str, BodyPart] = {
    "right foot": BodyPart.RIGHT_FOOT,
    "right_foot": BodyPart.RIGHT_FOOT,
    "left foot": BodyPart.LEFT_FOOT,
    "left_foot": BodyPart.LEFT_FOOT,
    "head": BodyPart.HEAD,
    "chest": BodyPart.OTHER,
    "knee": BodyPart.OTHER,
    "no touch": BodyPart.OTHER,
    "no_touch": BodyPart.OTHER,
}

# ── Result mapping ────────────────────────────────────────────────────────────

_RESULT_MAP: dict[str, ActionResult] = {
    "goal": ActionResult.GOAL,
    "success": ActionResult.SUCCESS,
    "success in play": ActionResult.SUCCESS,
    "success out": ActionResult.SUCCESS,
    "complete": ActionResult.SUCCESS,
    "incomplete": ActionResult.FAIL,
    "fail": ActionResult.FAIL,
    "out": ActionResult.OUT,
    "wayward": ActionResult.FAIL,
    "blocked": ActionResult.FAIL,
    "saved": ActionResult.FAIL,
    "saved off target": ActionResult.FAIL,
    "saved to post": ActionResult.FAIL,
    "off t": ActionResult.FAIL,
    "post": ActionResult.FAIL,
    "no touch": ActionResult.FAIL,
    "offside": ActionResult.OFFSIDE,
}


def _parse_timestamp(ts: str) -> float:
    """Convert 'HH:MM:SS.mmm' to seconds (float)."""
    try:
        parts = ts.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
    except (ValueError, IndexError):
        pass
    return 0.0


def _map_type(event_type: str | None) -> str:
    if not event_type:
        return ActionType.UNKNOWN.value
    return _TYPE_MAP.get(event_type.lower().strip(), ActionType.UNKNOWN).value


def _map_body_part(body_part: str | None) -> str:
    if not body_part:
        return BodyPart.UNKNOWN.value
    return _BODY_PART_MAP.get(body_part.lower().strip(), BodyPart.UNKNOWN).value


def _map_result(outcome: str | None, event_type: str | None) -> str:
    if not outcome:
        # Infer from event type when no explicit outcome
        if event_type and event_type.lower() in ("carry", "carries"):
            return ActionResult.SUCCESS.value
        return ActionResult.UNKNOWN.value
    mapped = _RESULT_MAP.get(outcome.lower().strip())
    if mapped:
        return mapped.value
    return ActionResult.UNKNOWN.value


def _make_action_id(match_id: str, index: int) -> str:
    """Stable, unique action ID."""
    raw = f"{match_id}_{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class SPADLNormaliser:
    """
    Converts RawEvent objects into SPADLAction objects.

    Filters out non-action events (half-start, substitutions, etc.)
    and maps vendor-specific values to canonical enums.

    Usage
    -----
    >>> normaliser = SPADLNormaliser()
    >>> actions = normaliser.normalise(raw_events)
    """

    # Action types that represent real on-ball actions (filter out admin events)
    _ACTION_TYPES = {at.value for at in ActionType} - {ActionType.UNKNOWN.value}

    def normalise(self, events: list[RawEvent]) -> list[SPADLAction]:
        """
        Convert raw events to SPADL actions.

        Non-action events (substitutions, half-start, etc.) are dropped.
        Events with missing player or team data are dropped.

        Returns
        -------
        list[SPADLAction]
            Sorted by (period, timestamp).
        """
        actions: list[SPADLAction] = []
        dropped = 0

        for evt in events:
            # Skip events without a player (administrative events)
            if not evt.player_id or evt.player_id in ("None", "nan", ""):
                dropped += 1
                continue

            action_type = _map_type(evt.event_type)

            # Skip purely administrative events
            if action_type == ActionType.UNKNOWN.value:
                dropped += 1
                continue

            action = SPADLAction(
                match_id=evt.match_id,
                action_id=_make_action_id(evt.match_id, evt.index),
                index=evt.index,
                period=evt.period,
                timestamp=_parse_timestamp(evt.timestamp),
                minute=evt.minute,
                second=evt.second,
                team_id=str(evt.team_id or "unknown"),
                team_name=str(evt.team_name or "unknown"),
                player_id=str(evt.player_id),
                player_name=str(evt.player_name or "unknown"),
                position=evt.player_position,
                action_type=action_type,
                body_part=_map_body_part(evt.body_part),
                result=_map_result(evt.outcome, evt.event_type),
                start_x=evt.start_x or 0.0,
                start_y=evt.start_y or 0.0,
                end_x=evt.end_x or evt.start_x or 0.0,
                end_y=evt.end_y or evt.start_y or 0.0,
                duration=evt.duration or 0.0,
                under_pressure=evt.under_pressure,
                xg=evt.xg or 0.0,
            )
            actions.append(action)

        logger.info(
            "SPADL normalisation: %d actions produced, %d events dropped",
            len(actions),
            dropped,
        )

        # Sort by (period, timestamp)
        actions.sort(key=lambda a: (a.period, a.timestamp))
        return actions
