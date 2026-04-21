"""Unit tests for football_ai.utils — geometry and helper functions."""

from __future__ import annotations

import pytest

from football_ai.constants import GOAL_X, GOAL_Y, XT_ZONES_X, XT_ZONES_Y
from football_ai.utils import (
    angle_to_goal,
    clip_probability,
    distance_to_goal,
    safe_divide,
    stable_hash,
    zone_id,
    zone_to_coords,
)


class TestDistanceToGoal:
    def test_at_goal_centre(self) -> None:
        assert distance_to_goal(GOAL_X, GOAL_Y) == pytest.approx(0.0)

    def test_at_centre_spot(self) -> None:
        # Centre spot is at (60, 40); goal at (120, 40)
        d = distance_to_goal(60.0, 40.0)
        assert d == pytest.approx(60.0)

    def test_positive_distance(self) -> None:
        assert distance_to_goal(0.0, 0.0) > 0


class TestAngleToGoal:
    def test_behind_goal_line_returns_zero(self) -> None:
        assert angle_to_goal(121.0, 40.0) == pytest.approx(0.0)

    def test_angle_positive_from_front(self) -> None:
        angle = angle_to_goal(110.0, 40.0)
        assert angle > 0

    def test_angle_decreases_with_distance(self) -> None:
        close = angle_to_goal(110.0, 40.0)
        far = angle_to_goal(60.0, 40.0)
        assert close > far


class TestZoneId:
    def test_bottom_left_corner(self) -> None:
        z = zone_id(0.0, 0.0)
        assert z == 0

    def test_top_right_corner(self) -> None:
        z = zone_id(119.9, 79.9)
        assert z == XT_ZONES_X * XT_ZONES_Y - 1

    def test_valid_range(self) -> None:
        for x in [0.0, 30.0, 60.0, 90.0, 119.0]:
            for y in [0.0, 20.0, 40.0, 60.0, 79.0]:
                z = zone_id(x, y)
                assert 0 <= z < XT_ZONES_X * XT_ZONES_Y


class TestZoneToCoords:
    def test_round_trip(self) -> None:
        """zone_id → zone_to_coords should recover approximately the same zone."""
        x, y = 75.0, 35.0
        z = zone_id(x, y)
        cx, cy = zone_to_coords(z)
        # The recovered coords should map back to the same zone
        assert zone_id(cx, cy) == z


class TestMathHelpers:
    def test_safe_divide_normal(self) -> None:
        assert safe_divide(10.0, 2.0) == pytest.approx(5.0)

    def test_safe_divide_by_zero(self) -> None:
        assert safe_divide(10.0, 0.0) == pytest.approx(0.0)

    def test_safe_divide_custom_default(self) -> None:
        assert safe_divide(1.0, 0.0, default=-1.0) == pytest.approx(-1.0)

    def test_clip_probability_in_range(self) -> None:
        assert clip_probability(0.5) == pytest.approx(0.5)

    def test_clip_probability_above_one(self) -> None:
        assert clip_probability(1.5) == pytest.approx(1.0)

    def test_clip_probability_below_zero(self) -> None:
        assert clip_probability(-0.3) == pytest.approx(0.0)


class TestStableHash:
    def test_deterministic(self) -> None:
        assert stable_hash("a", "b", 1) == stable_hash("a", "b", 1)

    def test_different_inputs_different_hash(self) -> None:
        assert stable_hash("a", 1) != stable_hash("b", 1)

    def test_length_8(self) -> None:
        assert len(stable_hash("test")) == 8
