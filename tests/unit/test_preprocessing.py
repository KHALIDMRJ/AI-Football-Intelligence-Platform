"""Unit tests for preprocessing — SPADL normaliser, zone mapper, possession detector."""

from __future__ import annotations

import pandas as pd

from football_ai.constants import Cols
from football_ai.preprocessing.game_state.builder import GameStateBuilder
from football_ai.preprocessing.pitch.zone_mapper import ZoneMapper
from football_ai.preprocessing.possession.detector import PossessionDetector
from football_ai.preprocessing.spadl.normaliser import SPADLNormaliser
from football_ai.schemas import RawEvent


def _make_raw_event(**kwargs) -> RawEvent:
    defaults = dict(
        match_id="test",
        index=1,
        period=1,
        timestamp="00:01:00.000",
        minute=1,
        second=0,
        event_type="pass",
        team_id="team_a",
        team_name="Team A",
        player_id="p1",
        player_name="Player One",
        start_x=50.0,
        start_y=40.0,
        end_x=60.0,
        end_y=40.0,
    )
    defaults.update(kwargs)
    return RawEvent(**defaults)


# ── SPADL normaliser ──────────────────────────────────────────────────────────

class TestSPADLNormaliser:
    def test_pass_is_normalised(self) -> None:
        normaliser = SPADLNormaliser()
        events = [_make_raw_event(event_type="Pass")]
        actions = normaliser.normalise(events)
        assert len(actions) == 1
        assert actions[0].action_type == "pass"

    def test_shot_normalised(self) -> None:
        normaliser = SPADLNormaliser()
        events = [_make_raw_event(event_type="Shot", outcome="Goal", xg=0.5)]
        actions = normaliser.normalise(events)
        assert len(actions) == 1
        assert actions[0].action_type == "shot"
        assert actions[0].result == "goal"

    def test_admin_events_dropped(self) -> None:
        normaliser = SPADLNormaliser()
        events = [
            _make_raw_event(event_type="Half Start", player_id=None),
            _make_raw_event(event_type="Substitution", player_id=None),
            _make_raw_event(event_type="Pass"),
        ]
        actions = normaliser.normalise(events)
        assert len(actions) == 1

    def test_sorted_by_period_timestamp(self) -> None:
        normaliser = SPADLNormaliser()
        events = [
            _make_raw_event(index=3, period=2, timestamp="00:05:00.000"),
            _make_raw_event(index=1, period=1, timestamp="00:01:00.000"),
            _make_raw_event(index=2, period=1, timestamp="00:03:00.000"),
        ]
        actions = normaliser.normalise(events)
        timestamps = [a.timestamp for a in actions]
        assert timestamps == sorted(timestamps)

    def test_action_id_stable(self) -> None:
        normaliser = SPADLNormaliser()
        events = [_make_raw_event(index=42)]
        a1 = normaliser.normalise(events)[0].action_id
        a2 = normaliser.normalise(events)[0].action_id
        assert a1 == a2


# ── Zone mapper ───────────────────────────────────────────────────────────────

class TestZoneMapper:
    def test_adds_zone_id_column(self) -> None:
        mapper = ZoneMapper()
        df = pd.DataFrame({
            "start_x": [60.0, 100.0],
            "start_y": [40.0, 20.0],
        })
        result = mapper.add_zones(df)
        assert Cols.ZONE_ID in result.columns
        assert result[Cols.ZONE_ID].notna().all()

    def test_zone_id_in_valid_range(self) -> None:
        mapper = ZoneMapper()
        df = pd.DataFrame({
            "start_x": [0.0, 60.0, 119.9],
            "start_y": [0.0, 40.0, 79.9],
        })
        result = mapper.add_zones(df)
        total = mapper.total_zones()
        assert (result[Cols.ZONE_ID] >= 0).all()
        assert (result[Cols.ZONE_ID] < total).all()

    def test_zone_centre_roundtrip(self) -> None:
        mapper = ZoneMapper()
        for zone in [0, 50, 100, 215]:
            x, y = mapper.zone_centre(zone)
            assert 0 <= x <= 120
            assert 0 <= y <= 80


# ── Possession detector ───────────────────────────────────────────────────────

class TestPossessionDetector:
    def _make_df(self, records: list[dict]) -> pd.DataFrame:
        defaults = dict(
            match_id="test",
            team_id="team_a",
            timestamp=0.0,
            action_type="pass",
        )
        rows = [{**defaults, **r} for r in records]
        return pd.DataFrame(rows)

    def test_adds_required_columns(self) -> None:
        detector = PossessionDetector()
        df = self._make_df([
            {"timestamp": 0.0},
            {"timestamp": 1.0},
        ])
        result = detector.detect(df)
        for col in [Cols.POSSESSION_ID, Cols.CHAIN_ID, Cols.CHAIN_INDEX]:
            assert col in result.columns

    def test_team_change_creates_new_possession(self) -> None:
        detector = PossessionDetector(max_gap_seconds=5.0)
        df = self._make_df([
            {"timestamp": 0.0, "team_id": "team_a"},
            {"timestamp": 1.0, "team_id": "team_b"},  # team change
        ])
        result = detector.detect(df)
        assert result[Cols.POSSESSION_ID].iloc[0] != result[Cols.POSSESSION_ID].iloc[1]

    def test_gap_creates_new_possession(self) -> None:
        detector = PossessionDetector(max_gap_seconds=5.0)
        df = self._make_df([
            {"timestamp": 0.0},
            {"timestamp": 100.0},  # 100s gap
        ])
        result = detector.detect(df)
        assert result[Cols.POSSESSION_ID].iloc[0] != result[Cols.POSSESSION_ID].iloc[1]

    def test_same_team_continuous_is_one_possession(self) -> None:
        detector = PossessionDetector(max_gap_seconds=5.0)
        df = self._make_df([
            {"timestamp": 0.0, "team_id": "team_a"},
            {"timestamp": 1.0, "team_id": "team_a"},
            {"timestamp": 2.0, "team_id": "team_a"},
        ])
        result = detector.detect(df)
        assert result[Cols.POSSESSION_ID].nunique() == 1


# ── Game state builder ────────────────────────────────────────────────────────

class TestGameStateBuilder:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"match_id": "t", "team_id": "h", "action_type": "pass",
             "result": "success", "possession_team_id": "h", "timestamp": 0.0},
            {"match_id": "t", "team_id": "h", "action_type": "shot",
             "result": "goal", "possession_team_id": "h", "timestamp": 1.0},
            {"match_id": "t", "team_id": "a", "action_type": "pass",
             "result": "success", "possession_team_id": "a", "timestamp": 2.0},
        ])

    def test_adds_label_columns(self) -> None:
        builder = GameStateBuilder(k_actions=10)
        df = self._make_df()
        result = builder.build(df, home_team_id="h")
        assert Cols.LABEL_SCORES in result.columns
        assert Cols.LABEL_CONCEDES in result.columns

    def test_scores_label_before_goal(self) -> None:
        builder = GameStateBuilder(k_actions=10)
        df = self._make_df()
        result = builder.build(df, home_team_id="h")
        # First action (index 0) should have label_scores=1 because goal is within k
        assert result[Cols.LABEL_SCORES].iloc[0] == 1

    def test_score_tracking(self) -> None:
        builder = GameStateBuilder(k_actions=10)
        df = self._make_df()
        result = builder.build(df, home_team_id="h")
        # After goal at index 1, score_home should be 1
        assert result["score_home"].iloc[2] == 1
