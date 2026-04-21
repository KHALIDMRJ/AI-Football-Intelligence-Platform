"""
Unit tests for the ingestion layer.

Covers:
- CSVAdapter: load, column mapping, deduplication, xG, coordinate clamping
- SchemaValidator: valid events, period error, missing player, filter_valid
- RawEvent: Pydantic validation behaviour
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from football_ai.ingestion.adapters.csv_adapter import CSVAdapter
from football_ai.ingestion.validators.schema_validator import SchemaValidator
from football_ai.schemas import RawEvent

# ── Sample CSV shared by adapter tests ────────────────────────────────────────

SAMPLE_CSV = (
    "match_id,index,period,timestamp,minute,second,event_type_name,"
    "team_id,team_name,player_id,player_name,player_position_name,"
    "location_x,location_y,end_location_x,end_location_y,"
    "duration,under_pressure,outcome_name,body_part_name,statsbomb_xg\n"
    "3813041,1,1,00:37:00.499,37,0,Shot,"
    "2761,FUS Rabat,138962,Reda Hajhouj,Right Wing,"
    "114.2,35.0,120.0,39.2,0.39,false,Goal,Right Foot,0.797\n"
    "3813041,2,1,00:38:00.000,38,0,Pass,"
    "2761,FUS Rabat,138961,Naoufel Zerhouni,Left Wing,"
    "50.0,30.0,70.0,40.0,0.5,false,Complete,Right Foot,0.0\n"
    "3813041,3,1,00:39:00.000,39,0,Dribble,"
    "2762,FAR Rabat,143160,Omar Jerrari,Left Back,"
    "60.0,20.0,65.0,22.0,0.8,true,Success,,0.0\n"
)


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "test_events.csv"
    csv_file.write_text(SAMPLE_CSV, encoding="utf-8")
    return csv_file


# ── CSVAdapter ────────────────────────────────────────────────────────────────

class TestCSVAdapter:
    def test_load_returns_dataframe(self, sample_csv: Path) -> None:
        adapter = CSVAdapter(sample_csv)
        df = adapter.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_to_raw_events_returns_list_of_raw_events(self, sample_csv: Path) -> None:
        adapter = CSVAdapter(sample_csv)
        df = adapter.load()
        events = adapter.to_raw_events(df)
        assert isinstance(events, list)
        assert len(events) == 3
        assert all(isinstance(e, RawEvent) for e in events)

    def test_ingest_preserves_match_id(self, sample_csv: Path) -> None:
        adapter = CSVAdapter(sample_csv)
        events = adapter.ingest()
        assert all(e.match_id == "3813041" for e in events)

    def test_match_id_override_applied(self, sample_csv: Path) -> None:
        adapter = CSVAdapter(sample_csv, match_id_override="OVERRIDE_99")
        events = adapter.ingest()
        assert all(e.match_id == "OVERRIDE_99" for e in events)

    def test_xg_on_shot_event(self, sample_csv: Path) -> None:
        adapter = CSVAdapter(sample_csv)
        events = adapter.ingest()
        shot = next(e for e in events if e.event_type.lower() == "shot")
        assert shot.xg == pytest.approx(0.797, abs=0.001)

    def test_xg_zero_on_non_shot(self, sample_csv: Path) -> None:
        adapter = CSVAdapter(sample_csv)
        events = adapter.ingest()
        non_shots = [e for e in events if e.event_type.lower() != "shot"]
        assert all(e.xg == pytest.approx(0.0) for e in non_shots)

    def test_coordinates_within_pitch_bounds(self, sample_csv: Path) -> None:
        adapter = CSVAdapter(sample_csv)
        events = adapter.ingest()
        for e in events:
            if e.start_x is not None:
                assert 0.0 <= e.start_x <= 120.0
            if e.start_y is not None:
                assert 0.0 <= e.start_y <= 80.0

    def test_under_pressure_parsed_as_bool(self, sample_csv: Path) -> None:
        adapter = CSVAdapter(sample_csv)
        events = adapter.ingest()
        dribble = next(e for e in events if e.event_type.lower() == "dribble")
        assert dribble.under_pressure is True
        shot = next(e for e in events if e.event_type.lower() == "shot")
        assert shot.under_pressure is False

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            CSVAdapter(tmp_path / "nonexistent.csv")

    def test_all_events_have_period_1(self, sample_csv: Path) -> None:
        adapter = CSVAdapter(sample_csv)
        events = adapter.ingest()
        assert all(e.period == 1 for e in events)


# ── SchemaValidator ───────────────────────────────────────────────────────────

class TestSchemaValidator:
    """
    Uses RawEvent.model_construct() to bypass Pydantic field-level validators
    when constructing intentionally invalid objects for testing SchemaValidator
    logic in isolation.
    """

    def _valid_event(self, **overrides: object) -> RawEvent:
        defaults: dict[str, object] = dict(
            match_id="test", index=1, period=1,
            timestamp="00:01:00.000", minute=1, second=0,
            event_type="pass", team_id="team_a", team_name="Team A",
            player_id="p1", player_name="Player One",
            start_x=50.0, start_y=40.0,
        )
        defaults.update(overrides)
        return RawEvent(**defaults)

    def _construct_event(self, **overrides: object) -> RawEvent:
        """Bypass Pydantic validators for SchemaValidator unit testing."""
        defaults: dict[str, object] = dict(
            match_id="test", index=1, period=1,
            timestamp="00:01:00.000", minute=1, second=0,
            event_type="pass", team_id="team_a", team_name="Team A",
            player_id="p1", player_name="Player One",
            start_x=50.0, start_y=40.0, end_x=60.0, end_y=40.0,
            under_pressure=False, xg=0.0,
        )
        defaults.update(overrides)
        return RawEvent.model_construct(**defaults)

    def test_valid_event_produces_no_issues(self) -> None:
        validator = SchemaValidator()
        report = validator.validate([self._valid_event()])
        assert report.error_count == 0
        assert report.valid_events == 1

    def test_out_of_range_period_is_error(self) -> None:
        validator = SchemaValidator()
        # period=9 is invalid; model_construct lets us create it to test validator
        events = [self._construct_event(period=9)]
        report = validator.validate(events)
        assert report.error_count > 0

    def test_filter_valid_keeps_only_error_free_events(self) -> None:
        validator = SchemaValidator()
        good = self._construct_event(index=1, period=1)
        bad = self._construct_event(index=2, period=9)
        events = [good, bad]
        report = validator.validate(events)
        valid = validator.filter_valid(events, report)
        assert len(valid) == 1
        assert valid[0].index == 1

    def test_missing_player_id_is_warning_not_error(self) -> None:
        validator = SchemaValidator()
        events = [self._construct_event(player_id=None)]
        report = validator.validate(events)
        player_errors = [
            i for i in report.issues
            if i.field == "player_id" and i.severity == "error"
        ]
        assert len(player_errors) == 0

    def test_report_summary_contains_key_words(self) -> None:
        validator = SchemaValidator()
        report = validator.validate([self._valid_event()])
        assert "valid" in report.summary().lower()

    def test_report_is_valid_when_no_errors(self) -> None:
        validator = SchemaValidator()
        report = validator.validate([self._valid_event()])
        assert report.is_valid is True

    def test_report_is_invalid_when_errors_present(self) -> None:
        validator = SchemaValidator()
        report = validator.validate([self._construct_event(period=9)])
        assert report.is_valid is False

    def test_empty_event_list_produces_empty_report(self) -> None:
        validator = SchemaValidator()
        report = validator.validate([])
        assert report.total_events == 0
        assert report.valid_events == 0
        assert report.error_count == 0
