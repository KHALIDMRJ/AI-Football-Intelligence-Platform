"""
Unit tests for Phase 7 — Tactical Intelligence.

Covers:
- PlayerRanker: all ranking methods, top_n, tier assignment, efficiency
- WeaknessDetector: opponent zone aggregation, risk levels, no-opponent guard
- FormationAnalyser: cluster count, formation string format, low-data fallback
- PressureMap: three map outputs, high_risk_zones intersection
- ReportBuilder: MatchReport, PlayerReport, ActionSummary, TeamReport
- TacticalPipeline: end-to-end run with temp store, JSON artefact written
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_ai.constants import ActionResult, ActionType, Cols
from football_ai.tactical.formation_analyser import FormationAnalyser
from football_ai.tactical.pipeline import TacticalPipeline, TacticalResult
from football_ai.tactical.player_ranker import PlayerRanker
from football_ai.tactical.pressure_map import PressureMap
from football_ai.tactical.report_builder import ReportBuilder
from football_ai.tactical.weakness_detector import WeaknessDetector

# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_vaep_df(n: int = 60) -> pd.DataFrame:
    """
    Realistic VAEP-scored DataFrame with 2 teams, 6 players,
    all required columns for tactical modules.
    """
    rng = np.random.default_rng(42)
    n_half = n // 2

    team_ids   = (["team_a"] * n_half) + (["team_b"] * n_half)
    team_names = (["FUS Rabat"] * n_half) + (["FAR Rabat"] * n_half)
    player_ids = [f"p{i % 3}" for i in range(n_half)] + \
                 [f"p{3 + i % 3}" for i in range(n_half)]
    player_names = [f"Player {i % 3}" for i in range(n_half)] + \
                   [f"Player {3 + i % 3}" for i in range(n_half)]

    action_pool = [ActionType.PASS.value, ActionType.SHOT.value,
                   ActionType.CARRY.value, ActionType.DRIBBLE.value]

    df = pd.DataFrame({
        Cols.MATCH_ID:           ["3813041"] * n,
        Cols.ACTION_ID:          [f"a{i}" for i in range(n)],
        Cols.INDEX:              list(range(n)),
        Cols.PERIOD:             [1] * n,
        Cols.TIMESTAMP:          np.linspace(0.0, 5400.0, n),
        Cols.MINUTE:             (np.linspace(0, 90, n)).astype(int),
        Cols.SECOND:             [0] * n,
        Cols.TEAM_ID:            team_ids,
        Cols.TEAM_NAME:          team_names,
        Cols.PLAYER_ID:          player_ids,
        Cols.PLAYER_NAME:        player_names,
        Cols.ACTION_TYPE:        rng.choice(action_pool, n),
        Cols.RESULT:             rng.choice(
            [ActionResult.SUCCESS.value, ActionResult.FAIL.value], n
        ),
        Cols.START_X:            rng.uniform(0, 120, n),
        Cols.START_Y:            rng.uniform(0, 80, n),
        Cols.END_X:              rng.uniform(0, 120, n),
        Cols.END_Y:              rng.uniform(0, 80, n),
        Cols.UNDER_PRESSURE:     rng.integers(0, 2, n).astype(bool),
        Cols.POSSESSION_ID:      np.arange(n) // 5,
        Cols.POSSESSION_TEAM_ID: team_ids,
        Cols.ZONE_ID:            rng.integers(0, 216, n),
        Cols.XG:                 rng.uniform(0, 0.5, n) * (rng.random(n) < 0.1),
        Cols.VAEP_VALUE:         rng.uniform(-0.5, 0.5, n),
        Cols.VAEP_OFFENSIVE:     rng.uniform(0, 0.4, n),
        Cols.VAEP_DEFENSIVE:     rng.uniform(0, 0.2, n),
        Cols.XT_DELTA:           rng.uniform(-0.1, 0.1, n),
        "delta_p_scores":        rng.uniform(-0.1, 0.1, n),
        "delta_p_concedes":      rng.uniform(-0.1, 0.1, n),
        "state_value":           rng.uniform(-0.5, 0.5, n),
        Cols.P_SCORES:           rng.uniform(0, 0.3, n),
        Cols.P_CONCEDES:         rng.uniform(0, 0.1, n),
        "score_diff":            [0] * n,
    })
    return df


def _make_player_summary(vaep_df: pd.DataFrame) -> pd.DataFrame:
    from football_ai.vaep.aggregator import VAEPAggregator
    return VAEPAggregator(min_actions=1).player_summary(vaep_df)


def _make_team_summary(vaep_df: pd.DataFrame) -> pd.DataFrame:
    from football_ai.vaep.aggregator import VAEPAggregator
    return VAEPAggregator(min_actions=1).team_summary(vaep_df)


def _make_store(tmp_path: Path):
    from football_ai.ingestion.storage.parquet_store import ParquetStore
    return ParquetStore(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
    )


# ── PlayerRanker ──────────────────────────────────────────────────────────────

class TestPlayerRanker:
    def test_rank_overall_returns_dataframe(self) -> None:
        ranker = PlayerRanker(min_actions=1)
        result = ranker.rank_overall(_make_vaep_df())
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_rank_overall_sorted_descending(self) -> None:
        ranker = PlayerRanker(min_actions=1)
        result = ranker.rank_overall(_make_vaep_df())
        vals = result["vaep_total"].values
        assert list(vals) == sorted(vals, reverse=True)

    def test_rank_offensive_sorted_by_vaep_offensive(self) -> None:
        ranker = PlayerRanker(min_actions=1)
        result = ranker.rank_offensive(_make_vaep_df())
        vals = result["vaep_offensive"].values
        assert list(vals) == sorted(vals, reverse=True)

    def test_rank_defensive_sorted_by_vaep_defensive(self) -> None:
        ranker = PlayerRanker(min_actions=1)
        result = ranker.rank_defensive(_make_vaep_df())
        vals = result["vaep_defensive"].values
        assert list(vals) == sorted(vals, reverse=True)

    def test_rank_per_90_excludes_insufficient_minutes(self) -> None:
        ranker = PlayerRanker(min_actions=1, min_minutes=9999.0)
        result = ranker.rank_per_90(_make_vaep_df())
        assert len(result) == 0

    def test_rank_efficiency_has_vaep_per_action(self) -> None:
        ranker = PlayerRanker(min_actions=1)
        result = ranker.rank_efficiency(_make_vaep_df())
        assert "vaep_per_action" in result.columns

    def test_top_n_returns_at_most_n_rows(self) -> None:
        ranker = PlayerRanker(min_actions=1)
        result = ranker.top_n(_make_vaep_df(), n=3)
        assert len(result) <= 3

    def test_top_n_raises_on_invalid_metric(self) -> None:
        ranker = PlayerRanker(min_actions=1)
        with pytest.raises(ValueError, match="Unknown metric"):
            ranker.top_n(_make_vaep_df(), metric="nonexistent")

    def test_tier_column_present_and_valid(self) -> None:
        ranker = PlayerRanker(min_actions=1)
        result = ranker.rank_overall(_make_vaep_df())
        valid_tiers = {"elite", "strong", "average", "below_average"}
        assert set(result["tier"].unique()).issubset(valid_tiers)

    def test_all_rankings_returns_five_keys(self) -> None:
        ranker = PlayerRanker(min_actions=1)
        rankings = ranker.all_rankings(_make_vaep_df())
        assert set(rankings.keys()) == {
            "overall", "offensive", "defensive", "per_90", "efficiency"
        }

    def test_min_actions_filters_players(self) -> None:
        ranker_strict = PlayerRanker(min_actions=1000)
        result = ranker_strict.rank_overall(_make_vaep_df())
        assert len(result) == 0

    def test_index_is_one_based_rank(self) -> None:
        ranker = PlayerRanker(min_actions=1)
        result = ranker.rank_overall(_make_vaep_df())
        assert result.index[0] == 1


# ── WeaknessDetector ──────────────────────────────────────────────────────────

class TestWeaknessDetector:
    def test_returns_dataframe(self) -> None:
        detector = WeaknessDetector(min_actions_per_zone=1)
        result = detector.detect(_make_vaep_df(), team_id="team_a")
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self) -> None:
        detector = WeaknessDetector(min_actions_per_zone=1)
        result = detector.detect(_make_vaep_df(), team_id="team_a")
        for col in ["zone_id", "zone_x", "zone_y",
                    "mean_opponent_vaep", "action_count", "risk_level"]:
            assert col in result.columns, f"Missing: {col}"

    def test_sorted_by_vaep_descending(self) -> None:
        detector = WeaknessDetector(min_actions_per_zone=1)
        result = detector.detect(_make_vaep_df(), team_id="team_a")
        vals = result["mean_opponent_vaep"].values
        assert list(vals) == sorted(vals, reverse=True)

    def test_risk_levels_are_valid(self) -> None:
        detector = WeaknessDetector(min_actions_per_zone=1)
        result = detector.detect(_make_vaep_df(), team_id="team_a")
        valid = {"critical", "high", "medium", "low"}
        assert set(result["risk_level"].unique()).issubset(valid)

    def test_no_opponent_returns_empty(self) -> None:
        detector = WeaknessDetector(min_actions_per_zone=1)
        df = _make_vaep_df()
        # Make all actions belong to team_a
        df[Cols.POSSESSION_TEAM_ID] = "team_a"
        result = detector.detect(df, team_id="team_a")
        assert result.empty

    def test_raises_on_missing_vaep_column(self) -> None:
        detector = WeaknessDetector()
        df = _make_vaep_df().drop(columns=[Cols.VAEP_VALUE])
        with pytest.raises(ValueError, match="vaep_value"):
            detector.detect(df, team_id="team_a")

    def test_detect_all_teams_returns_dict(self) -> None:
        detector = WeaknessDetector(min_actions_per_zone=1)
        result = detector.detect_all_teams(_make_vaep_df())
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_zone_coords_in_pitch_bounds(self) -> None:
        detector = WeaknessDetector(min_actions_per_zone=1)
        result = detector.detect(_make_vaep_df(), team_id="team_a")
        if not result.empty:
            assert (result["zone_x"] >= 0).all()
            assert (result["zone_x"] <= 120).all()
            assert (result["zone_y"] >= 0).all()
            assert (result["zone_y"] <= 80).all()


# ── FormationAnalyser ─────────────────────────────────────────────────────────

class TestFormationAnalyser:
    def test_returns_formation_result(self) -> None:
        from football_ai.tactical.formation_analyser import FormationResult
        analyser = FormationAnalyser(n_clusters=5)
        result = analyser.analyse(_make_vaep_df(), team_id="team_a")
        assert isinstance(result, FormationResult)

    def test_formation_str_has_dash_format(self) -> None:
        analyser = FormationAnalyser(n_clusters=5)
        result = analyser.analyse(_make_vaep_df(), team_id="team_a")
        if result.formation_str != "unknown":
            parts = result.formation_str.split("-")
            assert len(parts) == 3
            assert all(p.isdigit() for p in parts)

    def test_confidence_in_unit_range(self) -> None:
        analyser = FormationAnalyser(n_clusters=5)
        result = analyser.analyse(_make_vaep_df(), team_id="team_a")
        assert 0.0 <= result.confidence <= 1.0

    def test_centroids_dataframe_has_columns(self) -> None:
        analyser = FormationAnalyser(n_clusters=5)
        result = analyser.analyse(_make_vaep_df(), team_id="team_a")
        if not result.centroids.empty:
            for col in ["cluster_id", "x", "y", "depth", "width"]:
                assert col in result.centroids.columns

    def test_low_data_returns_unknown(self) -> None:
        analyser = FormationAnalyser()
        df = _make_vaep_df(n=4)   # only 2 actions per team < _MIN_ACTIONS
        result = analyser.analyse(df, team_id="team_a")
        assert result.formation_str == "unknown"

    def test_analyse_all_teams_has_both_teams(self) -> None:
        analyser = FormationAnalyser(n_clusters=5)
        results = analyser.analyse_all_teams(_make_vaep_df())
        assert "team_a" in results
        assert "team_b" in results

    def test_team_name_propagated(self) -> None:
        analyser = FormationAnalyser(n_clusters=5)
        result = analyser.analyse(_make_vaep_df(), team_id="team_a")
        assert result.team_name == "FUS Rabat"


# ── PressureMap ───────────────────────────────────────────────────────────────

class TestPressureMap:
    def test_build_returns_pressure_map_result(self) -> None:
        from football_ai.tactical.pressure_map import PressureMapResult
        pm = PressureMap(min_zone_actions=1)
        result = pm.build(_make_vaep_df(), team_id="team_a")
        assert isinstance(result, PressureMapResult)

    def test_pressure_intensity_has_required_cols(self) -> None:
        pm = PressureMap(min_zone_actions=1)
        result = pm.build(_make_vaep_df(), team_id="team_a")
        for col in ["zone_id", "zone_x", "zone_y", "intensity"]:
            assert col in result.pressure_intensity.columns

    def test_opponent_threat_has_required_cols(self) -> None:
        pm = PressureMap(min_zone_actions=1)
        result = pm.build(_make_vaep_df(), team_id="team_a")
        for col in ["zone_id", "zone_x", "zone_y", "threat"]:
            assert col in result.opponent_threat.columns

    def test_action_density_has_required_cols(self) -> None:
        pm = PressureMap(min_zone_actions=1)
        result = pm.build(_make_vaep_df(), team_id="team_a")
        for col in ["zone_id", "zone_x", "zone_y", "density"]:
            assert col in result.action_density.columns

    def test_intensity_in_unit_range(self) -> None:
        pm = PressureMap(min_zone_actions=1)
        result = pm.build(_make_vaep_df(), team_id="team_a")
        if not result.pressure_intensity.empty:
            assert (result.pressure_intensity["intensity"] >= 0).all()
            assert (result.pressure_intensity["intensity"] <= 1).all()

    def test_density_in_unit_range(self) -> None:
        pm = PressureMap(min_zone_actions=1)
        result = pm.build(_make_vaep_df(), team_id="team_a")
        if not result.action_density.empty:
            assert (result.action_density["density"] >= 0).all()
            assert (result.action_density["density"] <= 1).all()

    def test_threat_non_negative(self) -> None:
        pm = PressureMap(min_zone_actions=1)
        result = pm.build(_make_vaep_df(), team_id="team_a")
        if not result.opponent_threat.empty:
            assert (result.opponent_threat["threat"] >= 0).all()

    def test_build_all_teams_returns_both(self) -> None:
        pm = PressureMap(min_zone_actions=1)
        results = pm.build_all_teams(_make_vaep_df())
        assert "team_a" in results
        assert "team_b" in results


# ── ReportBuilder ─────────────────────────────────────────────────────────────

class TestReportBuilder:
    def test_build_match_report_returns_match_report(self) -> None:
        from football_ai.tactical.report_builder import MatchReport
        df   = _make_vaep_df()
        ps   = _make_player_summary(df)
        ts   = _make_team_summary(df)
        rb   = ReportBuilder()
        report = rb.build_match_report(df, ps, ts, "team_a", "team_b")
        assert isinstance(report, MatchReport)

    def test_match_report_has_correct_team_ids(self) -> None:
        df   = _make_vaep_df()
        rb   = ReportBuilder()
        report = rb.build_match_report(
            df, _make_player_summary(df), _make_team_summary(df),
            "team_a", "team_b",
        )
        assert report.home_team_id == "team_a"
        assert report.away_team_id == "team_b"

    def test_match_report_to_dict_is_serialisable(self) -> None:
        import json
        df   = _make_vaep_df()
        rb   = ReportBuilder()
        report = rb.build_match_report(
            df, _make_player_summary(df), _make_team_summary(df),
            "team_a", "team_b",
        )
        d = report.to_dict()
        # Should not raise
        json.dumps(d, default=str)

    def test_most_valuable_actions_present(self) -> None:
        df   = _make_vaep_df()
        rb   = ReportBuilder(top_actions_n=3)
        report = rb.build_match_report(
            df, _make_player_summary(df), _make_team_summary(df),
            "team_a", "team_b",
        )
        assert len(report.most_valuable_actions) <= 3

    def test_build_player_report_returns_player_report(self) -> None:
        from football_ai.tactical.report_builder import PlayerReport
        df   = _make_vaep_df()
        ps   = _make_player_summary(df)
        rb   = ReportBuilder()
        player_id = str(ps[Cols.PLAYER_ID].iloc[0])
        report = rb.build_player_report(df, ps, player_id)
        assert isinstance(report, PlayerReport)

    def test_build_player_report_returns_none_for_unknown(self) -> None:
        df   = _make_vaep_df()
        ps   = _make_player_summary(df)
        rb   = ReportBuilder()
        report = rb.build_player_report(df, ps, "nonexistent_player_99")
        assert report is None

    def test_player_report_vaep_matches_summary(self) -> None:
        df   = _make_vaep_df()
        ps   = _make_player_summary(df)
        rb   = ReportBuilder()
        player_id = str(ps[Cols.PLAYER_ID].iloc[0])
        report = rb.build_player_report(df, ps, player_id)
        expected = float(ps[ps[Cols.PLAYER_ID].astype(str) == player_id]["vaep_total"].iloc[0])
        assert report.vaep_total == pytest.approx(expected, abs=1e-3)

    def test_home_formation_in_match_report(self) -> None:
        from football_ai.tactical.formation_analyser import FormationResult
        df   = _make_vaep_df()
        rb   = ReportBuilder()
        formations = {
            "team_a": FormationResult(
                team_id="team_a", team_name="FUS", formation_str="4-3-3"
            ),
            "team_b": FormationResult(
                team_id="team_b", team_name="FAR", formation_str="4-4-2"
            ),
        }
        report = rb.build_match_report(
            df, _make_player_summary(df), _make_team_summary(df),
            "team_a", "team_b",
            formations=formations,
        )
        assert report.home_formation == "4-3-3"
        assert report.away_formation == "4-4-2"


# ── TacticalPipeline ──────────────────────────────────────────────────────────

class TestTacticalPipeline:
    def _save_vaep_parquets(
        self,
        store,
        match_id: str,
        df: pd.DataFrame,
    ) -> None:
        """Save the three Parquet files that TacticalPipeline expects."""
        from football_ai.utils import save_parquet
        save_parquet(df, store.processed_dir / f"match_{match_id}_vaep.parquet")
        ps = _make_player_summary(df)
        ts = _make_team_summary(df)
        save_parquet(ps, store.processed_dir / f"match_{match_id}_player_summary.parquet")
        save_parquet(ts, store.processed_dir / f"match_{match_id}_team_summary.parquet")

    def test_run_returns_tactical_result(self, tmp_path: Path) -> None:
        store    = _make_store(tmp_path)
        match_id = "test_tactical"
        df       = _make_vaep_df()
        self._save_vaep_parquets(store, match_id, df)

        pipeline = TacticalPipeline(store=store)
        result   = pipeline.run(match_id, force=True)
        assert isinstance(result, TacticalResult)
        assert result.match_id == match_id

    def test_run_has_match_report(self, tmp_path: Path) -> None:
        store    = _make_store(tmp_path)
        match_id = "test_report"
        self._save_vaep_parquets(store, match_id, _make_vaep_df())

        result = TacticalPipeline(store=store).run(match_id, force=True)
        assert result.match_report is not None

    def test_run_has_weaknesses(self, tmp_path: Path) -> None:
        store    = _make_store(tmp_path)
        match_id = "test_weak"
        self._save_vaep_parquets(store, match_id, _make_vaep_df())

        result = TacticalPipeline(store=store).run(match_id, force=True)
        assert isinstance(result.weaknesses, dict)
        assert len(result.weaknesses) > 0

    def test_run_has_formations(self, tmp_path: Path) -> None:
        store    = _make_store(tmp_path)
        match_id = "test_form"
        self._save_vaep_parquets(store, match_id, _make_vaep_df())

        result = TacticalPipeline(store=store).run(match_id, force=True)
        assert isinstance(result.formations, dict)
        assert len(result.formations) == 2

    def test_run_has_pressure_maps(self, tmp_path: Path) -> None:
        store    = _make_store(tmp_path)
        match_id = "test_pressure"
        self._save_vaep_parquets(store, match_id, _make_vaep_df())

        result = TacticalPipeline(store=store).run(match_id, force=True)
        assert isinstance(result.pressure_maps, dict)
        assert len(result.pressure_maps) == 2

    def test_json_report_written_to_disk(self, tmp_path: Path) -> None:
        store    = _make_store(tmp_path)
        match_id = "test_json"
        self._save_vaep_parquets(store, match_id, _make_vaep_df())

        TacticalPipeline(store=store).run(match_id, force=True)
        json_path = store.processed_dir / f"match_{match_id}_tactical_report.json"
        assert json_path.exists()

    def test_json_report_is_valid_json(self, tmp_path: Path) -> None:
        import json
        store    = _make_store(tmp_path)
        match_id = "test_valid_json"
        self._save_vaep_parquets(store, match_id, _make_vaep_df())

        TacticalPipeline(store=store).run(match_id, force=True)
        json_path = store.processed_dir / f"match_{match_id}_tactical_report.json"
        with open(json_path) as fh:
            data = json.load(fh)
        assert "home_team_name" in data

    def test_empty_vaep_returns_empty_result(self, tmp_path: Path) -> None:
        store    = _make_store(tmp_path)
        match_id = "test_empty"
        # Do NOT save any Parquet files → VAEP load returns empty DF
        result = TacticalPipeline(store=store).run(match_id, force=True)
        assert result.match_report is None

    def test_rankings_has_all_five_keys(self, tmp_path: Path) -> None:
        store    = _make_store(tmp_path)
        match_id = "test_rankings"
        self._save_vaep_parquets(store, match_id, _make_vaep_df())

        result = TacticalPipeline(store=store).run(match_id, force=True)
        assert set(result.rankings.keys()) == {
            "overall", "offensive", "defensive", "per_90", "efficiency"
        }
