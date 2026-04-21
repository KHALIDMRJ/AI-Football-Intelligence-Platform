"""
Unit tests for Phase 6 — VAEP Engine.

Covers:
- StateValueComputer: V(Si) range, column presence, empty input guard
- ActionValueComputer: V(ai) = V(Si) - V(Si-1), possession boundary reset,
  offensive/defensive decomposition, edge cases
- VAEPAggregator: player summary columns, team summary, vaep_per_90,
  min_actions filtering, top_players
- VAEPPipeline: end-to-end run with mocked models, output shapes,
  Parquet persistence
- VAEPResult: repr, total_actions, n_players
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from football_ai.constants import ActionResult, ActionType, Cols
from football_ai.vaep.action_value import ActionValueComputer
from football_ai.vaep.aggregator import VAEPAggregator
from football_ai.vaep.pipeline import VAEPPipeline, VAEPResult
from football_ai.vaep.state_value import StateValueComputer

# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_model_mock(proba_values: list[float]) -> Any:
    """Return a mock model whose score_actions() returns the given values."""
    mock = MagicMock()
    mock.score_actions.return_value = pd.Series(proba_values)
    return mock


def _make_feature_df(
    n: int = 6,
    poss_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Minimal feature DataFrame with all columns the VAEP pipeline expects."""
    if poss_ids is None:
        poss_ids = [0] * n
    return pd.DataFrame({
        Cols.MATCH_ID:           ["m1"] * n,
        Cols.ACTION_ID:          [f"a{i}" for i in range(n)],
        Cols.INDEX:              list(range(n)),
        Cols.PERIOD:             [1] * n,
        Cols.TIMESTAMP:          [float(i * 5) for i in range(n)],
        Cols.MINUTE:             list(range(n)),
        Cols.TEAM_ID:            ["team_a"] * n,
        Cols.TEAM_NAME:          ["Team A"] * n,
        Cols.PLAYER_ID:          [f"p{i % 3}" for i in range(n)],
        Cols.PLAYER_NAME:        [f"Player {i % 3}" for i in range(n)],
        Cols.ACTION_TYPE:        [ActionType.PASS.value] * (n - 1) + [ActionType.SHOT.value],
        Cols.RESULT:             [ActionResult.SUCCESS.value] * (n - 1) + [ActionResult.GOAL.value],
        Cols.START_X:            [30.0 + i * 10 for i in range(n)],
        Cols.START_Y:            [40.0] * n,
        Cols.END_X:              [40.0 + i * 10 for i in range(n)],
        Cols.END_Y:              [40.0] * n,
        Cols.POSSESSION_ID:      poss_ids,
        Cols.POSSESSION_TEAM_ID: ["team_a"] * n,
        Cols.XG:                 [0.0] * (n - 1) + [0.35],
        # minimal f_* columns so score_actions can index them
        "f_sp_dist_to_goal_start": [float(i) for i in range(n)],
    })


def _make_state_valued_df(n: int = 6) -> pd.DataFrame:
    """Feature DF that already has p_scores, p_concedes, state_value columns."""
    df = _make_feature_df(n)
    df[Cols.P_SCORES]   = [0.05, 0.10, 0.08, 0.20, 0.30, 0.80][:n]
    df[Cols.P_CONCEDES] = [0.02, 0.02, 0.03, 0.02, 0.01, 0.00][:n]
    df["state_value"]   = df[Cols.P_SCORES] - df[Cols.P_CONCEDES]
    return df


def _make_vaep_df(n: int = 6) -> pd.DataFrame:
    """DF with full VAEP columns for aggregation tests."""
    df = _make_state_valued_df(n)
    df[Cols.VAEP_VALUE]     = [0.0, 0.03, -0.01, 0.08, 0.10, 0.50][:n]
    df[Cols.VAEP_OFFENSIVE] = [0.0, 0.03, 0.00,  0.08, 0.10, 0.50][:n]
    df[Cols.VAEP_DEFENSIVE] = [0.0, 0.00, 0.01,  0.00, 0.00, 0.00][:n]
    df[Cols.XT_DELTA]       = [0.0, 0.01, 0.01,  0.02, 0.03, 0.05][:n]
    return df


def _make_registry(tmp_path: Path) -> Any:
    from football_ai.ml.serving.model_registry import ModelRegistry
    return ModelRegistry(models_dir=tmp_path / "models")


def _make_store(tmp_path: Path) -> Any:
    from football_ai.ingestion.storage.parquet_store import ParquetStore
    return ParquetStore(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
    )


# ── StateValueComputer ────────────────────────────────────────────────────────

class TestStateValueComputer:
    def _make_computer(
        self,
        p_scores_vals: list[float],
        p_concedes_vals: list[float],
    ) -> StateValueComputer:
        return StateValueComputer(
            p_scores_model=_make_model_mock(p_scores_vals),
            p_concedes_model=_make_model_mock(p_concedes_vals),
        )

    def test_adds_required_columns(self) -> None:
        computer = self._make_computer(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        )
        result = computer.compute(_make_feature_df())
        for col in [Cols.P_SCORES, Cols.P_CONCEDES, "state_value"]:
            assert col in result.columns, f"Missing: {col}"

    def test_state_value_equals_p_scores_minus_p_concedes(self) -> None:
        p_s  = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
        p_c  = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        computer = self._make_computer(p_s, p_c)
        result = computer.compute(_make_feature_df())
        expected = [s - c for s, c in zip(p_s, p_c)]
        np.testing.assert_array_almost_equal(
            result["state_value"].values, expected, decimal=5
        )

    def test_state_value_clipped_to_minus_one_one(self) -> None:
        computer = self._make_computer(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        result = computer.compute(_make_feature_df())
        assert (result["state_value"] >= -1.0).all()
        assert (result["state_value"] <=  1.0).all()

    def test_p_scores_clipped_to_unit_range(self) -> None:
        computer = self._make_computer(
            [2.0, -0.5, 0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
        )
        result = computer.compute(_make_feature_df())
        assert (result[Cols.P_SCORES] >= 0.0).all()
        assert (result[Cols.P_SCORES] <= 1.0).all()

    def test_empty_dataframe_returns_empty_with_columns(self) -> None:
        computer = self._make_computer([], [])
        result = computer.compute(pd.DataFrame())
        for col in [Cols.P_SCORES, Cols.P_CONCEDES, "state_value"]:
            assert col in result.columns

    def test_state_value_series_helper(self) -> None:
        p_s = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        p_c = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        computer = self._make_computer(p_s, p_c)
        series = computer.state_value_series(_make_feature_df())
        assert isinstance(series, pd.Series)
        assert len(series) == 6


# ── ActionValueComputer ───────────────────────────────────────────────────────

class TestActionValueComputer:
    def test_adds_required_columns(self) -> None:
        computer = ActionValueComputer()
        result = computer.compute(_make_state_valued_df())
        for col in [Cols.VAEP_VALUE, Cols.VAEP_OFFENSIVE, Cols.VAEP_DEFENSIVE,
                    "delta_p_scores", "delta_p_concedes"]:
            assert col in result.columns, f"Missing: {col}"

    def test_first_action_vaep_is_zero(self) -> None:
        """First action in a possession has no previous state — V = 0."""
        computer = ActionValueComputer()
        result = computer.compute(_make_state_valued_df())
        assert result[Cols.VAEP_VALUE].iloc[0] == pytest.approx(0.0)

    def test_vaep_equals_state_delta_within_possession(self) -> None:
        """V(ai) = V(Si) - V(Si-1) for consecutive same-possession actions."""
        df = _make_state_valued_df(n=4)
        # Force single possession
        df[Cols.POSSESSION_ID] = [0, 0, 0, 0]
        computer = ActionValueComputer()
        result = computer.compute(df)
        for i in range(1, 4):
            expected_vaep = df["state_value"].iloc[i] - df["state_value"].iloc[i - 1]
            actual_vaep   = result[Cols.VAEP_VALUE].iloc[i]
            assert actual_vaep == pytest.approx(expected_vaep, abs=1e-6), (
                f"Row {i}: expected {expected_vaep:.6f}, got {actual_vaep:.6f}"
            )

    def test_vaep_is_zero_across_possession_boundary(self) -> None:
        """When possession changes, VAEP is 0 (no attributable state change)."""
        df = _make_state_valued_df(n=4)
        df[Cols.POSSESSION_ID] = [0, 0, 1, 1]   # boundary between row 1 and 2
        computer = ActionValueComputer()
        result = computer.compute(df)
        # Row 2 is first in a new possession → VAEP = 0
        assert result[Cols.VAEP_VALUE].iloc[2] == pytest.approx(0.0)

    def test_offensive_value_non_negative(self) -> None:
        computer = ActionValueComputer()
        result = computer.compute(_make_state_valued_df())
        assert (result[Cols.VAEP_OFFENSIVE] >= 0.0).all()

    def test_defensive_value_non_negative(self) -> None:
        computer = ActionValueComputer()
        result = computer.compute(_make_state_valued_df())
        assert (result[Cols.VAEP_DEFENSIVE] >= 0.0).all()

    def test_vaep_value_clipped_to_minus_one_one(self) -> None:
        df = _make_state_valued_df()
        df["state_value"] = [0.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        df[Cols.POSSESSION_ID] = [0] * 6
        computer = ActionValueComputer()
        result = computer.compute(df)
        assert (result[Cols.VAEP_VALUE] >= -1.0).all()
        assert (result[Cols.VAEP_VALUE] <=  1.0).all()

    def test_offensive_is_positive_scoring_delta(self) -> None:
        """Offensive value = max(0, Δ_p_scores)."""
        df = _make_state_valued_df()
        df[Cols.P_SCORES]   = [0.10, 0.30, 0.20, 0.40, 0.35, 0.80]
        df[Cols.P_CONCEDES] = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
        df["state_value"]   = df[Cols.P_SCORES] - df[Cols.P_CONCEDES]
        df[Cols.POSSESSION_ID] = [0] * 6
        computer = ActionValueComputer()
        result = computer.compute(df)
        # Row 1: Δ_p_scores = 0.30 - 0.10 = 0.20 → offensive = 0.20
        assert result[Cols.VAEP_OFFENSIVE].iloc[1] == pytest.approx(0.20, abs=1e-5)
        # Row 2: Δ_p_scores = 0.20 - 0.30 = -0.10 → offensive = 0.0
        assert result[Cols.VAEP_OFFENSIVE].iloc[2] == pytest.approx(0.0, abs=1e-5)

    def test_defensive_is_positive_for_reduced_concede_risk(self) -> None:
        """Defensive value = max(0, -Δ_p_concedes)."""
        df = _make_state_valued_df()
        df[Cols.P_SCORES]   = [0.10] * 6
        df[Cols.P_CONCEDES] = [0.10, 0.05, 0.08, 0.02, 0.01, 0.00]
        df["state_value"]   = df[Cols.P_SCORES] - df[Cols.P_CONCEDES]
        df[Cols.POSSESSION_ID] = [0] * 6
        computer = ActionValueComputer()
        result = computer.compute(df)
        # Row 1: Δ_p_concedes = 0.05 - 0.10 = -0.05 → defensive = 0.05
        assert result[Cols.VAEP_DEFENSIVE].iloc[1] == pytest.approx(0.05, abs=1e-5)

    def test_raises_when_state_value_missing(self) -> None:
        df = _make_feature_df()   # no state_value column
        computer = ActionValueComputer()
        with pytest.raises(ValueError, match="state_value"):
            computer.compute(df)

    def test_empty_dataframe_returns_empty_with_columns(self) -> None:
        computer = ActionValueComputer()
        df_with_cols = _make_state_valued_df()[:0]   # empty, but has columns
        result = computer.compute(df_with_cols)
        for col in [Cols.VAEP_VALUE, Cols.VAEP_OFFENSIVE, Cols.VAEP_DEFENSIVE]:
            assert col in result.columns


# ── VAEPAggregator ────────────────────────────────────────────────────────────

class TestVAEPAggregator:
    def test_player_summary_has_required_columns(self) -> None:
        agg = VAEPAggregator(min_actions=1)
        summary = agg.player_summary(_make_vaep_df())
        for col in ["player_id", "player_name", "team_id", "team_name",
                    "action_count", "vaep_total", "vaep_offensive",
                    "vaep_defensive", "vaep_per_90"]:
            assert col in summary.columns, f"Missing: {col}"

    def test_player_summary_sorted_by_vaep_total(self) -> None:
        agg = VAEPAggregator(min_actions=1)
        summary = agg.player_summary(_make_vaep_df())
        vaep_vals = summary["vaep_total"].values
        assert list(vaep_vals) == sorted(vaep_vals, reverse=True)

    def test_player_vaep_total_matches_manual_sum(self) -> None:
        agg = VAEPAggregator(min_actions=1)
        df  = _make_vaep_df()
        summary = agg.player_summary(df)
        # For each player, check the sum matches
        for _, row in summary.iterrows():
            pid = row["player_id"]
            expected = float(df[df[Cols.PLAYER_ID] == pid][Cols.VAEP_VALUE].sum())
            assert row["vaep_total"] == pytest.approx(round(expected, 4), abs=1e-3)

    def test_min_actions_filters_players(self) -> None:
        agg_strict = VAEPAggregator(min_actions=100)   # nothing passes
        agg_loose  = VAEPAggregator(min_actions=1)
        df = _make_vaep_df()
        assert len(agg_strict.player_summary(df)) == 0
        assert len(agg_loose.player_summary(df))  > 0

    def test_vaep_per_90_is_nan_when_no_time(self) -> None:
        """Players with 0 minutes played get NaN vaep_per_90."""
        agg = VAEPAggregator(min_actions=1)
        df  = _make_vaep_df()
        # All actions at same timestamp → 0 minutes
        df[Cols.TIMESTAMP] = 0.0
        summary = agg.player_summary(df)
        assert summary["vaep_per_90"].isna().all()

    def test_vaep_per_90_positive_when_time_available(self) -> None:
        agg = VAEPAggregator(min_actions=1)
        df  = _make_vaep_df()
        # Timestamps span > 0
        df[Cols.TIMESTAMP] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
        df[Cols.PLAYER_ID] = ["p0"] * 6   # single player to simplify
        df[Cols.VAEP_VALUE] = [0.1] * 6
        summary = agg.player_summary(df)
        assert (summary["vaep_per_90"].dropna() >= 0).all()

    def test_team_summary_has_required_columns(self) -> None:
        agg = VAEPAggregator(min_actions=1)
        summary = agg.team_summary(_make_vaep_df())
        for col in ["team_id", "team_name", "action_count",
                    "vaep_total", "vaep_offensive", "vaep_defensive"]:
            assert col in summary.columns, f"Missing: {col}"

    def test_team_summary_vaep_total_matches_all_players(self) -> None:
        agg = VAEPAggregator(min_actions=1)
        df  = _make_vaep_df()
        team_summary = agg.team_summary(df)
        expected_total = float(df[Cols.VAEP_VALUE].sum())
        actual_total   = float(team_summary["vaep_total"].sum())
        assert actual_total == pytest.approx(round(expected_total, 4), abs=1e-3)

    def test_top_players_returns_n_rows(self) -> None:
        agg = VAEPAggregator(min_actions=1)
        df  = _make_vaep_df()
        top = agg.top_players(df, n=2)
        assert len(top) <= 2

    def test_top_players_sorted_by_requested_metric(self) -> None:
        agg = VAEPAggregator(min_actions=1)
        df  = _make_vaep_df()
        top = agg.top_players(df, n=10, metric="vaep_offensive")
        vals = top["vaep_offensive"].values
        assert list(vals) == sorted(vals, reverse=True)

    def test_raises_when_vaep_columns_missing(self) -> None:
        agg = VAEPAggregator()
        with pytest.raises(ValueError, match="vaep_value"):
            agg.player_summary(_make_feature_df())

    def test_xt_total_included_when_present(self) -> None:
        agg = VAEPAggregator(min_actions=1)
        df  = _make_vaep_df()   # _make_vaep_df includes xt_delta
        summary = agg.player_summary(df)
        assert "xt_total" in summary.columns

    def test_xg_total_included_when_present(self) -> None:
        agg = VAEPAggregator(min_actions=1)
        df  = _make_vaep_df()
        summary = agg.player_summary(df)
        assert "xg_total" in summary.columns


# ── VAEPResult ────────────────────────────────────────────────────────────────

class TestVAEPResult:
    def test_total_actions(self) -> None:
        result = VAEPResult(
            match_id="test",
            actions=pd.DataFrame({"a": range(42)}),
        )
        assert result.total_actions == 42

    def test_n_players(self) -> None:
        result = VAEPResult(
            match_id="test",
            player_summary=pd.DataFrame({"player_id": ["p1", "p2", "p3"]}),
        )
        assert result.n_players == 3

    def test_repr_contains_match_id(self) -> None:
        result = VAEPResult(match_id="3813041")
        assert "3813041" in repr(result)


# ── VAEPPipeline ──────────────────────────────────────────────────────────────

class TestVAEPPipeline:
    """End-to-end pipeline test using mocked models and a temp ParquetStore."""

    def _setup(self, tmp_path: Path):
        """Return (pipeline, feature_df, match_id).

        Mocks are injected directly into the pipeline's private attributes —
        MagicMock objects cannot be pickled by joblib, so we never pass them
        through the registry.  The registry is only needed for the pipeline
        constructor; _load_models() is bypassed by pre-setting the attributes.
        """
        store    = _make_store(tmp_path)
        registry = _make_registry(tmp_path)

        match_id   = "test_match"
        feature_df = _make_feature_df(n=10)

        # Save feature Parquet so the pipeline can load it
        store.save_features(feature_df, match_id)

        # Build mocks — pure Python objects, never serialised
        p_scores_mock = MagicMock()
        p_scores_mock.score_actions.return_value = pd.Series(
            np.linspace(0.02, 0.50, 10), name=Cols.P_SCORES
        )

        p_concedes_mock = MagicMock()
        p_concedes_mock.score_actions.return_value = pd.Series(
            np.linspace(0.01, 0.05, 10), name=Cols.P_CONCEDES
        )

        pipeline = VAEPPipeline(
            registry=registry,
            store=store,
            include_xt=False,   # no xT model needed for basic tests
        )
        # Inject mocks directly — bypasses _load_models() / registry reads
        pipeline._p_scores_model   = p_scores_mock
        pipeline._p_concedes_model = p_concedes_mock

        return pipeline, feature_df, match_id

    def test_run_returns_vaep_result(self, tmp_path: Path) -> None:
        pipeline, _, match_id = self._setup(tmp_path)
        result = pipeline.run(match_id, force=True)
        assert isinstance(result, VAEPResult)
        assert result.match_id == match_id

    def test_result_has_all_vaep_columns(self, tmp_path: Path) -> None:
        pipeline, _, match_id = self._setup(tmp_path)
        result = pipeline.run(match_id, force=True)
        for col in [Cols.P_SCORES, Cols.P_CONCEDES, "state_value",
                    Cols.VAEP_VALUE, Cols.VAEP_OFFENSIVE, Cols.VAEP_DEFENSIVE]:
            assert col in result.actions.columns, f"Missing: {col}"

    def test_result_action_count_matches_feature_df(self, tmp_path: Path) -> None:
        pipeline, feature_df, match_id = self._setup(tmp_path)
        result = pipeline.run(match_id, force=True)
        assert result.total_actions == len(feature_df)

    def test_player_summary_is_dataframe(self, tmp_path: Path) -> None:
        pipeline, _, match_id = self._setup(tmp_path)
        result = pipeline.run(match_id, force=True)
        assert isinstance(result.player_summary, pd.DataFrame)

    def test_team_summary_is_dataframe(self, tmp_path: Path) -> None:
        pipeline, _, match_id = self._setup(tmp_path)
        result = pipeline.run(match_id, force=True)
        assert isinstance(result.team_summary, pd.DataFrame)

    def test_vaep_parquet_written_to_disk(self, tmp_path: Path) -> None:
        pipeline, _, match_id = self._setup(tmp_path)
        pipeline.run(match_id, force=True)
        vaep_file = tmp_path / "processed" / f"match_{match_id}_vaep.parquet"
        assert vaep_file.exists()

    def test_second_call_uses_cache(self, tmp_path: Path) -> None:
        """Second call without force=True should load from Parquet, not recompute."""
        pipeline, _, match_id = self._setup(tmp_path)
        pipeline.run(match_id, force=True)
        # Second call — p_scores model should NOT be called again
        call_count_before = pipeline._p_scores_model.score_actions.call_count
        pipeline.run(match_id, force=False)
        call_count_after = pipeline._p_scores_model.score_actions.call_count
        assert call_count_after == call_count_before, (
            "score_actions() was called again on cached run — caching not working."
        )

    def test_vaep_values_are_finite(self, tmp_path: Path) -> None:
        pipeline, _, match_id = self._setup(tmp_path)
        result = pipeline.run(match_id, force=True)
        vaep_col = result.actions[Cols.VAEP_VALUE].values
        assert np.isfinite(vaep_col).all(), "Non-finite VAEP values found."

    def test_p_scores_in_unit_range(self, tmp_path: Path) -> None:
        pipeline, _, match_id = self._setup(tmp_path)
        result = pipeline.run(match_id, force=True)
        ps = result.actions[Cols.P_SCORES]
        assert (ps >= 0.0).all() and (ps <= 1.0).all()
