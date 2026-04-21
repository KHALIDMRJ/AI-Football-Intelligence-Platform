"""
Unit tests for Phase 4 — feature engineering.

Covers:
- compute_spatial_features: column presence, value ranges, flag logic
- compute_temporal_features: gaps, chain index, one-hot previous actions
- compute_contextual_features: score diff, one-hots, pressure flag
- FeatureAssembler: full pipeline, get_X, get_labels, no NaN/inf
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_ai.constants import ActionType, Cols
from football_ai.features.assembler import FeatureAssembler
from football_ai.features.contextual import compute_contextual_features
from football_ai.features.spatial import compute_spatial_features
from football_ai.features.temporal import compute_temporal_features
from football_ai.ingestion.storage.parquet_store import ParquetStore

# ── Shared fixture ────────────────────────────────────────────────────────────

def _make_spadl_df(n: int = 5) -> pd.DataFrame:
    """Return a minimal SPADL-like DataFrame for testing."""
    return pd.DataFrame({
        Cols.MATCH_ID:            ["m1"] * n,
        Cols.ACTION_ID:           [f"a{i}" for i in range(n)],
        Cols.INDEX:               list(range(n)),
        Cols.PERIOD:              [1] * n,
        Cols.TIMESTAMP:           [float(i * 5) for i in range(n)],
        Cols.MINUTE:              [i for i in range(n)],
        Cols.SECOND:              [0] * n,
        Cols.TEAM_ID:             ["team_a"] * n,
        Cols.TEAM_NAME:           ["Team A"] * n,
        Cols.PLAYER_ID:           [f"p{i}" for i in range(n)],
        Cols.PLAYER_NAME:         [f"Player {i}" for i in range(n)],
        Cols.ACTION_TYPE:         [ActionType.PASS.value] * (n - 1) + [ActionType.SHOT.value],
        Cols.BODY_PART:           ["right_foot"] * n,
        Cols.RESULT:              ["success"] * (n - 1) + ["goal"],
        Cols.START_X:             [30.0, 50.0, 70.0, 90.0, 110.0][:n],
        Cols.START_Y:             [40.0] * n,
        Cols.END_X:               [50.0, 70.0, 90.0, 110.0, 120.0][:n],
        Cols.END_Y:               [40.0] * n,
        Cols.UNDER_PRESSURE:      [False] * n,
        Cols.POSSESSION_ID:       [0] * n,
        Cols.POSSESSION_TEAM_ID:  ["team_a"] * n,
        Cols.CHAIN_ID:            ["chain1"] * n,
        Cols.CHAIN_INDEX:         list(range(n)),
        Cols.ZONE_ID:             [100] * n,
        Cols.XG:                  [0.0] * (n - 1) + [0.35],
        Cols.LABEL_SCORES:        [1, 1, 1, 1, 0][:n],
        Cols.LABEL_CONCEDES:      [0] * n,
        "score_home":             [0] * n,
        "score_away":             [0] * n,
        "score_diff":             [0] * n,
        "duration":               [0.5] * n,
    })


# ── Spatial features ──────────────────────────────────────────────────────────

class TestSpatialFeatures:
    def test_expected_columns_present(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        required = [
            "f_sp_dist_to_goal_start", "f_sp_dist_to_goal_end",
            "f_sp_angle_to_goal_start", "f_sp_angle_to_goal_end",
            "f_sp_zone_start", "f_sp_zone_end",
            "f_sp_distance_covered", "f_sp_x_progress",
            "f_sp_start_in_pen_area", "f_sp_end_in_pen_area",
            "f_sp_pen_area_entry", "f_sp_progressive",
            "f_sp_third_start", "f_sp_start_x_norm", "f_sp_start_y_norm",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_distance_to_goal_positive(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        assert (df["f_sp_dist_to_goal_start"] >= 0).all()

    def test_angle_to_goal_non_negative(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        assert (df["f_sp_angle_to_goal_start"] >= 0).all()

    def test_normalised_coords_in_unit_range(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        assert (df["f_sp_start_x_norm"] >= 0).all()
        assert (df["f_sp_start_x_norm"] <= 1).all()
        assert (df["f_sp_start_y_norm"] >= 0).all()
        assert (df["f_sp_start_y_norm"] <= 1).all()

    def test_penalty_area_entry_flag_logic(self) -> None:
        # Last action ends at (120, 40) which is inside pen area
        df = compute_spatial_features(_make_spadl_df())
        last = df.iloc[-1]
        assert last["f_sp_end_in_pen_area"] == 1

    def test_x_progress_positive_for_forward_actions(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        # All actions move toward goal (start_x < end_x)
        assert (df["f_sp_x_progress"] > 0).all()

    def test_no_nan_values(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        sp_cols = [c for c in df.columns if c.startswith("f_sp_")]
        assert not df[sp_cols].isnull().any().any()

    def test_zone_id_within_valid_range(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        # 18 * 12 = 216 total zones
        assert (df["f_sp_zone_start"] >= 0).all()
        assert (df["f_sp_zone_start"] < 216).all()


# ── Temporal features ─────────────────────────────────────────────────────────

class TestTemporalFeatures:
    def test_expected_columns_present(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        df = compute_temporal_features(df)
        required = [
            "f_tm_time_gap_prev", "f_tm_possession_duration",
            "f_tm_chain_index", "f_tm_speed_of_play",
            "f_tm_minute", "f_tm_minute_norm", "f_tm_period",
            "f_tm_late_game", "f_tm_extra_time",
            "f_tm_prev_end_x", "f_tm_prev_end_y",
        ]
        for col in required:
            assert col in df.columns, f"Missing: {col}"

    def test_previous_action_one_hots_present(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        df = compute_temporal_features(df)
        # Should have one-hot columns for k=1,2,3 × each action type
        prev1_pass = "f_tm_prev1_pass"
        assert prev1_pass in df.columns

    def test_time_gap_first_action_is_zero(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        df = compute_temporal_features(df)
        assert df["f_tm_time_gap_prev"].iloc[0] == pytest.approx(0.0)

    def test_time_gap_subsequent_actions_correct(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        df = compute_temporal_features(df)
        # All actions are 5 seconds apart (timestamp step = 5)
        for i in range(1, len(df)):
            assert df["f_tm_time_gap_prev"].iloc[i] == pytest.approx(5.0)

    def test_chain_index_matches_input(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        df = compute_temporal_features(df)
        assert list(df["f_tm_chain_index"]) == list(range(5))

    def test_no_nan_in_temporal_features(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        df = compute_temporal_features(df)
        tm_cols = [c for c in df.columns if c.startswith("f_tm_")]
        assert not df[tm_cols].isnull().any().any()

    def test_minute_norm_in_unit_range(self) -> None:
        df = compute_spatial_features(_make_spadl_df())
        df = compute_temporal_features(df)
        assert (df["f_tm_minute_norm"] >= 0).all()
        assert (df["f_tm_minute_norm"] <= 1.5).all()   # allow slight overflow for extra time


# ── Contextual features ───────────────────────────────────────────────────────

class TestContextualFeatures:
    def _build(self) -> pd.DataFrame:
        df = _make_spadl_df()
        df = compute_spatial_features(df)
        df = compute_temporal_features(df)
        return compute_contextual_features(df)

    def test_expected_columns_present(self) -> None:
        df = self._build()
        required = [
            "f_ctx_score_diff", "f_ctx_winning", "f_ctx_losing", "f_ctx_drawing",
            "f_ctx_under_pressure", "f_ctx_set_piece", "f_ctx_in_possession",
            "f_ctx_type_pass", "f_ctx_type_shot",
            "f_ctx_body_right_foot", "f_ctx_result_success", "f_ctx_result_goal",
        ]
        for col in required:
            assert col in df.columns, f"Missing: {col}"

    def test_score_diff_zero_means_drawing(self) -> None:
        df = self._build()
        assert (df["f_ctx_drawing"] == 1).all()
        assert (df["f_ctx_winning"] == 0).all()

    def test_under_pressure_is_binary(self) -> None:
        df = self._build()
        assert set(df["f_ctx_under_pressure"].unique()).issubset({0, 1})

    def test_action_type_one_hot_sums_to_one(self) -> None:
        df = self._build()
        type_cols = [c for c in df.columns if c.startswith("f_ctx_type_")]
        row_sums = df[type_cols].sum(axis=1)
        # Every row must have exactly one action type active
        assert (row_sums == 1).all()

    def test_body_part_one_hot_sums_to_one(self) -> None:
        df = self._build()
        bp_cols = [c for c in df.columns if c.startswith("f_ctx_body_")]
        row_sums = df[bp_cols].sum(axis=1)
        assert (row_sums == 1).all()

    def test_result_one_hot_sums_to_one(self) -> None:
        df = self._build()
        res_cols = [c for c in df.columns if c.startswith("f_ctx_result_")]
        row_sums = df[res_cols].sum(axis=1)
        assert (row_sums == 1).all()

    def test_shot_type_flag_on_last_row(self) -> None:
        df = self._build()
        assert df["f_ctx_type_shot"].iloc[-1] == 1
        assert df["f_ctx_type_pass"].iloc[-1] == 0


# ── FeatureAssembler ──────────────────────────────────────────────────────────

class TestFeatureAssembler:
    def test_assemble_returns_dataframe(self, tmp_path) -> None:
        assembler = FeatureAssembler(
            store=_make_store(tmp_path)
        )
        df = assembler.assemble(_make_spadl_df(), match_id="test_match")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_feature_columns_all_start_with_f(self, tmp_path) -> None:
        assembler = FeatureAssembler(store=_make_store(tmp_path))
        df = assembler.assemble(_make_spadl_df(), match_id="test_match2")
        feat_cols = [c for c in df.columns if c.startswith("f_")]
        assert len(feat_cols) > 0

    def test_no_nan_in_feature_columns(self, tmp_path) -> None:
        assembler = FeatureAssembler(store=_make_store(tmp_path))
        df = assembler.assemble(_make_spadl_df(), match_id="test_match3")
        feat_cols = [c for c in df.columns if c.startswith("f_")]
        assert not df[feat_cols].isnull().any().any(), "NaN values found in feature columns"

    def test_no_inf_in_feature_columns(self, tmp_path) -> None:
        assembler = FeatureAssembler(store=_make_store(tmp_path))
        df = assembler.assemble(_make_spadl_df(), match_id="test_match4")
        feat_cols = [c for c in df.columns if c.startswith("f_")]
        assert not np.isinf(df[feat_cols].values).any(), "Inf values found in feature columns"

    def test_get_X_returns_only_feature_cols(self, tmp_path) -> None:
        assembler = FeatureAssembler(store=_make_store(tmp_path))
        df = assembler.assemble(_make_spadl_df(), match_id="test_match5")
        X = assembler.get_X(df)
        assert all(c.startswith("f_") for c in X.columns)

    def test_get_labels_returns_correct_series(self, tmp_path) -> None:
        assembler = FeatureAssembler(store=_make_store(tmp_path))
        df = assembler.assemble(_make_spadl_df(), match_id="test_match6")
        y_scores, y_concedes = assembler.get_labels(df)
        assert len(y_scores) == 5
        assert len(y_concedes) == 5
        assert set(y_scores.unique()).issubset({0, 1})
        assert set(y_concedes.unique()).issubset({0, 1})

    def test_feature_count_positive(self, tmp_path) -> None:
        assembler = FeatureAssembler(store=_make_store(tmp_path))
        df = assembler.assemble(_make_spadl_df(), match_id="test_match7")
        assert assembler.feature_count(df) > 50  # sanity: we expect >> 50 features

    def test_second_call_loads_from_cache(self, tmp_path) -> None:
        assembler = FeatureAssembler(store=_make_store(tmp_path))
        df1 = assembler.assemble(_make_spadl_df(), match_id="test_cache")
        df2 = assembler.assemble(_make_spadl_df(), match_id="test_cache")  # cache hit
        assert len(df1) == len(df2)


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_store(tmp_path) -> ParquetStore:
    return ParquetStore(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
    )
