"""
Unit tests for Phase 5 — ML models.

Covers:
- XGModel: fit, predict_proba, score_actions, save/load, unfitted guard
- XTModel: fit, zone_xT, score_actions, grid shape, save/load
- PScoresModel: fit, predict_proba, score_actions, save/load
- PConcedesModel: fit, predict_proba, score_actions, save/load
- ModelRegistry: save, load, exists, list_available, metadata
- Evaluator: evaluate_binary_classifier, calibration_summary
"""

from __future__ import annotations  # noqa: I001

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_ai.constants import ActionResult, ActionType, Cols
from football_ai.ml.models.p_concedes_model import PConcedesModel
from football_ai.ml.models.p_scores_model import PScoresModel
from football_ai.ml.models.xg_model import XGModel
from football_ai.ml.models.xt_model import XTModel
from football_ai.ml.serving.model_registry import ModelRegistry
from football_ai.ml.training.evaluator import (
    calibration_summary,
    evaluate_binary_classifier,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(models_dir=tmp_path / "models")


def _make_feature_df(n: int = 200, n_pos: int = 20) -> pd.DataFrame:
    """Minimal feature DataFrame with label columns."""
    rng = np.random.default_rng(42)
    n_features = 30
    data = {
        Cols.MATCH_ID:           ["m1"] * n,
        Cols.PLAYER_ID:          [f"p{i % 10}" for i in range(n)],
        Cols.ACTION_TYPE:        (
            [ActionType.SHOT.value] * n_pos +
            [ActionType.PASS.value] * (n - n_pos)
        ),
        Cols.RESULT:             (
            [ActionResult.GOAL.value] * (n_pos // 2) +
            [ActionResult.FAIL.value] * (n_pos - n_pos // 2) +
            [ActionResult.SUCCESS.value] * (n - n_pos)
        ),
        Cols.START_X:            rng.uniform(0, 120, n),
        Cols.START_Y:            rng.uniform(0, 80, n),
        Cols.END_X:              rng.uniform(0, 120, n),
        Cols.END_Y:              rng.uniform(0, 80, n),
        Cols.LABEL_SCORES:       ([1] * n_pos + [0] * (n - n_pos)),
        Cols.LABEL_CONCEDES:     ([0] * (n - 5) + [1] * 5),
        Cols.POSSESSION_TEAM_ID: ["team_a"] * n,
    }
    # Feature columns
    for i in range(n_features):
        data[f"f_feat_{i:02d}"] = rng.standard_normal(n)
    # Add some realistic spatial features that xG model looks for
    data["f_sp_dist_to_goal_start"] = rng.uniform(5, 50, n)
    data["f_sp_angle_to_goal_start"] = rng.uniform(0, 1, n)
    data["f_sp_start_in_pen_area"]   = rng.integers(0, 2, n)
    data["f_ctx_body_right_foot"]    = rng.integers(0, 2, n)
    data["f_ctx_under_pressure"]     = rng.integers(0, 2, n)
    data["f_ctx_type_shot"]          = [1] * n_pos + [0] * (n - n_pos)
    return pd.DataFrame(data)


def _make_spadl_df(n: int = 300) -> pd.DataFrame:
    """Minimal SPADL DataFrame for xT model fitting."""
    rng = np.random.default_rng(7)
    action_pool = [
        ActionType.PASS.value,
        ActionType.CARRY.value,
        ActionType.SHOT.value,
        ActionType.DRIBBLE.value,
    ]
    return pd.DataFrame({
        Cols.START_X:    rng.uniform(0, 120, n),
        Cols.START_Y:    rng.uniform(0, 80, n),
        Cols.END_X:      rng.uniform(0, 120, n),
        Cols.END_Y:      rng.uniform(0, 80, n),
        Cols.ACTION_TYPE: rng.choice(action_pool, n),
        Cols.RESULT:     rng.choice(
            [ActionResult.SUCCESS.value, ActionResult.FAIL.value, ActionResult.GOAL.value],
            n, p=[0.7, 0.25, 0.05],
        ),
    })


# ── ModelRegistry ─────────────────────────────────────────────────────────────

class TestModelRegistry:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        obj = {"key": "value", "number": 42}
        registry.save("xg", obj, metrics={"roc_auc": 0.75})
        loaded = registry.load("xg")
        assert loaded == obj

    def test_exists_false_before_save(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        assert registry.exists("xg") is False

    def test_exists_true_after_save(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        registry.save("p_scores", {"m": 1})
        assert registry.exists("p_scores") is True

    def test_list_available_empty_initially(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        assert registry.list_available() == []

    def test_list_available_after_saves(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        registry.save("xg", {"x": 1})
        registry.save("xt", {"y": 2})
        available = registry.list_available()
        assert "xg" in available
        assert "xt" in available

    def test_load_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        with pytest.raises(FileNotFoundError):
            registry.load("p_scores")

    def test_invalid_model_name_raises_value_error(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        with pytest.raises(ValueError, match="Unknown model name"):
            registry.save("unknown_model", {})

    def test_metadata_written_on_save(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        registry.save("xg", {"x": 1}, metrics={"roc_auc": 0.80}, feature_cols=["f_a", "f_b"])
        meta = registry.get_metadata()
        assert "xg" in meta
        assert meta["xg"]["metrics"]["roc_auc"] == pytest.approx(0.80)
        assert meta["xg"]["feature_count"] == 2

    def test_get_feature_cols_returns_list(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        cols = ["f_sp_dist", "f_ctx_body"]
        registry.save("xg", {"x": 1}, feature_cols=cols)
        assert registry.get_feature_cols("xg") == cols


# ── XGModel ───────────────────────────────────────────────────────────────────

class TestXGModel:
    def test_fit_and_predict_proba(self) -> None:
        feature_df = _make_feature_df()
        X, y = XGModel.prepare_training_data(feature_df)
        model = XGModel(calibrate=False)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X),)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_proba_with_calibration(self) -> None:
        feature_df = _make_feature_df(n=100, n_pos=15)
        X, y = XGModel.prepare_training_data(feature_df)
        model = XGModel(calibrate=True)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_score_actions_zeros_for_non_shots(self) -> None:
        feature_df = _make_feature_df()
        X, y = XGModel.prepare_training_data(feature_df)
        model = XGModel(calibrate=False)
        model.fit(X, y)
        xg = model.score_actions(feature_df)
        non_shot_mask = ~feature_df[Cols.ACTION_TYPE].isin({"shot", "header"})
        assert (xg[non_shot_mask] == 0.0).all()

    def test_score_actions_nonzero_for_shots(self) -> None:
        feature_df = _make_feature_df()
        X, y = XGModel.prepare_training_data(feature_df)
        model = XGModel(calibrate=False)
        model.fit(X, y)
        xg = model.score_actions(feature_df)
        shot_mask = feature_df[Cols.ACTION_TYPE].isin({"shot", "header"})
        # At least some shots should have non-zero xG
        assert (xg[shot_mask] > 0).any()

    def test_predict_proba_raises_when_unfitted(self) -> None:
        model = XGModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(pd.DataFrame({"f_sp_x": [1.0]}))

    def test_save_and_load(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        feature_df = _make_feature_df()
        X, y = XGModel.prepare_training_data(feature_df)
        model = XGModel(calibrate=False)
        model.fit(X, y)
        model.save(registry)
        loaded = XGModel.load(registry)
        probs_orig   = model.predict_proba(X)
        probs_loaded = loaded.predict_proba(X)
        np.testing.assert_array_almost_equal(probs_orig, probs_loaded)

    def test_prepare_training_data_raises_no_shots(self) -> None:
        feature_df = _make_feature_df()
        feature_df[Cols.ACTION_TYPE] = ActionType.PASS.value  # all passes
        with pytest.raises(ValueError, match="No shot events"):
            XGModel.prepare_training_data(feature_df)


# ── XTModel ───────────────────────────────────────────────────────────────────

class TestXTModel:
    def test_fit_produces_grid(self) -> None:
        model = XTModel()
        model.fit(_make_spadl_df())
        assert model.xT_grid is not None
        assert model.xT_grid.shape == (model.zones_y, model.zones_x)

    def test_xT_values_in_unit_range(self) -> None:
        model = XTModel()
        model.fit(_make_spadl_df())
        flat = model.xT_flat()
        assert (flat >= 0.0).all()
        assert (flat <= 1.0).all()

    def test_xT_high_near_goal(self) -> None:
        """
        Zones near the goal should have higher xT than the own half.
        We build a realistic dataset where shots cluster near x=110-120
        and passes spread across the whole pitch.
        """
        rng = np.random.default_rng(99)
        n_passes = 3000
        n_shots  = 200

        pass_data = {
            Cols.START_X:    rng.uniform(0, 120, n_passes),
            Cols.START_Y:    rng.uniform(0, 80, n_passes),
            Cols.END_X:      rng.uniform(0, 120, n_passes),
            Cols.END_Y:      rng.uniform(0, 80, n_passes),
            Cols.ACTION_TYPE: [ActionType.PASS.value] * n_passes,
            Cols.RESULT:     [ActionResult.SUCCESS.value] * n_passes,
        }
        # Shots concentrated near goal, 15% scoring rate
        shot_data = {
            Cols.START_X:    rng.uniform(100, 120, n_shots),
            Cols.START_Y:    rng.uniform(20, 60, n_shots),
            Cols.END_X:      np.full(n_shots, 120.0),
            Cols.END_Y:      np.full(n_shots, 40.0),
            Cols.ACTION_TYPE: [ActionType.SHOT.value] * n_shots,
            Cols.RESULT:     rng.choice(
                [ActionResult.GOAL.value, ActionResult.FAIL.value],
                n_shots, p=[0.15, 0.85],
            ),
        }
        df = pd.concat([pd.DataFrame(pass_data), pd.DataFrame(shot_data)], ignore_index=True)
        model = XTModel(smoothing=0.1)
        model.fit(df)
        near_goal = model.zone_xT(115.0, 40.0)
        own_half  = model.zone_xT(10.0, 40.0)
        assert near_goal > own_half, (
            f"xT near goal={near_goal:.4f} should be > xT own half={own_half:.4f}"
        )

    def test_score_actions_adds_columns(self) -> None:
        model = XTModel()
        model.fit(_make_spadl_df())
        df_scored = model.score_actions(_make_spadl_df())
        for col in [Cols.XT_START, Cols.XT_END, Cols.XT_DELTA]:
            assert col in df_scored.columns

    def test_xt_delta_column_finite(self) -> None:
        model = XTModel()
        model.fit(_make_spadl_df())
        df_scored = model.score_actions(_make_spadl_df())
        assert np.isfinite(df_scored[Cols.XT_DELTA].values).all()

    def test_zone_xt_raises_when_unfitted(self) -> None:
        model = XTModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.zone_xT(60.0, 40.0)

    def test_save_and_load(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        model = XTModel()
        model.fit(_make_spadl_df())
        model.save(registry)
        loaded = XTModel.load(registry)
        assert loaded.xT_grid is not None
        np.testing.assert_array_almost_equal(
            model.xT_flat(), loaded.xT_flat()
        )


# ── PScoresModel ──────────────────────────────────────────────────────────────

class TestPScoresModel:
    def test_fit_and_predict_proba(self) -> None:
        feature_df = _make_feature_df()
        X, y = PScoresModel.prepare_training_data(feature_df)
        model = PScoresModel(n_estimators=30, early_stopping_rounds=5)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X),)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_score_actions_returns_series(self) -> None:
        feature_df = _make_feature_df()
        X, y = PScoresModel.prepare_training_data(feature_df)
        model = PScoresModel(n_estimators=30, early_stopping_rounds=5)
        model.fit(X, y)
        result = model.score_actions(feature_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(feature_df)
        assert result.name == Cols.P_SCORES

    def test_predict_proba_raises_when_unfitted(self) -> None:
        model = PScoresModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(pd.DataFrame({"f_x": [1.0]}))

    def test_feature_cols_recorded(self) -> None:
        feature_df = _make_feature_df()
        X, y = PScoresModel.prepare_training_data(feature_df)
        model = PScoresModel(n_estimators=20, early_stopping_rounds=5)
        model.fit(X, y)
        assert len(model.feature_cols) == X.shape[1]
        assert model.feature_cols == list(X.columns)

    def test_save_and_load(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        feature_df = _make_feature_df()
        X, y = PScoresModel.prepare_training_data(feature_df)
        model = PScoresModel(n_estimators=20, early_stopping_rounds=5)
        model.fit(X, y)
        model.save(registry)
        loaded = PScoresModel.load(registry)
        probs_orig   = model.predict_proba(X)
        probs_loaded = loaded.predict_proba(X)
        np.testing.assert_array_almost_equal(probs_orig, probs_loaded)

    def test_prepare_data_raises_missing_label(self) -> None:
        feature_df = _make_feature_df()
        feature_df = feature_df.drop(columns=[Cols.LABEL_SCORES])
        with pytest.raises(ValueError, match="label_scores"):
            PScoresModel.prepare_training_data(feature_df)


# ── PConcedesModel ────────────────────────────────────────────────────────────

class TestPConcedesModel:
    def test_fit_and_predict_proba(self) -> None:
        feature_df = _make_feature_df()
        X, y = PConcedesModel.prepare_training_data(feature_df)
        model = PConcedesModel(n_estimators=30, early_stopping_rounds=5)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X),)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_score_actions_returns_series(self) -> None:
        feature_df = _make_feature_df()
        X, y = PConcedesModel.prepare_training_data(feature_df)
        model = PConcedesModel(n_estimators=30, early_stopping_rounds=5)
        model.fit(X, y)
        result = model.score_actions(feature_df)
        assert isinstance(result, pd.Series)
        assert result.name == Cols.P_CONCEDES

    def test_predict_proba_raises_when_unfitted(self) -> None:
        model = PConcedesModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(pd.DataFrame({"f_x": [1.0]}))

    def test_save_and_load(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        feature_df = _make_feature_df()
        X, y = PConcedesModel.prepare_training_data(feature_df)
        model = PConcedesModel(n_estimators=20, early_stopping_rounds=5)
        model.fit(X, y)
        model.save(registry)
        loaded = PConcedesModel.load(registry)
        probs_orig   = model.predict_proba(X)
        probs_loaded = loaded.predict_proba(X)
        np.testing.assert_array_almost_equal(probs_orig, probs_loaded)

    def test_prepare_data_raises_missing_label(self) -> None:
        feature_df = _make_feature_df()
        feature_df = feature_df.drop(columns=[Cols.LABEL_CONCEDES])
        with pytest.raises(ValueError, match="label_concedes"):
            PConcedesModel.prepare_training_data(feature_df)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class TestEvaluator:
    def _make_binary(self, n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, n)
        y_prob = rng.uniform(0, 1, n)
        return y_true, y_prob

    def test_evaluate_returns_expected_keys(self) -> None:
        y_true, y_prob = self._make_binary()
        metrics = evaluate_binary_classifier(y_true, y_prob)
        for key in ["roc_auc", "brier_score", "log_loss", "precision",
                    "recall", "f1", "mean_pred_prob", "mean_actual_prob",
                    "n_samples", "n_positive"]:
            assert key in metrics, f"Missing metric: {key}"

    def test_roc_auc_in_valid_range(self) -> None:
        y_true, y_prob = self._make_binary()
        metrics = evaluate_binary_classifier(y_true, y_prob)
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_brier_score_in_valid_range(self) -> None:
        y_true, y_prob = self._make_binary()
        metrics = evaluate_binary_classifier(y_true, y_prob)
        assert 0.0 <= metrics["brier_score"] <= 1.0

    def test_perfect_classifier_high_auc(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = evaluate_binary_classifier(y_true, y_prob)
        assert metrics["roc_auc"] == pytest.approx(1.0)

    def test_degenerate_single_class_returns_nan_auc(self) -> None:
        y_true = np.zeros(50, dtype=int)
        y_prob = np.random.default_rng(0).uniform(0, 1, 50)
        metrics = evaluate_binary_classifier(y_true, y_prob)
        assert np.isnan(metrics["roc_auc"])

    def test_calibration_summary_returns_dataframe(self) -> None:
        y_true, y_prob = self._make_binary()
        cal_df = calibration_summary(y_true, y_prob, n_bins=5)
        assert isinstance(cal_df, pd.DataFrame)
        assert "mean_pred" in cal_df.columns
        assert "mean_actual" in cal_df.columns
        assert len(cal_df) <= 5

    def test_n_samples_correct(self) -> None:
        y_true, y_prob = self._make_binary(n=150)
        metrics = evaluate_binary_classifier(y_true, y_prob)
        assert metrics["n_samples"] == pytest.approx(150.0)
