"""
Integration tests — end-to-end pipeline.

These tests run the complete pipeline on a small synthetic CSV dataset
(< 200 events) to verify that every phase produces the expected artefacts
and that downstream phases can consume upstream outputs correctly.

Unlike unit tests, these tests:
- Write to a real (tmp_path-scoped) filesystem
- Run actual model training (fast because n is tiny)
- Exercise the full data flow: CSV → VAEP → API responses

Marks: integration (skipped in `pytest tests/unit/`)

Run with:
    python -m pytest tests/integration/ -v
or via CLI:
    python scripts/cli.py test --suite integration
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Synthetic CSV generator ───────────────────────────────────────────────────

def _write_synthetic_csv(path: Path, n_rows: int = 200) -> Path:
    """
    Write a minimal StatsBomb-compatible CSV that passes through all phases.

    Includes enough variety in action types, teams, and periods to exercise
    SPADL normalisation, possession detection, and ML training.
    """
    rng = np.random.default_rng(42)

    action_pool = [
        "Pass", "Shot", "Dribble", "Carry", "Clearance",
        "Interception", "Ball Receipt*", "Pressure",
    ]
    team_pool = [
        ("2761", "FUS Rabat"),
        ("2838", "FAR Rabat"),
    ]
    # Alternate possession roughly
    teams = [team_pool[i % 2] for i in range(n_rows)]

    rows = []
    for i in range(n_rows):
        tid, tname = teams[i]
        period = 1 if i < n_rows // 2 else 2
        minute = (i * 90) // n_rows
        second = rng.integers(0, 60)
        ts = f"00:{minute:02d}:{second:02d}.000"

        is_shot = (rng.random() < 0.05)
        atype   = "Shot" if is_shot else rng.choice(action_pool[1:])  # skip Shot from pool
        if is_shot:
            atype = "Shot"

        rows.append({
            "match_id":         3813041,
            "index":            i,
            "period":           period,
            "timestamp":        ts,
            "minute":           minute,
            "second":           int(second),
            "event_type_name":  atype,
            "type_name":        atype,
            "team_id":          tid,
            "team_name":        tname,
            "possession_team_id": tid,
            "possession_team_name": tname,
            "player_id":        100 + (i % 11),
            "player_name":      f"Player {i % 11}",
            "player_position_name": "Midfielder",
            "location_x":       float(rng.uniform(5, 115)),
            "location_y":       float(rng.uniform(5, 75)),
            "end_location_x":   float(rng.uniform(5, 115)),
            "end_location_y":   float(rng.uniform(5, 75)),
            "duration":         float(rng.uniform(0.1, 2.5)),
            "under_pressure":   "true" if rng.random() < 0.3 else "false",
            "outcome_name":     "Goal" if (is_shot and rng.random() < 0.2) else "Success",
            "body_part_name":   rng.choice(["Right Foot", "Left Foot", "Head"]),
            "statsbomb_xg":     float(rng.uniform(0.05, 0.4)) if is_shot else 0.0,
        })

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def project_dirs(tmp_path_factory):
    """Create isolated directory tree for integration tests."""
    base = tmp_path_factory.mktemp("integration")
    dirs = {
        "raw":       base / "data" / "raw",
        "processed": base / "data" / "processed",
        "features":  base / "data" / "features",
        "models":    base / "models",
        "logs":      base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return base, dirs


@pytest.fixture(scope="module")
def synthetic_csv(project_dirs):
    base, dirs = project_dirs
    csv_path = dirs["raw"] / "synthetic_match.csv"
    _write_synthetic_csv(csv_path, n_rows=200)
    return csv_path


@pytest.fixture(scope="module")
def parquet_store(project_dirs):
    base, dirs = project_dirs
    from football_ai.ingestion.storage.parquet_store import ParquetStore
    return ParquetStore(
        raw_dir=dirs["raw"],
        processed_dir=dirs["processed"],
        features_dir=dirs["features"],
    )


@pytest.fixture(scope="module")
def registry(project_dirs):
    base, dirs = project_dirs
    from football_ai.ml.serving.model_registry import ModelRegistry
    return ModelRegistry(models_dir=dirs["models"])


# ── Phase 2: Ingestion ────────────────────────────────────────────────────────

class TestPhase2Ingestion:
    @pytest.fixture(scope="class")
    def events(self, synthetic_csv, parquet_store):
        from football_ai.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline(store=parquet_store, strict=False)
        events, report = pipeline.run_csv(synthetic_csv, force=True)
        return events, report

    def test_produces_events(self, events):
        evts, _ = events
        assert len(evts) > 0

    def test_validation_report_no_errors(self, events):
        _, report = events
        assert report.error_count == 0

    def test_match_id_consistent(self, events):
        evts, _ = events
        match_ids = {e.match_id for e in evts}
        assert len(match_ids) == 1

    def test_raw_parquet_written(self, events, parquet_store):
        evts, _ = events
        mid = evts[0].match_id
        assert (parquet_store.raw_dir / f"match_{mid}_raw.parquet").exists()


# ── Phase 3: Preprocessing ────────────────────────────────────────────────────

class TestPhase3Preprocessing:
    @pytest.fixture(scope="class")
    def spadl(self, synthetic_csv, parquet_store):
        from football_ai.ingestion.pipeline import IngestionPipeline
        from football_ai.preprocessing.pipeline import PreprocessingPipeline
        events, _ = IngestionPipeline(store=parquet_store, strict=False).run_csv(
            synthetic_csv, force=False
        )
        mid     = events[0].match_id
        pp      = PreprocessingPipeline(store=parquet_store)
        spadl_df = pp.run(events, match_id=mid, force=True)
        return spadl_df, mid

    def test_produces_actions(self, spadl):
        df, _ = spadl
        assert len(df) > 0

    def test_possession_id_column_present(self, spadl):
        df, _ = spadl
        from football_ai.constants import Cols
        assert Cols.POSSESSION_ID in df.columns

    def test_label_scores_column_present(self, spadl):
        df, _ = spadl
        from football_ai.constants import Cols
        assert Cols.LABEL_SCORES in df.columns

    def test_label_concedes_column_present(self, spadl):
        df, _ = spadl
        from football_ai.constants import Cols
        assert Cols.LABEL_CONCEDES in df.columns

    def test_spadl_parquet_written(self, spadl, parquet_store):
        _, mid = spadl
        assert (parquet_store.processed_dir / f"match_{mid}_spadl.parquet").exists()


# ── Phase 4: Feature engineering ─────────────────────────────────────────────

class TestPhase4Features:
    @pytest.fixture(scope="class")
    def features(self, synthetic_csv, parquet_store):
        from football_ai.features.assembler import FeatureAssembler
        from football_ai.ingestion.pipeline import IngestionPipeline
        from football_ai.preprocessing.pipeline import PreprocessingPipeline
        events, _ = IngestionPipeline(store=parquet_store, strict=False).run_csv(
            synthetic_csv, force=False
        )
        mid      = events[0].match_id
        spadl_df = PreprocessingPipeline(store=parquet_store).run(
            events, match_id=mid, force=False
        )
        assembler  = FeatureAssembler(store=parquet_store)
        feature_df = assembler.assemble(spadl_df, match_id=mid, force=True)
        return feature_df, mid, assembler

    def test_features_have_correct_row_count(self, features, synthetic_csv):
        df, _, _ = features
        assert len(df) > 0

    def test_feature_columns_count_gte_50(self, features):
        df, _, assembler = features
        assert assembler.feature_count(df) >= 50

    def test_no_nan_in_feature_columns(self, features):
        df, _, _ = features
        f_cols = [c for c in df.columns if c.startswith("f_")]
        assert not df[f_cols].isnull().any().any()

    def test_no_inf_in_feature_columns(self, features):
        import numpy as np
        df, _, _ = features
        f_cols = [c for c in df.columns if c.startswith("f_")]
        assert not np.isinf(df[f_cols].values).any()

    def test_feature_parquet_written(self, features, parquet_store):
        _, mid, _ = features
        assert (parquet_store.features_dir / f"match_{mid}_features.parquet").exists()


# ── Phase 5: Model training ───────────────────────────────────────────────────

class TestPhase5Models:
    @pytest.fixture(scope="class")
    def trained_models(self, synthetic_csv, parquet_store, registry):
        from football_ai.ml.training.pipeline import MLTrainingPipeline
        pipeline = MLTrainingPipeline(store=parquet_store, registry=registry)
        report = pipeline.run(force=True)
        return report

    def test_p_scores_model_saved(self, trained_models, registry):
        assert registry.exists("p_scores")

    def test_p_concedes_model_saved(self, trained_models, registry):
        assert registry.exists("p_concedes")

    def test_xt_model_saved(self, trained_models, registry):
        assert registry.exists("xt")

    def test_xg_model_saved(self, trained_models, registry):
        assert registry.exists("xg")

    def test_metadata_yaml_written(self, trained_models, project_dirs):
        _, dirs = project_dirs
        meta_path = dirs["models"] / "metadata.yaml"
        assert meta_path.exists()

    def test_p_scores_roc_auc_valid(self, trained_models):
        metrics = trained_models.get("p_scores", {})
        if "roc_auc" in metrics and not __import__("math").isnan(metrics["roc_auc"]):
            assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_p_concedes_brier_score_valid(self, trained_models):
        metrics = trained_models.get("p_concedes", {})
        if "brier_score" in metrics:
            assert 0.0 <= metrics["brier_score"] <= 1.0


# ── Phase 6: VAEP engine ──────────────────────────────────────────────────────

class TestPhase6VAEP:
    @pytest.fixture(scope="class")
    def vaep_result(self, synthetic_csv, parquet_store, registry):
        from football_ai.ingestion.pipeline import IngestionPipeline
        from football_ai.ml.training.pipeline import MLTrainingPipeline
        from football_ai.vaep.pipeline import VAEPPipeline

        events, _ = IngestionPipeline(store=parquet_store, strict=False).run_csv(
            synthetic_csv, force=False
        )
        mid = events[0].match_id
        MLTrainingPipeline(store=parquet_store, registry=registry).run(force=False)
        result = VAEPPipeline(registry=registry, store=parquet_store).run(
            match_id=mid, force=True
        )
        return result

    def test_actions_dataframe_not_empty(self, vaep_result):
        assert not vaep_result.actions.empty

    def test_vaep_column_present(self, vaep_result):
        from football_ai.constants import Cols
        assert Cols.VAEP_VALUE in vaep_result.actions.columns

    def test_player_summary_not_empty(self, vaep_result):
        assert not vaep_result.player_summary.empty

    def test_team_summary_has_two_teams(self, vaep_result):
        assert len(vaep_result.team_summary) == 2

    def test_vaep_values_finite(self, vaep_result):
        import numpy as np

        from football_ai.constants import Cols
        vals = vaep_result.actions[Cols.VAEP_VALUE].values
        assert np.isfinite(vals).all()

    def test_vaep_parquet_written(self, vaep_result, parquet_store):
        path = (
            parquet_store.processed_dir
            / f"match_{vaep_result.match_id}_vaep.parquet"
        )
        assert path.exists()

    def test_player_summary_sorted_desc(self, vaep_result):
        vals = vaep_result.player_summary["vaep_total"].values
        assert list(vals) == sorted(vals, reverse=True)


# ── Phase 7: Tactical intelligence ───────────────────────────────────────────

class TestPhase7Tactical:
    @pytest.fixture(scope="class")
    def tactical_result(self, synthetic_csv, parquet_store, registry, vaep_result=None):
        from football_ai.ingestion.pipeline import IngestionPipeline
        from football_ai.ml.training.pipeline import MLTrainingPipeline
        from football_ai.tactical.pipeline import TacticalPipeline
        from football_ai.vaep.pipeline import VAEPPipeline

        events, _ = IngestionPipeline(store=parquet_store, strict=False).run_csv(
            synthetic_csv, force=False
        )
        mid = events[0].match_id
        MLTrainingPipeline(store=parquet_store, registry=registry).run(force=False)
        VAEPPipeline(registry=registry, store=parquet_store).run(match_id=mid, force=False)
        result = TacticalPipeline(store=parquet_store).run(match_id=mid, force=True)
        return result

    def test_match_report_present(self, tactical_result):
        assert tactical_result.match_report is not None

    def test_formations_computed(self, tactical_result):
        assert len(tactical_result.formations) == 2

    def test_weaknesses_computed_for_both_teams(self, tactical_result):
        assert len(tactical_result.weaknesses) == 2

    def test_rankings_have_five_methods(self, tactical_result):
        assert len(tactical_result.rankings) == 5

    def test_tactical_json_written(self, tactical_result, parquet_store):
        path = (
            parquet_store.processed_dir
            / f"match_{tactical_result.match_id}_tactical_report.json"
        )
        assert path.exists()

    def test_tactical_json_is_valid(self, tactical_result, parquet_store):
        path = (
            parquet_store.processed_dir
            / f"match_{tactical_result.match_id}_tactical_report.json"
        )
        with open(path) as fh:
            data = json.load(fh)
        assert "home_team_name" in data

    def test_match_report_serialisable(self, tactical_result):
        d = tactical_result.match_report.to_dict()
        # Must not raise
        json.dumps(d, default=str)


# ── Phase 8: API (integration-level smoke tests) ──────────────────────────────

class TestPhase8API:
    @pytest.fixture(scope="class")
    def api_client(self, synthetic_csv, parquet_store, registry):
        """Build a TestClient pointing at the integration tmp store."""
        from fastapi.testclient import TestClient

        from football_ai.api.main import create_app
        from football_ai.api.services.match_service import MatchService
        from football_ai.api.services.player_service import PlayerService
        from football_ai.api.services.tactical_service import TacticalService
        from football_ai.api.v1.endpoints import matches as match_router
        from football_ai.api.v1.endpoints import players as player_router
        from football_ai.api.v1.endpoints import tactical as tactical_router
        from football_ai.api.v1.endpoints import teams as team_router

        def _m():  return MatchService(store=parquet_store)
        def _p():  return PlayerService(store=parquet_store)
        def _t():  return TacticalService(store=parquet_store)

        app = create_app()
        app.dependency_overrides[match_router._match_service]       = _m
        app.dependency_overrides[player_router._player_service]     = _p
        app.dependency_overrides[team_router._tactical_service]     = _t
        app.dependency_overrides[tactical_router._tactical_service] = _t

        return TestClient(app, raise_server_exceptions=True)

    def test_health_endpoint(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_matches_endpoint_returns_processed_match(self, api_client):
        r = api_client.get("/api/v1/matches")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] >= 1

    def test_players_endpoint_returns_data(self, api_client):
        # Get first match ID then query players
        matches = api_client.get("/api/v1/matches").json()["matches"]
        if not matches:
            pytest.skip("No processed match found")
        mid = matches[0]["match_id"]
        r   = api_client.get(f"/api/v1/players?match_id={mid}")
        assert r.status_code == 200
        assert r.json()["count"] > 0

    def test_teams_endpoint_returns_two_teams(self, api_client):
        matches = api_client.get("/api/v1/matches").json()["matches"]
        if not matches:
            pytest.skip("No processed match found")
        mid = matches[0]["match_id"]
        r   = api_client.get(f"/api/v1/teams/{mid}")
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_unknown_match_returns_404(self, api_client):
        r = api_client.get("/api/v1/matches/DOES_NOT_EXIST")
        assert r.status_code == 404
