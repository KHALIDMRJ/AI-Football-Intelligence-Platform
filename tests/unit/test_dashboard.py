"""
Unit tests for Phase 9 — Streamlit Dashboard.

Strategy
--------
We cannot call Streamlit's rendering functions (``st.title``, ``st.plotly_chart``, …)
in a plain pytest run because they require a live Streamlit runtime.

What we CAN test without a runtime:
1. data_loader.py — all pure functions that read Parquet/JSON and return
   Python objects (no ``st.*`` calls).
2. Import health — every page module imports without raising.
3. ``pipeline_status()`` — reads from filesystem, no Streamlit calls.
4. Plotly figure builders — extracted from pages and called directly.

This gives us solid coverage of the data layer (the part most likely to
break) without needing a browser or Streamlit test runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from football_ai.constants import ActionResult, ActionType, Cols

_MATCH_ID = "test_dash_match"
_HOME_ID  = "dash_home"
_AWAY_ID  = "dash_away"


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _write_artefacts(processed_dir: Path) -> None:
    """Write minimal Parquet and JSON artefacts for data_loader tests."""
    from football_ai.utils import save_parquet
    from football_ai.vaep.aggregator import VAEPAggregator

    processed_dir.mkdir(parents=True, exist_ok=True)
    n   = 30
    rng = np.random.default_rng(1)

    df = pd.DataFrame({
        Cols.MATCH_ID:           [_MATCH_ID] * n,
        Cols.ACTION_ID:          [f"a{i}" for i in range(n)],
        Cols.INDEX:              list(range(n)),
        Cols.PERIOD:             [1] * n,
        Cols.TIMESTAMP:          np.linspace(0, 5400, n),
        Cols.MINUTE:             (np.linspace(0, 90, n)).astype(int),
        Cols.SECOND:             [0] * n,
        Cols.TEAM_ID:            ([_HOME_ID] * 15) + ([_AWAY_ID] * 15),
        Cols.TEAM_NAME:          (["Home FC"] * 15) + (["Away FC"] * 15),
        Cols.PLAYER_ID:          [f"p{i % 3}" for i in range(n)],
        Cols.PLAYER_NAME:        [f"Player {i % 3}" for i in range(n)],
        Cols.ACTION_TYPE:        rng.choice([ActionType.PASS.value, ActionType.SHOT.value], n),
        Cols.RESULT:             rng.choice(
            [ActionResult.SUCCESS.value, ActionResult.GOAL.value], n,
        ),
        Cols.START_X:            rng.uniform(0, 120, n),
        Cols.START_Y:            rng.uniform(0, 80, n),
        Cols.END_X:              rng.uniform(0, 120, n),
        Cols.END_Y:              rng.uniform(0, 80, n),
        Cols.UNDER_PRESSURE:     rng.integers(0, 2, n).astype(bool),
        Cols.POSSESSION_ID:      np.arange(n) // 5,
        Cols.POSSESSION_TEAM_ID: ([_HOME_ID] * 15) + ([_AWAY_ID] * 15),
        Cols.ZONE_ID:            rng.integers(0, 216, n),
        Cols.XG:                 rng.uniform(0, 0.3, n) * (rng.random(n) < 0.1),
        Cols.VAEP_VALUE:         rng.uniform(-0.3, 0.4, n),
        Cols.VAEP_OFFENSIVE:     rng.uniform(0, 0.3, n),
        Cols.VAEP_DEFENSIVE:     rng.uniform(0, 0.1, n),
        Cols.XT_DELTA:           rng.uniform(-0.05, 0.08, n),
        Cols.P_SCORES:           rng.uniform(0, 0.2, n),
        Cols.P_CONCEDES:         rng.uniform(0, 0.1, n),
        "state_value":           rng.uniform(-0.2, 0.2, n),
        "score_diff":            [0] * n,
    })

    save_parquet(df, processed_dir / f"match_{_MATCH_ID}_vaep.parquet")

    agg = VAEPAggregator(min_actions=1)
    ps  = agg.player_summary(df)
    ts  = agg.team_summary(df)
    save_parquet(ps, processed_dir / f"match_{_MATCH_ID}_player_summary.parquet")
    save_parquet(ts, processed_dir / f"match_{_MATCH_ID}_team_summary.parquet")

    # Weakness parquets
    from football_ai.tactical.weakness_detector import WeaknessDetector
    wd = WeaknessDetector(min_actions_per_zone=1)
    for tid in [_HOME_ID, _AWAY_ID]:
        w = wd.detect(df, team_id=tid)
        if not w.empty:
            save_parquet(w, processed_dir / f"match_{_MATCH_ID}_weaknesses_{tid}.parquet")

    # Pressure parquets
    from football_ai.tactical.pressure_map import PressureMap
    pm = PressureMap(min_zone_actions=1)
    for tid in [_HOME_ID, _AWAY_ID]:
        pmr = pm.build(df, team_id=tid)
        if not pmr.opponent_threat.empty:
            save_parquet(
                pmr.opponent_threat,
                processed_dir / f"match_{_MATCH_ID}_pressure_{tid}.parquet",
            )

    # Tactical report JSON
    report: dict[str, Any] = {
        "match_id":       _MATCH_ID,
        "home_team_id":   _HOME_ID,
        "away_team_id":   _AWAY_ID,
        "home_team_name": "Home FC",
        "away_team_name": "Away FC",
        "home_formation": "4-3-3",
        "away_formation": "4-4-2",
    }
    with open(processed_dir / f"match_{_MATCH_ID}_tactical_report.json", "w") as fh:
        json.dump(report, fh)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """Module-scoped temp dir with all artefacts written once."""
    base = tmp_path_factory.mktemp("dash_data")
    _write_artefacts(base / "processed")
    return base


@pytest.fixture(scope="module")
def patched_loader(data_dir, monkeypatch_module):
    """
    Patch the data_loader module to use tmp_path processed dir.
    Returns the patched module.
    """
    from football_ai.ingestion.storage.parquet_store import ParquetStore
    store = ParquetStore(
        raw_dir=data_dir / "raw",
        processed_dir=data_dir / "processed",
        features_dir=data_dir / "features",
    )
    import football_ai.dashboard.data_loader as dl
    monkeypatch_module.setattr(dl, "_PROCESSED", data_dir / "processed")
    monkeypatch_module.setattr(dl, "_STORE", store)
    # Clear Streamlit cache between uses
    dl.list_match_ids.clear()
    dl.load_vaep.clear()
    dl.load_player_summary.clear()
    dl.load_team_summary.clear()
    dl.load_tactical_report.clear()
    dl.load_weaknesses.clear()
    dl.load_pressure.clear()
    return dl


# monkeypatch_module fixture (module scope workaround)
@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


# ── Data loader tests ─────────────────────────────────────────────────────────

class TestDataLoader:
    def test_list_match_ids_returns_list(self, patched_loader) -> None:
        ids = patched_loader.list_match_ids()
        assert isinstance(ids, list)
        assert _MATCH_ID in ids

    def test_load_vaep_returns_dataframe(self, patched_loader) -> None:
        df = patched_loader.load_vaep(_MATCH_ID)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_vaep_has_vaep_column(self, patched_loader) -> None:
        df = patched_loader.load_vaep(_MATCH_ID)
        assert Cols.VAEP_VALUE in df.columns

    def test_load_vaep_missing_returns_none(self, patched_loader) -> None:
        df = patched_loader.load_vaep("nonexistent_match_id")
        assert df is None

    def test_load_player_summary_returns_dataframe(self, patched_loader) -> None:
        df = patched_loader.load_player_summary(_MATCH_ID)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_load_player_summary_has_vaep_total(self, patched_loader) -> None:
        df = patched_loader.load_player_summary(_MATCH_ID)
        assert "vaep_total" in df.columns

    def test_load_team_summary_returns_dataframe(self, patched_loader) -> None:
        df = patched_loader.load_team_summary(_MATCH_ID)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_load_tactical_report_returns_dict(self, patched_loader) -> None:
        report = patched_loader.load_tactical_report(_MATCH_ID)
        assert isinstance(report, dict)
        assert "home_formation" in report

    def test_load_tactical_report_missing_returns_none(self, patched_loader) -> None:
        report = patched_loader.load_tactical_report("no_such_match")
        assert report is None

    def test_load_weaknesses_returns_dataframe(self, patched_loader) -> None:
        df = patched_loader.load_weaknesses(_MATCH_ID, _HOME_ID)
        assert df is None or isinstance(df, pd.DataFrame)

    def test_load_pressure_returns_dataframe(self, patched_loader) -> None:
        df = patched_loader.load_pressure(_MATCH_ID, _HOME_ID)
        assert df is None or isinstance(df, pd.DataFrame)

    def test_get_team_ids_returns_two(self, patched_loader) -> None:
        ids = patched_loader.get_team_ids(_MATCH_ID)
        assert len(ids) == 2

    def test_get_team_name_resolves(self, patched_loader) -> None:
        name = patched_loader.get_team_name(_MATCH_ID, _HOME_ID)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_team_name_unknown_returns_id(self, patched_loader) -> None:
        name = patched_loader.get_team_name(_MATCH_ID, "nonexistent_team")
        assert name == "nonexistent_team"

    def test_match_label_contains_vs(self, patched_loader) -> None:
        label = patched_loader.match_label(_MATCH_ID)
        assert "vs" in label or _MATCH_ID in label

    def test_pipeline_status_returns_dict(self, patched_loader) -> None:
        status = patched_loader.pipeline_status()
        assert isinstance(status, dict)
        assert "matches_processed" in status
        assert "vaep_computed" in status
        assert "tactical_computed" in status

    def test_pipeline_status_vaep_true(self, patched_loader) -> None:
        status = patched_loader.pipeline_status()
        assert status["vaep_computed"] is True

    def test_pipeline_status_tactical_true(self, patched_loader) -> None:
        status = patched_loader.pipeline_status()
        assert status["tactical_computed"] is True


# ── Page import health ────────────────────────────────────────────────────────

class TestPageImports:
    """Verify every page module imports without raising."""

    def test_overview_imports(self) -> None:
        from football_ai.dashboard.pages import overview
        assert hasattr(overview, "render")

    def test_match_analysis_imports(self) -> None:
        from football_ai.dashboard.pages import match_analysis
        assert hasattr(match_analysis, "render")

    def test_player_analysis_imports(self) -> None:
        from football_ai.dashboard.pages import player_analysis
        assert hasattr(player_analysis, "render")

    def test_rankings_imports(self) -> None:
        from football_ai.dashboard.pages import rankings
        assert hasattr(rankings, "render")

    def test_tactical_heatmap_imports(self) -> None:
        from football_ai.dashboard.pages import tactical_heatmap
        assert hasattr(tactical_heatmap, "render")

    def test_action_timeline_imports(self) -> None:
        from football_ai.dashboard.pages import action_timeline
        assert hasattr(action_timeline, "render")

    def test_passing_network_imports(self) -> None:
        from football_ai.dashboard.pages import passing_network
        assert hasattr(passing_network, "render")
        assert hasattr(passing_network, "_build_pass_network")
        assert hasattr(passing_network, "_draw_network")

    def test_style_module_imports(self) -> None:
        from football_ai.dashboard import style
        assert hasattr(style, "inject_css")
        assert hasattr(style, "kpi_card")
        assert hasattr(style, "section_header")
        assert hasattr(style, "match_banner")
        assert hasattr(style, "HOME_COLOR")
        assert hasattr(style, "AWAY_COLOR")

    def test_all_pages_in_package_init(self) -> None:
        from football_ai.dashboard import pages
        for name in [
            "overview", "match_analysis", "player_analysis",
            "rankings", "tactical_heatmap", "action_timeline",
            "passing_network", "xt_heatmap", "player_comparison",
            "formation_visualizer", "tactical_report",
        ]:
            assert hasattr(pages, name)


# ── Plotly figure builders ────────────────────────────────────────────────────

class TestPlotlyBuilders:
    """
    Call the private figure-builder functions directly (no Streamlit needed).
    """

    def _make_vaep_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(5)
        n   = 20
        return pd.DataFrame({
            Cols.MINUTE:        list(range(n)),
            Cols.TEAM_ID:       ([_HOME_ID] * 10) + ([_AWAY_ID] * 10),
            Cols.TEAM_NAME:     (["Home FC"] * 10) + (["Away FC"] * 10),
            Cols.PLAYER_NAME:   [f"P{i}" for i in range(n)],
            Cols.ACTION_TYPE:   [ActionType.PASS.value] * n,
            Cols.VAEP_VALUE:    rng.uniform(-0.3, 0.5, n),
            Cols.VAEP_OFFENSIVE:rng.uniform(0, 0.3, n),
            Cols.VAEP_DEFENSIVE:rng.uniform(0, 0.1, n),
            Cols.START_X:       rng.uniform(0, 120, n),
            Cols.START_Y:       rng.uniform(0, 80, n),
        })

    def test_vaep_timeline_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.action_timeline import _vaep_timeline
        df  = self._make_vaep_df()
        fig = _vaep_timeline(df, "Test timeline")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_pitch_scatter_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.action_timeline import _pitch_scatter
        df  = self._make_vaep_df()
        fig = _pitch_scatter(df, "Test scatter")
        assert isinstance(fig, go.Figure)

    def test_pitch_scatter_empty_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.action_timeline import _pitch_scatter
        fig = _pitch_scatter(pd.DataFrame(), "Empty")
        assert isinstance(fig, go.Figure)

    def test_weakness_map_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.tactical_heatmap import _weakness_map
        weak_df = pd.DataFrame({
            "zone_x":            [30.0, 60.0, 90.0],
            "zone_y":            [20.0, 40.0, 60.0],
            "risk_level":        ["critical", "high", "medium"],
            "mean_opponent_vaep":[0.15, 0.08, 0.03],
        })
        fig = _weakness_map(weak_df, "Test weakness")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_weakness_map_none_input_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.tactical_heatmap import _weakness_map
        fig = _weakness_map(None, "Empty weakness")
        assert isinstance(fig, go.Figure)

    def test_pressure_heatmap_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.tactical_heatmap import _pressure_heatmap
        press_df = pd.DataFrame({
            "zone_x": [30.0, 60.0, 90.0],
            "zone_y": [20.0, 40.0, 60.0],
            "threat": [0.10, 0.06, 0.03],
        })
        fig = _pressure_heatmap(press_df, "Home FC")
        assert isinstance(fig, go.Figure)

    def test_vaep_zone_heatmap_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.tactical_heatmap import _vaep_zone_heatmap
        df  = self._make_vaep_df()
        df[Cols.ZONE_ID] = list(range(20))
        fig = _vaep_zone_heatmap(df, _HOME_ID, "Home FC")
        assert isinstance(fig, go.Figure)

    def test_radar_chart_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.player_analysis import _radar_chart
        player_row = pd.Series({
            "player_name":    "Test Player",
            "vaep_total":     0.5,
            "vaep_offensive": 0.4,
            "vaep_defensive": 0.1,
            "xg_total":       0.3,
            "xt_total":       0.2,
        })
        team_avg = pd.Series({
            "vaep_total":     0.3,
            "vaep_offensive": 0.2,
            "vaep_defensive": 0.05,
            "xg_total":       0.15,
            "xt_total":       0.1,
        })
        fig = _radar_chart(player_row, team_avg)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2   # player + team avg


# ── ASE 1: new figure builders ────────────────────────────────────────────────

class TestASE1Builders:
    """
    Tests for the ASE 1 features:
        - Match Momentum graph  (_momentum_graph)
        - Comparison chart      (_comparison_chart)
        - Action distribution   (_action_distribution)
        - KPI metric helpers    (_possession_pct, _dangerous_attacks, etc.)
        - Passing network       (_build_pass_network, _draw_network)
        - Style helpers         (kpi_card, section_header)
    """

    _HOME_ID = _HOME_ID
    _AWAY_ID = _AWAY_ID

    def _make_vaep_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n   = 40
        return pd.DataFrame({
            Cols.MATCH_ID:       [_MATCH_ID] * n,
            Cols.INDEX:          list(range(n)),
            Cols.PERIOD:         [1] * n,
            Cols.MINUTE:         list(range(n)),
            Cols.SECOND:         [0] * n,
            Cols.TEAM_ID:        ([_HOME_ID] * 20) + ([_AWAY_ID] * 20),
            Cols.TEAM_NAME:      (["Home FC"] * 20) + (["Away FC"] * 20),
            Cols.PLAYER_ID:      [f"p{i % 4}" for i in range(n)],
            Cols.PLAYER_NAME:    [f"Player {i % 4}" for i in range(n)],
            Cols.ACTION_TYPE:    ["pass"] * n,
            Cols.START_X:        rng.uniform(0, 120, n),
            Cols.START_Y:        rng.uniform(0, 80, n),
            Cols.VAEP_VALUE:     rng.uniform(-0.3, 0.5, n),
            Cols.VAEP_OFFENSIVE: rng.uniform(0, 0.3, n),
            Cols.VAEP_DEFENSIVE: rng.uniform(0, 0.1, n),
            Cols.ZONE_ID:        rng.integers(0, 50, n),
            "under_pressure":    rng.integers(0, 2, n).astype(bool),
        })

    def _make_team_summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            Cols.TEAM_ID:    [_HOME_ID, _AWAY_ID],
            "xg_total":      [0.85, 0.42],
            "vaep_total":    [1.20, 0.75],
            "vaep_offensive":[0.90, 0.55],
            "vaep_defensive":[0.30, 0.20],
        })

    # ── match_analysis helpers ────────────────────────────────────────────────

    def test_possession_pct_sums_to_100(self) -> None:
        from football_ai.dashboard.pages.match_analysis import _possession_pct
        df  = self._make_vaep_df()
        hp  = _possession_pct(df, _HOME_ID)
        ap  = _possession_pct(df, _AWAY_ID)
        assert abs(hp + ap - 100.0) < 0.1

    def test_possession_pct_none_returns_50(self) -> None:
        from football_ai.dashboard.pages.match_analysis import _possession_pct
        assert _possession_pct(None, _HOME_ID) == 50.0  # type: ignore[arg-type]

    def test_dangerous_attacks_non_negative(self) -> None:
        from football_ai.dashboard.pages.match_analysis import _dangerous_attacks
        df  = self._make_vaep_df()
        da  = _dangerous_attacks(df, _HOME_ID)
        assert da >= 0

    def test_dangerous_attacks_final_third_only(self) -> None:
        from football_ai.dashboard.pages.match_analysis import _dangerous_attacks
        # All x < 80 → zero dangerous attacks
        df = self._make_vaep_df()
        df[Cols.START_X] = 50.0
        assert _dangerous_attacks(df, _HOME_ID) == 0

    def test_pressure_intensity_returns_string(self) -> None:
        from football_ai.dashboard.pages.match_analysis import _pressure_intensity
        df  = self._make_vaep_df()
        val = _pressure_intensity(df, _HOME_ID)
        assert val.endswith("%") or val == "N/A"

    def test_most_dangerous_zone_returns_string(self) -> None:
        from football_ai.dashboard.pages.match_analysis import _most_dangerous_zone
        df  = self._make_vaep_df()
        val = _most_dangerous_zone(df, _HOME_ID)
        assert isinstance(val, str)

    def test_most_dangerous_zone_none_df(self) -> None:
        from football_ai.dashboard.pages.match_analysis import _most_dangerous_zone
        assert _most_dangerous_zone(None, _HOME_ID) == "N/A"  # type: ignore[arg-type]

    def test_top_player_name_returns_string(self) -> None:
        from football_ai.dashboard.pages.match_analysis import _top_player_name
        ps  = pd.DataFrame({
            Cols.TEAM_ID:   [_HOME_ID, _HOME_ID],
            "player_name":  ["Alice", "Bob"],
            "vaep_total":   [1.5, 0.8],
        })
        name = _top_player_name(ps, _HOME_ID)
        assert name == "Alice"

    # ── Momentum graph ────────────────────────────────────────────────────────

    def test_momentum_graph_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.match_analysis import _momentum_graph
        df  = self._make_vaep_df()
        fig = _momentum_graph(df, _HOME_ID, _AWAY_ID, "Home FC", "Away FC")
        assert isinstance(fig, go.Figure)
        # Expect bars for home + away and cumulative lines = 4 traces
        assert len(fig.data) == 4

    def test_momentum_graph_none_df(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.match_analysis import _momentum_graph
        fig = _momentum_graph(None, _HOME_ID, _AWAY_ID, "H", "A")  # type: ignore[arg-type]
        assert isinstance(fig, go.Figure)

    def test_momentum_graph_has_halftime_annotation(self) -> None:
        from football_ai.dashboard.pages.match_analysis import _momentum_graph
        df  = self._make_vaep_df()
        fig = _momentum_graph(df, _HOME_ID, _AWAY_ID, "Home FC", "Away FC")
        assert any(a.text == "HT" for a in fig.layout.annotations)

    # ── Comparison chart ──────────────────────────────────────────────────────

    def test_comparison_chart_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.match_analysis import _comparison_chart
        ts  = self._make_team_summary()
        fig = _comparison_chart(ts, _HOME_ID, _AWAY_ID, "Home FC", "Away FC")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # home bar + away bar

    def test_comparison_chart_none_summary(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.match_analysis import _comparison_chart
        fig = _comparison_chart(None, _HOME_ID, _AWAY_ID, "H", "A")
        assert isinstance(fig, go.Figure)

    # ── Action distribution ───────────────────────────────────────────────────

    def test_action_distribution_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.match_analysis import _action_distribution
        df  = self._make_vaep_df()
        fig = _action_distribution(df, _HOME_ID, "Home FC", "#1565C0")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    # ── KPI card helper ───────────────────────────────────────────────────────

    def test_kpi_card_returns_html_string(self) -> None:
        from football_ai.dashboard.style import kpi_card
        html = kpi_card("xG Diff", "+0.42", delta="Home leads", delta_positive=True)
        assert "xG Diff" in html
        assert "+0.42" in html
        assert "positive" in html
        assert "kpi-card" in html

    def test_kpi_card_negative_delta(self) -> None:
        from football_ai.dashboard.style import kpi_card
        html = kpi_card("VAEP", "-0.10", delta="Away leads", delta_positive=False)
        assert "negative" in html

    def test_kpi_card_neutral_delta(self) -> None:
        from football_ai.dashboard.style import kpi_card
        # delta text must be non-empty for the delta div to render
        html = kpi_card("Poss", "50% — 50%", delta="equal", delta_positive=None)
        assert "neutral" in html
        # without delta text, no delta div is produced
        html_no_delta = kpi_card("Poss", "50%")
        assert "kpi-delta" not in html_no_delta

    # ── Passing network ───────────────────────────────────────────────────────

    def test_build_pass_network_returns_dicts(self) -> None:
        from football_ai.dashboard.pages.passing_network import _build_pass_network
        df  = self._make_vaep_df()
        pos, edges = _build_pass_network(df, _HOME_ID, min_passes=1)
        assert isinstance(pos, dict)
        assert isinstance(edges, dict)

    def test_build_pass_network_positions_in_pitch_bounds(self) -> None:
        from football_ai.dashboard.pages.passing_network import _build_pass_network
        df  = self._make_vaep_df()
        pos, _ = _build_pass_network(df, _HOME_ID, min_passes=1)
        for player, (x, y) in pos.items():
            assert 0.0 <= x <= 120.0, f"{player} x={x} out of pitch bounds"
            assert 0.0 <= y <= 80.0,  f"{player} y={y} out of pitch bounds"

    def test_build_pass_network_none_df(self) -> None:
        from football_ai.dashboard.pages.passing_network import _build_pass_network
        pos, edges = _build_pass_network(None, _HOME_ID)  # type: ignore[arg-type]
        assert pos == {}
        assert edges == {}

    def test_build_pass_network_min_passes_filters(self) -> None:
        from football_ai.dashboard.pages.passing_network import _build_pass_network
        df  = self._make_vaep_df()
        _, edges_low  = _build_pass_network(df, _HOME_ID, min_passes=1)
        _, edges_high = _build_pass_network(df, _HOME_ID, min_passes=999)
        assert len(edges_low) >= len(edges_high)
        assert len(edges_high) == 0

    def test_draw_network_empty_positions(self) -> None:
        import matplotlib.pyplot as plt

        from football_ai.dashboard.pages.passing_network import _draw_network
        fig = _draw_network({}, {}, "Test Team", "#1565C0")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_network_with_data(self) -> None:
        import matplotlib.pyplot as plt

        from football_ai.dashboard.pages.passing_network import _build_pass_network, _draw_network
        df  = self._make_vaep_df()
        pos, edges = _build_pass_network(df, _HOME_ID, min_passes=1)
        fig = _draw_network(pos, edges, "Home FC", "#1565C0")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ── Phase 2: xT Heatmap ───────────────────────────────────────────────────────

class TestXTHeatmap:
    """Tests for xt_heatmap page builders."""

    def _make_vaep_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        n   = 30
        return pd.DataFrame({
            Cols.TEAM_ID:   ([_HOME_ID] * 15) + ([_AWAY_ID] * 15),
            Cols.PLAYER_NAME: [f"P{i}" for i in range(n)],
            Cols.START_X:   rng.uniform(0, 120, n),
            Cols.START_Y:   rng.uniform(0, 80, n),
            Cols.XT_DELTA:  rng.uniform(-0.05, 0.1, n),
            Cols.ACTION_TYPE: ["pass"] * n,
            Cols.MINUTE:    list(range(n)),
        })

    def test_build_xt_grid_returns_arrays(self) -> None:
        from football_ai.dashboard.pages.xt_heatmap import _build_xt_grid
        df = self._make_vaep_df()
        zi, xi, yi = _build_xt_grid(df, _HOME_ID)
        assert zi.shape == (8, 12)
        assert len(xi) == 12
        assert len(yi) == 8

    def test_build_xt_grid_none_df(self) -> None:
        from football_ai.dashboard.pages.xt_heatmap import _build_xt_grid
        zi, xi, yi = _build_xt_grid(None, _HOME_ID)  # type: ignore[arg-type]
        assert zi.shape == (8, 12)
        assert (zi == 0).all()

    def test_draw_xt_heatmap_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from football_ai.dashboard.pages.xt_heatmap import _draw_xt_heatmap
        df  = self._make_vaep_df()
        fig = _draw_xt_heatmap(df, _HOME_ID, "Home FC", "#1565C0")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_xt_heatmap_empty_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from football_ai.dashboard.pages.xt_heatmap import _draw_xt_heatmap
        fig = _draw_xt_heatmap(None, _HOME_ID, "Home FC", "#1565C0")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_xt_summary_stats_totals(self) -> None:
        from football_ai.dashboard.pages.xt_heatmap import _xt_summary_stats
        df    = self._make_vaep_df()
        stats = _xt_summary_stats(df, _HOME_ID)
        assert isinstance(stats["total"], float)
        assert isinstance(stats["positive"], float)
        assert stats["positive"] >= 0

    def test_xt_summary_stats_none_df(self) -> None:
        from football_ai.dashboard.pages.xt_heatmap import _xt_summary_stats
        stats = _xt_summary_stats(None, _HOME_ID)  # type: ignore[arg-type]
        assert stats["total"] == 0.0

    def test_xt_heatmap_imports(self) -> None:
        from football_ai.dashboard.pages import xt_heatmap
        assert hasattr(xt_heatmap, "render")
        assert hasattr(xt_heatmap, "_build_xt_grid")
        assert hasattr(xt_heatmap, "_draw_xt_heatmap")


# ── Phase 2: Player Comparison ────────────────────────────────────────────────

class TestPlayerComparison:
    """Tests for player_comparison page builders."""

    def _make_player_rows(self) -> tuple[pd.Series, pd.Series]:
        row_a = pd.Series({
            "player_name":    "Alice",
            "vaep_total":     0.8,
            "vaep_offensive": 0.6,
            "vaep_defensive": 0.2,
            "xg_total":       0.4,
            "xt_total":       0.3,
            "action_count":   25,
            "vaep_per_90":    1.1,
            "minutes_played": 90,
        })
        row_b = pd.Series({
            "player_name":    "Bob",
            "vaep_total":     0.5,
            "vaep_offensive": 0.3,
            "vaep_defensive": 0.2,
            "xg_total":       0.2,
            "xt_total":       0.1,
            "action_count":   18,
            "vaep_per_90":    0.7,
            "minutes_played": 78,
        })
        return row_a, row_b

    def test_comparison_radar_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.player_comparison import _comparison_radar
        row_a, row_b = self._make_player_rows()
        fig = _comparison_radar(row_a, row_b, "Alice", "Bob")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2   # one trace per player

    def test_comparison_bar_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.player_comparison import _comparison_bar
        row_a, row_b = self._make_player_rows()
        fig = _comparison_bar(row_a, row_b, "Alice", "Bob")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_vaep_timeline_compare_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.player_comparison import _vaep_timeline_compare
        rng = np.random.default_rng(9)
        n   = 20
        df  = pd.DataFrame({
            Cols.PLAYER_ID:  (["pa"] * 10) + (["pb"] * 10),
            Cols.MINUTE:     list(range(n)),
            Cols.VAEP_VALUE: rng.uniform(-0.1, 0.3, n),
        })
        fig = _vaep_timeline_compare(df, "pa", "pb", "Alice", "Bob")
        assert isinstance(fig, go.Figure)

    def test_vaep_timeline_compare_none_df(self) -> None:
        from football_ai.dashboard.pages.player_comparison import _vaep_timeline_compare
        result = _vaep_timeline_compare(None, "pa", "pb", "A", "B")  # type: ignore[arg-type]
        assert result is None

    def test_player_comparison_imports(self) -> None:
        from football_ai.dashboard.pages import player_comparison
        assert hasattr(player_comparison, "render")
        assert hasattr(player_comparison, "_comparison_radar")
        assert hasattr(player_comparison, "_comparison_bar")


# ── Phase 2: Formation Visualizer ─────────────────────────────────────────────

class TestFormationVisualizer:
    """Tests for formation_visualizer page builders."""

    def _make_vaep_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        n   = 40
        return pd.DataFrame({
            Cols.TEAM_ID:    ([_HOME_ID] * 20) + ([_AWAY_ID] * 20),
            Cols.PLAYER_NAME: [f"Player{i % 5}" for i in range(n)],
            Cols.START_X:    rng.uniform(0, 120, n),
            Cols.START_Y:    rng.uniform(0, 80, n),
            Cols.ACTION_TYPE: ["pass"] * n,
        })

    def test_player_avg_positions_returns_dict(self) -> None:
        from football_ai.dashboard.pages.formation_visualizer import _player_avg_positions
        df  = self._make_vaep_df()
        pos = _player_avg_positions(df, _HOME_ID)
        assert isinstance(pos, dict)

    def test_player_avg_positions_in_pitch_bounds(self) -> None:
        from football_ai.dashboard.pages.formation_visualizer import _player_avg_positions
        df  = self._make_vaep_df()
        pos = _player_avg_positions(df, _HOME_ID)
        for name, (x, y) in pos.items():
            assert 0.0 <= x <= 120.0, f"{name} x={x} out of bounds"
            assert 0.0 <= y <= 80.0,  f"{name} y={y} out of bounds"

    def test_player_avg_positions_none_df(self) -> None:
        from football_ai.dashboard.pages.formation_visualizer import _player_avg_positions
        pos = _player_avg_positions(None, _HOME_ID)  # type: ignore[arg-type]
        assert pos == {}

    def test_draw_formation_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from football_ai.dashboard.pages.formation_visualizer import (
            _draw_formation,
            _player_avg_positions,
        )
        df  = self._make_vaep_df()
        pos = _player_avg_positions(df, _HOME_ID)
        fig = _draw_formation(pos, "Home FC", "4-3-3", "#1565C0", flip=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_formation_empty_positions(self) -> None:
        import matplotlib.pyplot as plt

        from football_ai.dashboard.pages.formation_visualizer import _draw_formation
        fig = _draw_formation({}, "Home FC", "4-4-2", "#1565C0")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_formation_stats_returns_dict(self) -> None:
        from football_ai.dashboard.pages.formation_visualizer import _formation_stats
        df    = self._make_vaep_df()
        stats = _formation_stats(df, _HOME_ID)
        assert "total" in stats
        assert "passes" in stats
        assert stats["total"] >= 0

    def test_formation_visualizer_imports(self) -> None:
        from football_ai.dashboard.pages import formation_visualizer
        assert hasattr(formation_visualizer, "render")
        assert hasattr(formation_visualizer, "_player_avg_positions")
        assert hasattr(formation_visualizer, "_draw_formation")


# ── Phase 2: Tactical Report ──────────────────────────────────────────────────

class TestTacticalReport:
    """Tests for tactical_report page builders."""

    def _make_report(self) -> dict:
        return {
            "match_id":        _MATCH_ID,
            "home_team_id":    _HOME_ID,
            "away_team_id":    _AWAY_ID,
            "home_team_name":  "Home FC",
            "away_team_name":  "Away FC",
            "home_formation":  "4-3-3",
            "away_formation":  "4-4-2",
            "xg_home":         1.2,
            "xg_away":         0.7,
            "vaep_home":       2.1,
            "vaep_away":       1.4,
            "total_actions":   300,
            "top_players": [
                {"player_name": "Alice", "team_name": "Home FC",
                 "vaep_total": 0.9, "vaep_offensive": 0.7, "vaep_defensive": 0.2},
                {"player_name": "Bob",   "team_name": "Away FC",
                 "vaep_total": 0.6, "vaep_offensive": 0.4, "vaep_defensive": 0.2},
            ],
            "most_valuable_actions": [
                {"minute": 35, "player_name": "Alice", "team_name": "Home FC",
                 "action_type": "shot", "vaep_value": 0.42, "start_x": 105.0, "start_y": 40.0},
            ],
            "home_report": {
                "team_id": _HOME_ID,
                "n_critical_zones": 1,
                "n_high_risk_zones": 2,
                "weakest_zone_x": 90.0,
                "weakest_zone_y": 20.0,
                "formation": "4-3-3",
            },
            "away_report": {
                "team_id": _AWAY_ID,
                "n_critical_zones": 0,
                "n_high_risk_zones": 1,
                "weakest_zone_x": 30.0,
                "weakest_zone_y": 60.0,
                "formation": "4-4-2",
            },
        }

    def test_insight_bullets_returns_list(self) -> None:
        from football_ai.dashboard.pages.tactical_report import _insight_bullets
        report  = self._make_report()
        bullets = _insight_bullets(report, "Home FC", "Away FC", _HOME_ID, _AWAY_ID)
        assert isinstance(bullets, list)
        assert len(bullets) > 0
        assert all(isinstance(b, str) for b in bullets)

    def test_insight_bullets_mentions_teams(self) -> None:
        from football_ai.dashboard.pages.tactical_report import _insight_bullets
        report  = self._make_report()
        bullets = _insight_bullets(report, "Home FC", "Away FC", _HOME_ID, _AWAY_ID)
        combined = " ".join(bullets)
        assert "Home FC" in combined or "Away FC" in combined

    def test_top_players_fig_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.tactical_report import _top_players_fig
        report  = self._make_report()
        fig     = _top_players_fig(report["top_players"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1   # single horizontal bar trace

    def test_top_players_fig_empty_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.tactical_report import _top_players_fig
        fig = _top_players_fig([])
        assert isinstance(fig, go.Figure)

    def test_conclusions_returns_list(self) -> None:
        from football_ai.dashboard.pages.tactical_report import _conclusions
        report = self._make_report()
        concs  = _conclusions(report, "Home FC", "Away FC")
        assert isinstance(concs, list)
        assert len(concs) > 0

    def test_tactical_report_imports(self) -> None:
        from football_ai.dashboard.pages import tactical_report
        assert hasattr(tactical_report, "render")
        assert hasattr(tactical_report, "_insight_bullets")
        assert hasattr(tactical_report, "_top_players_fig")
        assert hasattr(tactical_report, "_conclusions")


# ── Phase 2: Updated page-import health ───────────────────────────────────────

class TestPhase2PageImports:
    """Verify all Phase 2 page modules import without raising."""

    def test_xt_heatmap_page_imports(self) -> None:
        from football_ai.dashboard.pages import xt_heatmap
        assert hasattr(xt_heatmap, "render")

    def test_player_comparison_page_imports(self) -> None:
        from football_ai.dashboard.pages import player_comparison
        assert hasattr(player_comparison, "render")

    def test_formation_visualizer_page_imports(self) -> None:
        from football_ai.dashboard.pages import formation_visualizer
        assert hasattr(formation_visualizer, "render")

    def test_tactical_report_page_imports(self) -> None:
        from football_ai.dashboard.pages import tactical_report
        assert hasattr(tactical_report, "render")

    def test_all_phase2_pages_in_package_init(self) -> None:
        from football_ai.dashboard import pages
        for name in [
            "xt_heatmap", "player_comparison",
            "formation_visualizer", "tactical_report",
        ]:
            assert hasattr(pages, name), f"pages.{name} not found in __init__"


# ── Demo Mode ─────────────────────────────────────────────────────────────────

class TestDemoMode:
    """Tests for the Professional Demo Mode page."""

    def _make_player_summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            Cols.PLAYER_ID:      ["p0", "p1", "p2"],
            "player_name":       ["Alice", "Bob", "Carol"],
            Cols.TEAM_ID:        [_HOME_ID, _HOME_ID, _AWAY_ID],
            "team_name":         ["Home FC", "Home FC", "Away FC"],
            "action_count":      [30, 20, 25],
            "vaep_total":        [0.9, 0.5, 0.7],
            "vaep_offensive":    [0.7, 0.4, 0.5],
            "vaep_defensive":    [0.2, 0.1, 0.2],
            "xg_total":          [0.3, 0.2, 0.1],
        })

    def _make_team_summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            Cols.TEAM_ID:    [_HOME_ID, _AWAY_ID],
            "team_name":     ["Home FC", "Away FC"],
            "xg_total":      [1.2, 0.6],
            "vaep_total":    [2.0, 1.4],
            "vaep_offensive":[1.5, 1.0],
            "vaep_defensive":[0.5, 0.4],
        })

    def _make_vaep_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(13)
        n   = 30
        return pd.DataFrame({
            Cols.TEAM_ID:    ([_HOME_ID] * 15) + ([_AWAY_ID] * 15),
            Cols.PLAYER_ID:  [f"p{i % 3}" for i in range(n)],
            Cols.MINUTE:     list(range(n)),
            Cols.VAEP_VALUE: rng.uniform(-0.2, 0.4, n),
        })

    def test_demo_kpi_returns_html(self) -> None:
        from football_ai.dashboard.pages.demo_mode import _demo_kpi
        html = _demo_kpi("xG Advantage", "Home FC", sub="+0.6 xG", accent="#1565C0")
        assert "xG Advantage" in html
        assert "Home FC" in html
        assert "demo-kpi" in html

    def test_build_kpi_html_returns_string(self) -> None:
        from football_ai.dashboard.pages.demo_mode import _build_kpi_html
        vaep_df = self._make_vaep_df()
        ts      = self._make_team_summary()
        ps      = self._make_player_summary()
        html    = _build_kpi_html(vaep_df, ts, ps, _HOME_ID, _AWAY_ID, "Home FC", "Away FC")
        assert isinstance(html, str)
        assert "demo-kpi-strip" in html

    def test_build_kpi_html_none_inputs(self) -> None:
        from football_ai.dashboard.pages.demo_mode import _build_kpi_html
        html = _build_kpi_html(None, None, None, _HOME_ID, _AWAY_ID, "H", "A")
        assert isinstance(html, str)
        assert "demo-kpi-strip" in html

    def test_demo_momentum_returns_figure(self) -> None:
        import plotly.graph_objects as go

        from football_ai.dashboard.pages.demo_mode import _demo_momentum
        df  = self._make_vaep_df()
        fig = _demo_momentum(df, _HOME_ID, _AWAY_ID, "Home FC", "Away FC")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4

    def test_demo_top_players_returns_dataframe(self) -> None:
        from football_ai.dashboard.pages.demo_mode import _demo_top_players
        ps  = self._make_player_summary()
        top = _demo_top_players(ps, n=3)
        assert isinstance(top, pd.DataFrame)
        assert len(top) == 3
        assert "Rank" in top.columns
        assert "Player" in top.columns

    def test_demo_top_players_none_returns_empty(self) -> None:
        from football_ai.dashboard.pages.demo_mode import _demo_top_players
        top = _demo_top_players(None)
        assert isinstance(top, pd.DataFrame)
        assert top.empty

    def test_demo_top_players_sorted_by_vaep(self) -> None:
        from football_ai.dashboard.pages.demo_mode import _demo_top_players
        ps  = self._make_player_summary()
        top = _demo_top_players(ps, n=3)
        vaep_col = top["VAEP Total"].tolist()
        assert vaep_col == sorted(vaep_col, reverse=True)

    def test_demo_mode_imports(self) -> None:
        from football_ai.dashboard.pages import demo_mode
        assert hasattr(demo_mode, "render")
        assert hasattr(demo_mode, "_demo_kpi")
        assert hasattr(demo_mode, "_build_kpi_html")
        assert hasattr(demo_mode, "_demo_momentum")
        assert hasattr(demo_mode, "_demo_top_players")

    def test_demo_mode_in_package_init(self) -> None:
        from football_ai.dashboard import pages
        assert hasattr(pages, "demo_mode")
