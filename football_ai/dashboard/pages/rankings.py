"""
Rankings page.

Ranked player tables across multiple metrics with interactive
bar chart and sortable table.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from football_ai.constants import Cols
from football_ai.dashboard.data_loader import (
    list_match_ids,
    load_player_summary,
    match_label,
)

_METRICS = {
    "VAEP total":     "vaep_total",
    "Offensive VAEP": "vaep_offensive",
    "Defensive VAEP": "vaep_defensive",
    "VAEP per 90":    "vaep_per_90",
    "xG total":       "xg_total",
    "xT total":       "xt_total",
}


def render() -> None:
    st.title("🏆 Player Rankings")

    match_ids = list_match_ids()
    if not match_ids:
        st.warning("No processed matches found. Run the pipeline first.", icon="⚠️")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        match_id = st.selectbox("Match", match_ids, format_func=match_label)
    with col2:
        metric_label = st.selectbox("Rank by", list(_METRICS.keys()))
    with col3:
        top_n = st.number_input("Top N", min_value=5, max_value=50, value=15, step=5)

    metric_col = _METRICS[metric_label]

    summary = load_player_summary(match_id)
    if summary is None or summary.empty:
        st.error("Player summary not found. Run Phase 6 first.")
        return

    if metric_col not in summary.columns:
        st.error(f"Metric '{metric_col}' not available in this dataset.")
        return

    # ── Build ranking ─────────────────────────────────────────────────────────
    ranked = (
        summary
        .dropna(subset=[metric_col])
        .sort_values(metric_col, ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )
    ranked.index = ranked.index + 1
    ranked.index.name = "Rank"

    # ── Summary metrics ───────────────────────────────────────────────────────
    if not ranked.empty:
        best_player = ranked.iloc[0].get("player_name", "—")
        best_val    = ranked.iloc[0].get(metric_col, 0.0)
        mean_val    = ranked[metric_col].mean()

        m1, m2, m3 = st.columns(3)
        m1.metric("Top player",  best_player)
        m2.metric(f"Top {metric_label}", round(float(best_val), 4))
        m3.metric(f"Mean {metric_label} (top {top_n})", round(float(mean_val), 4))

    st.divider()

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.subheader(f"Top {top_n} — {metric_label}")

    colors = [
        "#FFD700" if i == 0 else "#94A3B8" if i == 1 else "#CD7F32" if i == 2
        else "#00E5A0"
        for i in range(len(ranked))
    ]

    fig = go.Figure(go.Bar(
        x=ranked["player_name"].tolist()[::-1],
        y=ranked[metric_col].tolist()[::-1],
        orientation="v",
        marker_color=colors[::-1],
        text=[f"{v:.4f}" for v in ranked[metric_col].tolist()[::-1]],
        textposition="outside",
        textfont=dict(
            color="#E2E8F0", size=10,
            family="JetBrains Mono, monospace",
        ),
    ))
    fig.update_layout(
        xaxis_title="",
        yaxis_title=metric_label,
        height=420,
        margin=dict(t=20, b=80),
        plot_bgcolor="#0A0E1A",
        paper_bgcolor="#0A0E1A",
        font=dict(color="#E2E8F0", family="Barlow Condensed, sans-serif"),
        xaxis=dict(
            gridcolor="#1E293B",
            tickfont=dict(family="Barlow Condensed, sans-serif", size=10, color="#64748B"),
        ),
        yaxis=dict(
            gridcolor="#1E293B",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#64748B"),
        ),
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # ── Full table ────────────────────────────────────────────────────────────
    st.subheader("Full ranking table")
    display_cols = [
        c for c in [
            "player_name", "team_name", "action_count",
            "vaep_total", "vaep_offensive", "vaep_defensive",
            "vaep_per_90", "xg_total", "xt_total",
        ] if c in ranked.columns
    ]
    display = ranked[display_cols].copy()
    for col in ["vaep_total", "vaep_offensive", "vaep_defensive",
                "vaep_per_90", "xg_total", "xt_total"]:
        if col in display.columns:
            display[col] = display[col].round(4)

    st.dataframe(display, width="stretch")

    # ── Team comparison ───────────────────────────────────────────────────────
    if Cols.TEAM_NAME in summary.columns and metric_col in summary.columns:
        st.divider()
        st.subheader("Team comparison")
        team_agg = (
            summary.groupby(Cols.TEAM_NAME)[metric_col]
            .agg(["mean", "sum", "max"])
            .round(4)
            .rename(columns={"mean": "Mean", "sum": "Total", "max": "Best player"})
        )
        st.dataframe(team_agg, width="stretch")
