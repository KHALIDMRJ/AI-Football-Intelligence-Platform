"""
Player Analysis page.

Individual player VAEP profile with radar chart, top actions table,
and comparison against team average.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_ai.constants import Cols
from football_ai.dashboard.data_loader import (
    list_match_ids,
    load_player_summary,
    load_vaep,
    match_label,
)


def _radar_chart(player_row: pd.Series, team_avg: pd.Series) -> go.Figure:
    """Build a radar chart comparing a player to their team average."""
    categories = ["VAEP total", "Offensive", "Defensive", "xG", "xT"]
    fields     = ["vaep_total", "vaep_offensive", "vaep_defensive", "xg_total", "xt_total"]

    def _safe(row: pd.Series, field: str) -> float:
        v = row.get(field, 0.0)
        return float(v) if pd.notna(v) else 0.0

    player_vals = [_safe(player_row, f) for f in fields]
    avg_vals    = [_safe(team_avg,   f) for f in fields]

    # Normalise so radar is readable regardless of scale
    max_vals = [max(abs(p), abs(a), 1e-9) for p, a in zip(player_vals, avg_vals)]
    p_norm   = [v / m for v, m in zip(player_vals, max_vals)]
    a_norm   = [v / m for v, m in zip(avg_vals,    max_vals)]

    cats = categories + [categories[0]]
    p_norm_closed = p_norm + [p_norm[0]]
    a_norm_closed = a_norm + [a_norm[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=p_norm_closed, theta=cats, fill="toself",
        name=str(player_row.get("player_name", "Player")),
        line_color="#00E5A0", fillcolor="rgba(0,229,160,0.12)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=a_norm_closed, theta=cats, fill="toself",
        name="Team average",
        line_color="#00B8D4", fillcolor="rgba(0,184,212,0.08)",
        line_dash="dot",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1E293B", linecolor="#1E293B"),
            angularaxis=dict(gridcolor="#1E293B", linecolor="#1E293B"),
            bgcolor="#0A0E1A",
        ),
        showlegend=True,
        height=380,
        margin=dict(t=30, b=30, l=30, r=30),
        paper_bgcolor="#0A0E1A",
        font=dict(color="#E2E8F0", family="Barlow Condensed, sans-serif"),
    )
    return fig


def render() -> None:
    st.title("👤 Player Analysis")

    match_ids = list_match_ids()
    if not match_ids:
        st.warning("No processed matches found. Run the pipeline first.", icon="⚠️")
        return

    match_id = st.selectbox("Select match", match_ids, format_func=match_label)
    summary  = load_player_summary(match_id)

    if summary is None or summary.empty:
        st.error("Player summary not found. Run Phase 6 first.")
        return

    # ── Player selector ───────────────────────────────────────────────────────
    player_names = summary["player_name"].tolist() if "player_name" in summary.columns else []
    if not player_names:
        st.error("No player data found.")
        return

    selected_name = st.selectbox("Select player", player_names)
    player_row    = summary[summary["player_name"] == selected_name].iloc[0]
    team_id       = str(player_row.get(Cols.TEAM_ID, ""))
    team_mask     = summary[Cols.TEAM_ID].astype(str) == team_id
    team_avg      = summary[team_mask][
        ["vaep_total", "vaep_offensive", "vaep_defensive", "xg_total", "xt_total"]
    ].mean()

    # ── Header metrics ────────────────────────────────────────────────────────
    st.subheader(
        f"{selected_name}  —  {player_row.get(Cols.TEAM_NAME, '')}  "
        f"({player_row.get('position', 'N/A')})"
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Actions",     int(player_row.get("action_count", 0)))
    m2.metric("VAEP total",  round(float(player_row.get("vaep_total", 0.0)), 4))
    m3.metric("Off. VAEP",   round(float(player_row.get("vaep_offensive", 0.0)), 4))
    m4.metric("Def. VAEP",   round(float(player_row.get("vaep_defensive", 0.0)), 4))

    per90 = player_row.get("vaep_per_90")
    m5.metric("VAEP/90", round(float(per90), 4) if pd.notna(per90) else "N/A")

    st.divider()

    # ── Radar chart ───────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Player profile vs team average")
        fig = _radar_chart(player_row, team_avg)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Full stats")
        stat_rows = [
            ("xG total",          round(float(player_row.get("xg_total",  0.0)), 4)),
            ("xT total",          round(float(player_row.get("xt_total",  0.0)), 4)),
            ("Minutes played",    round(float(player_row.get("minutes_played", 0.0)), 1)),
        ]
        for label, value in stat_rows:
            c1, c2 = st.columns([2, 1])
            c1.write(label)
            c2.write(f"**{value}**")

    st.divider()

    # ── Top actions ───────────────────────────────────────────────────────────
    st.subheader(f"Top 10 actions by {selected_name}")
    vaep_df = load_vaep(match_id)
    if vaep_df is not None and Cols.PLAYER_NAME in vaep_df.columns:
        pid = str(player_row.get(Cols.PLAYER_ID, ""))
        player_actions = vaep_df[vaep_df[Cols.PLAYER_ID].astype(str) == pid]
        if not player_actions.empty and Cols.VAEP_VALUE in player_actions.columns:
            display_cols = [
                c for c in [
                    Cols.MINUTE, Cols.ACTION_TYPE, Cols.RESULT,
                    Cols.VAEP_VALUE, Cols.VAEP_OFFENSIVE, Cols.VAEP_DEFENSIVE,
                    Cols.START_X, Cols.START_Y,
                ] if c in player_actions.columns
            ]
            top = player_actions.nlargest(10, Cols.VAEP_VALUE)[display_cols].copy()
            for col in [Cols.VAEP_VALUE, Cols.VAEP_OFFENSIVE, Cols.VAEP_DEFENSIVE]:
                if col in top.columns:
                    top[col] = top[col].round(4)
            st.dataframe(top.reset_index(drop=True), width="stretch", hide_index=True)

    st.divider()

    # ── VAEP over match time ──────────────────────────────────────────────────
    st.subheader("VAEP accumulation over match")
    if vaep_df is not None:
        pid = str(player_row.get(Cols.PLAYER_ID, ""))
        pa  = vaep_df[vaep_df[Cols.PLAYER_ID].astype(str) == pid].copy()
        if not pa.empty and Cols.VAEP_VALUE in pa.columns and Cols.MINUTE in pa.columns:
            pa = pa.sort_values(Cols.MINUTE)
            pa["cumulative_vaep"] = pa[Cols.VAEP_VALUE].cumsum()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=pa[Cols.MINUTE].tolist(),
                y=pa["cumulative_vaep"].tolist(),
                mode="lines+markers",
                line=dict(color="#00E5A0", width=2),
                marker=dict(size=5),
                name="Cumulative VAEP",
            ))
            fig2.add_hline(y=0, line_dash="dash", line_color="#334155", line_width=1)
            fig2.update_layout(
                xaxis_title="Match minute",
                yaxis_title="Cumulative VAEP",
                height=300,
                margin=dict(t=20, b=30),
                plot_bgcolor="#0A0E1A",
                paper_bgcolor="#0A0E1A",
                font=dict(color="#E2E8F0", family="Barlow Condensed, sans-serif"),
                xaxis=dict(
                    gridcolor="#1E293B",
                    tickfont=dict(
                        family="JetBrains Mono, monospace", size=10, color="#64748B",
                    ),
                ),
                yaxis=dict(
                    gridcolor="#1E293B",
                    tickfont=dict(
                        family="JetBrains Mono, monospace", size=10, color="#64748B",
                    ),
                ),
            )
            st.plotly_chart(fig2, width="stretch")
