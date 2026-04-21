"""
Player Comparison Tool page.

Compare any two players side-by-side:
  - Dual radar chart on one Plotly figure
  - Metric cards for each player
  - Delta comparison table
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
from football_ai.dashboard.style import (
    AWAY_COLOR,
    BG_PRIMARY,
    GRID_COLOR,
    HOME_COLOR,
    TEXT_MUTED,
    TEXT_PRIMARY,
    inject_css,
    kpi_card,
    section_header,
)

# ── Constants ─────────────────────────────────────────────────────────────────
_RADAR_FIELDS      = ["vaep_total", "vaep_offensive", "vaep_defensive", "xg_total", "xt_total"]
_RADAR_LABELS      = ["VAEP Total", "Offensive", "Defensive", "xG", "xT"]
_COMPARISON_FIELDS = [
    ("action_count",    "Actions"),
    ("vaep_total",      "VAEP Total"),
    ("vaep_offensive",  "Off. VAEP"),
    ("vaep_defensive",  "Def. VAEP"),
    ("vaep_per_90",     "VAEP / 90"),
    ("xg_total",        "xG Total"),
    ("xt_total",        "xT Total"),
    ("minutes_played",  "Minutes"),
]


# ── Figure builders ───────────────────────────────────────────────────────────

def _comparison_radar(
    row_a: pd.Series,
    row_b: pd.Series,
    name_a: str,
    name_b: str,
) -> go.Figure:
    """Dual radar chart comparing two players on normalised metrics."""

    def _safe(row: pd.Series, field: str) -> float:
        v = row.get(field, 0.0)
        return float(v) if pd.notna(v) else 0.0

    vals_a = [_safe(row_a, f) for f in _RADAR_FIELDS]
    vals_b = [_safe(row_b, f) for f in _RADAR_FIELDS]

    # Normalise per dimension
    max_vals = [max(abs(a), abs(b), 1e-9) for a, b in zip(vals_a, vals_b)]
    norm_a   = [v / m for v, m in zip(vals_a, max_vals)]
    norm_b   = [v / m for v, m in zip(vals_b, max_vals)]

    cats   = _RADAR_LABELS + [_RADAR_LABELS[0]]
    norm_a = norm_a + [norm_a[0]]
    norm_b = norm_b + [norm_b[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_a, theta=cats, fill="toself",
        name=name_a,
        line=dict(color=HOME_COLOR, width=2),
        fillcolor="rgba(0,184,212,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm_b, theta=cats, fill="toself",
        name=name_b,
        line=dict(color=AWAY_COLOR, width=2, dash="dot"),
        fillcolor="rgba(255,76,107,0.12)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=BG_PRIMARY,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color=TEXT_MUTED, size=9, family="JetBrains Mono, monospace"),
                gridcolor=GRID_COLOR,
                linecolor=GRID_COLOR,
            ),
            angularaxis=dict(
                tickfont=dict(color=TEXT_PRIMARY, size=10, family="Barlow Condensed, sans-serif"),
                gridcolor=GRID_COLOR,
                linecolor=GRID_COLOR,
            ),
        ),
        showlegend=True,
        legend=dict(
            font=dict(color=TEXT_PRIMARY, size=11, family="Barlow Condensed, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=420,
        margin=dict(t=40, b=40, l=50, r=50),
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_PRIMARY,
        font=dict(color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
    )
    return fig


def _comparison_bar(
    row_a: pd.Series,
    row_b: pd.Series,
    name_a: str,
    name_b: str,
) -> go.Figure:
    """Grouped bar chart comparing key metrics side-by-side."""
    labels, vals_a, vals_b = [], [], []
    for field, label in _COMPARISON_FIELDS:
        v_a = row_a.get(field, None)
        v_b = row_b.get(field, None)
        if v_a is None and v_b is None:
            continue
        labels.append(label)
        vals_a.append(float(v_a) if pd.notna(v_a) else 0.0)
        vals_b.append(float(v_b) if pd.notna(v_b) else 0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=name_a,
        x=labels, y=vals_a,
        marker_color=HOME_COLOR,
        text=[f"{v:.3f}" for v in vals_a],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=9),
    ))
    fig.add_trace(go.Bar(
        name=name_b,
        x=labels, y=vals_b,
        marker_color=AWAY_COLOR,
        text=[f"{v:.3f}" for v in vals_b],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=9),
    ))
    fig.update_layout(
        barmode="group",
        xaxis=dict(
            tickfont=dict(color=TEXT_PRIMARY, size=10, family="Barlow Condensed, sans-serif"),
            showgrid=False,
        ),
        yaxis=dict(
            tickfont=dict(color=TEXT_MUTED, family="JetBrains Mono, monospace"),
            showgrid=True,
            gridcolor=GRID_COLOR,
        ),
        legend=dict(
            font=dict(color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=350,
        margin=dict(t=20, b=60, l=50, r=20),
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_PRIMARY,
        font=dict(color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
    )
    return fig


def _vaep_timeline_compare(
    vaep_df: pd.DataFrame | None,
    pid_a: str,
    pid_b: str,
    name_a: str,
    name_b: str,
) -> go.Figure | None:
    """Cumulative VAEP over match time for two players."""
    if (
        vaep_df is None
        or Cols.VAEP_VALUE not in vaep_df.columns
        or Cols.MINUTE not in vaep_df.columns
    ):
        return None

    fig = go.Figure()
    for pid, name, color in [(pid_a, name_a, HOME_COLOR), (pid_b, name_b, AWAY_COLOR)]:
        pa = (
            vaep_df[vaep_df[Cols.PLAYER_ID].astype(str) == str(pid)]
            .sort_values(Cols.MINUTE)
            .copy()
        )
        if pa.empty:
            continue
        pa["cum"] = pa[Cols.VAEP_VALUE].cumsum()
        fig.add_trace(go.Scatter(
            x=pa[Cols.MINUTE].tolist(),
            y=pa["cum"].tolist(),
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))

    if not fig.data:
        return None

    fig.add_hline(y=0, line_dash="dash", line_color="#334155", line_width=1)
    fig.update_layout(
        xaxis_title="Match minute",
        yaxis_title="Cumulative VAEP",
        xaxis=dict(
            tickfont=dict(color=TEXT_MUTED, family="JetBrains Mono, monospace"),
            showgrid=False,
        ),
        yaxis=dict(
            tickfont=dict(color=TEXT_MUTED, family="JetBrains Mono, monospace"),
            showgrid=True, gridcolor=GRID_COLOR,
        ),
        legend=dict(
            font=dict(color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=300,
        margin=dict(t=20, b=40),
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_PRIMARY,
        font=dict(color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
    )
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(row: pd.Series, field: str, decimals: int = 4) -> str:
    v = row.get(field, None)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/A"
    return str(round(float(v), decimals))


def _delta_sign(a: pd.Series, b: pd.Series, field: str) -> bool | None:
    """Return True if a > b, False if a < b, None if equal."""
    va = a.get(field, None)
    vb = b.get(field, None)
    if va is None or vb is None:
        return None
    try:
        fa, fb = float(va), float(vb)
        if abs(fa - fb) < 1e-9:
            return None
        return fa > fb
    except (TypeError, ValueError):
        return None


# ── Public render ─────────────────────────────────────────────────────────────

def render() -> None:
    inject_css()
    st.title("Player Comparison Tool")

    match_ids = list_match_ids()
    if not match_ids:
        st.warning("No processed matches found. Run the pipeline first.", icon="⚠️")
        return

    match_id = st.selectbox("Select match", match_ids, format_func=match_label)
    summary  = load_player_summary(match_id)

    if summary is None or summary.empty:
        st.error("Player summary not found. Run Phase 6 first.")
        return

    player_names = sorted(summary["player_name"].dropna().unique().tolist()) \
        if "player_name" in summary.columns else []

    if len(player_names) < 2:
        st.warning("Need at least 2 players to compare.")
        return

    # ── Player selectors ──────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<span class="team-badge home">Player A</span>', unsafe_allow_html=True)
        name_a = st.selectbox("Player A", player_names, key="cmp_a")
    with col_b:
        st.markdown('<span class="team-badge away">Player B</span>', unsafe_allow_html=True)
        default_b = player_names[1] if player_names[1] != name_a else player_names[0]
        name_b = st.selectbox("Player B", player_names,
                              index=player_names.index(default_b), key="cmp_b")

    if name_a == name_b:
        st.info("Select two different players to compare.")
        return

    row_a = summary[summary["player_name"] == name_a].iloc[0]
    row_b = summary[summary["player_name"] == name_b].iloc[0]
    pid_a = str(row_a.get(Cols.PLAYER_ID, ""))
    pid_b = str(row_b.get(Cols.PLAYER_ID, ""))

    st.divider()

    # ── KPI metric cards ──────────────────────────────────────────────────────
    section_header("Key Metrics")
    metrics = [
        ("Actions",    "action_count",   0),
        ("VAEP Total", "vaep_total",      4),
        ("Off. VAEP",  "vaep_offensive",  4),
        ("Def. VAEP",  "vaep_defensive",  4),
        ("VAEP / 90",  "vaep_per_90",     4),
        ("xG Total",   "xg_total",        4),
    ]

    html_a, html_b = [], []
    for label, field, dec in metrics:
        val_a = _safe_float(row_a, field, dec)
        val_b = _safe_float(row_b, field, dec)
        sign  = _delta_sign(row_a, row_b, field)

        html_a.append(kpi_card(label, val_a,
                               delta=f"vs {name_b}: {val_b}",
                               delta_positive=sign,
                               accent=HOME_COLOR))
        html_b.append(kpi_card(label, val_b,
                               delta=f"vs {name_a}: {val_a}",
                               delta_positive=None if sign is None else (not sign),
                               accent=AWAY_COLOR))

    def _kpi_row(parts: list[str]) -> str:
        return '<div class="kpi-row">' + "".join(parts) + "</div>"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{name_a}**")
        st.markdown(_kpi_row(html_a[:3]), unsafe_allow_html=True)
        st.markdown(_kpi_row(html_a[3:]), unsafe_allow_html=True)
    with col2:
        st.markdown(f"**{name_b}**")
        st.markdown(_kpi_row(html_b[:3]), unsafe_allow_html=True)
        st.markdown(_kpi_row(html_b[3:]), unsafe_allow_html=True)

    st.divider()

    # ── Radar + bar chart ─────────────────────────────────────────────────────
    section_header("Profile Comparison")
    r1, r2 = st.columns([1, 1])
    with r1:
        st.subheader("Radar")
        fig_radar = _comparison_radar(row_a, row_b, name_a, name_b)
        st.plotly_chart(fig_radar, width="stretch")

    with r2:
        st.subheader("Metric Bars")
        fig_bar = _comparison_bar(row_a, row_b, name_a, name_b)
        st.plotly_chart(fig_bar, width="stretch")

    st.divider()

    # ── VAEP timeline ─────────────────────────────────────────────────────────
    section_header("VAEP Accumulation Over Match")
    vaep_df   = load_vaep(match_id)
    fig_time  = _vaep_timeline_compare(vaep_df, pid_a, pid_b, name_a, name_b)
    if fig_time:
        st.plotly_chart(fig_time, width="stretch")
    else:
        st.info("VAEP timeline data not available.")

    st.divider()

    # ── Delta table ───────────────────────────────────────────────────────────
    section_header("Full Stats Delta")
    rows = []
    for field, label in _COMPARISON_FIELDS:
        va = row_a.get(field, None)
        vb = row_b.get(field, None)
        if va is None and vb is None:
            continue
        try:
            fa = round(float(va), 4) if pd.notna(va) else None
            fb = round(float(vb), 4) if pd.notna(vb) else None
            delta = round(fa - fb, 4) if (fa is not None and fb is not None) else None
        except (TypeError, ValueError):
            fa, fb, delta = None, None, None
        rows.append({
            "Metric":   label,
            name_a:     fa if fa is not None else "N/A",
            name_b:     fb if fb is not None else "N/A",
            "Delta (A-B)": delta if delta is not None else "N/A",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
