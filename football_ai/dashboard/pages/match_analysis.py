"""
Match Analysis page — redesigned for ASE 1.

Features
--------
* KPI Intelligence Cards  (xG diff, VAEP diff, Possession %, Dangerous Attacks,
                           Pressure Intensity, Formations, Top Player, Danger Zone)
* Match Momentum Graph    (VAEP per minute — diverging bar + cumulative line)
* Team Comparison Chart   (grouped bar: xG, VAEP, Off/Def VAEP)
* Most Valuable Actions   (top-10 table)
* Action Type Distribution (horizontal bar per team)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_ai.constants import Cols
from football_ai.dashboard.data_loader import (
    get_team_ids,
    get_team_name,
    list_match_ids,
    load_player_summary,
    load_tactical_report,
    load_team_summary,
    load_vaep,
    match_label,
)
from football_ai.dashboard.style import (
    ACCENT_AMBER,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    ACCENT_TEAL,
    AWAY_COLOR,
    BG_PRIMARY,
    GRID_COLOR,
    HOME_COLOR,
    TEXT_MUTED,
    TEXT_PRIMARY,
    inject_css,
    kpi_card,
    match_banner,
    section_header,
)

# ── Metric helpers ─────────────────────────────────────────────────────────────

def _team_stat(
    team_summary: pd.DataFrame | None,
    col: str,
    tid: str,
) -> float:
    """Extract a scalar from the team summary DataFrame for a given team."""
    if team_summary is None or col not in team_summary.columns:
        return 0.0
    row = team_summary[team_summary[Cols.TEAM_ID].astype(str) == str(tid)]
    return float(row[col].iloc[0]) if not row.empty else 0.0


def _possession_pct(vaep_df: pd.DataFrame, team_id: str) -> float:
    """Estimate possession % as the share of total actions belonging to a team."""
    if vaep_df is None or Cols.TEAM_ID not in vaep_df.columns:
        return 50.0
    total = len(vaep_df)
    own   = (vaep_df[Cols.TEAM_ID].astype(str) == str(team_id)).sum()
    return round(own / total * 100, 1) if total else 50.0


def _dangerous_attacks(vaep_df: pd.DataFrame, team_id: str) -> int:
    """
    Count dangerous attacks: actions in the final third (x > 80 m)
    with positive VAEP value, as a proxy for offensive threat.
    """
    if vaep_df is None:
        return 0
    team_df = vaep_df[vaep_df[Cols.TEAM_ID].astype(str) == str(team_id)]
    if Cols.START_X not in team_df.columns or Cols.VAEP_VALUE not in team_df.columns:
        return 0
    mask = (team_df[Cols.START_X] > 80) & (team_df[Cols.VAEP_VALUE] > 0)
    return int(mask.sum())


def _pressure_intensity(vaep_df: pd.DataFrame, team_id: str) -> str:
    """Return the % of a team's actions performed under pressure."""
    if vaep_df is None or Cols.TEAM_ID not in vaep_df.columns:
        return "N/A"
    team_df = vaep_df[vaep_df[Cols.TEAM_ID].astype(str) == str(team_id)]
    if "under_pressure" not in team_df.columns:
        return "N/A"
    total = len(team_df)
    if total == 0:
        return "0%"
    pressed = team_df["under_pressure"].fillna(False).astype(bool).sum()
    return f"{pressed / total * 100:.1f}%"


def _most_dangerous_zone(vaep_df: pd.DataFrame, team_id: str) -> str:
    """
    Identify the pitch zone where the opponent generated the highest
    mean VAEP (proxy for the most exploited danger zone against this team).
    """
    if (
        vaep_df is None
        or Cols.ZONE_ID not in vaep_df.columns
        or Cols.VAEP_VALUE not in vaep_df.columns
    ):
        return "N/A"
    opp_df = vaep_df[vaep_df[Cols.TEAM_ID].astype(str) != str(team_id)]
    if opp_df.empty:
        return "N/A"
    zone_vaep = opp_df.groupby(Cols.ZONE_ID)[Cols.VAEP_VALUE].mean()
    if zone_vaep.empty:
        return "N/A"
    best_zone = int(zone_vaep.idxmax())
    return f"Zone {best_zone}"


def _top_player_name(
    player_summary: pd.DataFrame | None,
    team_id: str,
) -> str:
    """Return the name of the highest-VAEP player for a team."""
    if player_summary is None or "vaep_total" not in player_summary.columns:
        return "N/A"
    if Cols.TEAM_ID in player_summary.columns:
        ps = player_summary[player_summary[Cols.TEAM_ID].astype(str) == str(team_id)]
    else:
        ps = player_summary
    if ps.empty:
        return "N/A"
    row  = ps.nlargest(1, "vaep_total")
    name = row.get("player_name", row.get(Cols.PLAYER_NAME, pd.Series(["N/A"]))).iloc[0]
    return str(name) if pd.notna(name) else "N/A"


# ── KPI Cards ─────────────────────────────────────────────────────────────────

def _render_kpi_cards(
    vaep_df: pd.DataFrame,
    team_summary: pd.DataFrame | None,
    player_summary: pd.DataFrame | None,
    report: dict[str, Any] | None,
    home_id: str,
    away_id: str,
    home_name: str,
    away_name: str,
) -> None:
    """Render two rows of 4 KPI Intelligence Cards each."""

    # ── Compute all values ────────────────────────────────────────────────────
    home_xg   = _team_stat(team_summary, "xg_total",   home_id)
    away_xg   = _team_stat(team_summary, "xg_total",   away_id)
    home_vaep = _team_stat(team_summary, "vaep_total", home_id)
    away_vaep = _team_stat(team_summary, "vaep_total", away_id)

    home_poss  = _possession_pct(vaep_df, home_id)
    away_poss  = _possession_pct(vaep_df, away_id)
    home_da    = _dangerous_attacks(vaep_df, home_id)
    away_da    = _dangerous_attacks(vaep_df, away_id)
    home_press = _pressure_intensity(vaep_df, home_id)
    away_press = _pressure_intensity(vaep_df, away_id)
    home_form  = (report or {}).get("home_formation", "—")
    away_form  = (report or {}).get("away_formation", "—")
    home_top   = _top_player_name(player_summary, home_id)
    home_zone  = _most_dangerous_zone(vaep_df, home_id)

    xg_diff   = home_xg - away_xg
    vaep_diff = home_vaep - away_vaep

    def _shorten(name: str, n: int = 16) -> str:
        return name[:n] + "…" if len(name) > n else name

    # ── Row 1 — offensive / possession metrics ────────────────────────────────
    r1 = st.columns(4)

    with r1[0]:
        diff_lbl = "leads" if xg_diff > 0 else ("trails" if xg_diff < 0 else "equal")
        leader   = home_name if xg_diff >= 0 else away_name
        st.markdown(
            kpi_card(
                "xG Difference",
                f"{xg_diff:+.3f}",
                delta=f"{_shorten(leader)} {diff_lbl} by {abs(xg_diff):.3f}",
                delta_positive=xg_diff > 0 if xg_diff != 0 else None,
                accent=HOME_COLOR if xg_diff >= 0 else AWAY_COLOR,
            ),
            unsafe_allow_html=True,
        )

    with r1[1]:
        diff_lbl = "leads" if vaep_diff > 0 else ("trails" if vaep_diff < 0 else "equal")
        leader   = home_name if vaep_diff >= 0 else away_name
        st.markdown(
            kpi_card(
                "VAEP Difference",
                f"{vaep_diff:+.3f}",
                delta=f"{_shorten(leader)} {diff_lbl} by {abs(vaep_diff):.3f}",
                delta_positive=vaep_diff > 0 if vaep_diff != 0 else None,
                accent=HOME_COLOR if vaep_diff >= 0 else AWAY_COLOR,
            ),
            unsafe_allow_html=True,
        )

    with r1[2]:
        st.markdown(
            kpi_card(
                "Possession",
                f"{home_poss}% — {away_poss}%",
                delta=f"{_shorten(home_name)} · {_shorten(away_name)}",
                delta_positive=None,
                accent=ACCENT_AMBER,
            ),
            unsafe_allow_html=True,
        )

    with r1[3]:
        da_advantage = home_da > away_da
        st.markdown(
            kpi_card(
                "Dangerous Attacks",
                f"{home_da} — {away_da}",
                delta=f"{_shorten(home_name)} · {_shorten(away_name)}",
                delta_positive=da_advantage if home_da != away_da else None,
                accent=ACCENT_PURPLE,
            ),
            unsafe_allow_html=True,
        )

    # ── Row 2 — tactical / contextual metrics ─────────────────────────────────
    r2 = st.columns(4)

    with r2[0]:
        st.markdown(
            kpi_card(
                "Pressure Intensity",
                f"{home_press} — {away_press}",
                delta=f"{_shorten(home_name)} · {_shorten(away_name)}",
                delta_positive=None,
                accent=ACCENT_TEAL,
            ),
            unsafe_allow_html=True,
        )

    with r2[1]:
        st.markdown(
            kpi_card(
                "Formations",
                f"{home_form} · {away_form}",
                delta=f"{_shorten(home_name)} · {_shorten(away_name)}",
                delta_positive=None,
                accent=ACCENT_GREEN,
            ),
            unsafe_allow_html=True,
        )

    with r2[2]:
        top_val = _team_stat(team_summary, "vaep_total", home_id)
        st.markdown(
            kpi_card(
                "Top Player (Home)",
                _shorten(home_top, 18),
                delta=f"VAEP {top_val:.3f}",
                delta_positive=True,
                accent=HOME_COLOR,
            ),
            unsafe_allow_html=True,
        )

    with r2[3]:
        st.markdown(
            kpi_card(
                "Most Danger Zone",
                home_zone,
                delta=f"vs {_shorten(home_name)} defence",
                delta_positive=False,
                accent=AWAY_COLOR,
            ),
            unsafe_allow_html=True,
        )


# ── Match Momentum Graph ───────────────────────────────────────────────────────

def _momentum_graph(
    vaep_df: pd.DataFrame,
    home_id: str,
    away_id: str,
    home_name: str,
    away_name: str,
) -> go.Figure:
    """
    Interactive VAEP-based match momentum graph.

    Each minute's per-team VAEP is smoothed (5-min rolling window) and
    plotted as a diverging bar chart (home above x-axis, away below).
    Cumulative VAEP lines are overlaid on a secondary y-axis.
    A vertical dashed line marks half-time at minute 45.
    """
    if (
        vaep_df is None
        or Cols.MINUTE not in vaep_df.columns
        or Cols.VAEP_VALUE not in vaep_df.columns
        or Cols.TEAM_ID not in vaep_df.columns
    ):
        fig = go.Figure()
        fig.update_layout(
            title="Match Momentum — VAEP per Minute (no data)",
            plot_bgcolor=BG_PRIMARY,
            paper_bgcolor=BG_PRIMARY,
            font=dict(color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
            height=360,
        )
        return fig

    max_min = int(vaep_df[Cols.MINUTE].max()) + 2
    minutes = list(range(0, max_min))

    def _per_min(team_id: str) -> pd.Series:
        df = vaep_df[vaep_df[Cols.TEAM_ID].astype(str) == str(team_id)]
        return (
            df.groupby(Cols.MINUTE)[Cols.VAEP_VALUE]
            .sum()
            .reindex(minutes, fill_value=0.0)
        )

    home_raw = _per_min(home_id)
    away_raw = _per_min(away_id)

    # 5-minute rolling average for smoother momentum bars
    window = 5
    home_smooth = home_raw.rolling(window, center=True, min_periods=1).mean()
    away_smooth = away_raw.rolling(window, center=True, min_periods=1).mean()

    home_cum = home_raw.cumsum()
    away_cum = away_raw.cumsum()

    fig = go.Figure()

    # ── Diverging momentum bars ────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=minutes,
        y=home_smooth.values,
        name=f"{home_name}",
        marker_color=HOME_COLOR,
        opacity=0.82,
        hovertemplate="Min %{x} · %{y:.4f} VAEP<extra>" + home_name + "</extra>",
    ))
    fig.add_trace(go.Bar(
        x=minutes,
        y=(-away_smooth).values,
        name=f"{away_name}",
        marker_color=AWAY_COLOR,
        opacity=0.82,
        hovertemplate="Min %{x} · %{y:.4f} VAEP<extra>" + away_name + "</extra>",
    ))

    # ── Cumulative lines (secondary y-axis) ───────────────────────────────────
    fig.add_trace(go.Scatter(
        x=minutes,
        y=home_cum.values,
        name=f"{home_name} cumul.",
        line=dict(color=HOME_COLOR, width=1.8, dash="dot"),
        yaxis="y2",
        hovertemplate="Min %{x} · cumul. %{y:.4f}<extra>" + home_name + "</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=minutes,
        y=away_cum.values,
        name=f"{away_name} cumul.",
        line=dict(color=AWAY_COLOR, width=1.8, dash="dot"),
        yaxis="y2",
        hovertemplate="Min %{x} · cumul. %{y:.4f}<extra>" + away_name + "</extra>",
    ))

    # ── Half-time marker ──────────────────────────────────────────────────────
    fig.add_vline(
        x=45,
        line=dict(color="#90A4AE", width=1, dash="dash"),
    )
    fig.add_annotation(
        x=45, y=0,
        text="HT",
        showarrow=False,
        font=dict(color="#90A4AE", size=10),
        bgcolor="rgba(14,17,23,0.7)",
        borderpad=3,
    )

    fig.update_layout(
        title=dict(
            text="MATCH MOMENTUM — VAEP PER MINUTE",
            font=dict(size=14, color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
        ),
        xaxis=dict(
            title="Match Minute",
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            tickfont=dict(size=10, color=TEXT_MUTED, family="JetBrains Mono, monospace"),
        ),
        yaxis=dict(
            title="VAEP (5-min avg)",
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=True,
            zerolinecolor="#334155",
            zerolinewidth=1.5,
            tickfont=dict(size=10, color=TEXT_MUTED, family="JetBrains Mono, monospace"),
        ),
        yaxis2=dict(
            title="Cumulative VAEP",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=10, color="#475569", family="JetBrains Mono, monospace"),
        ),
        barmode="relative",
        plot_bgcolor=BG_PRIMARY,
        paper_bgcolor=BG_PRIMARY,
        font=dict(color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
        height=390,
        margin=dict(t=55, b=45, l=55, r=65),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )

    return fig


# ── Team Comparison Chart ─────────────────────────────────────────────────────

def _comparison_chart(
    team_summary: pd.DataFrame | None,
    home_id: str,
    away_id: str,
    home_name: str,
    away_name: str,
) -> go.Figure:
    """Grouped bar chart comparing xG, VAEP, Off. VAEP, Def. VAEP."""
    metrics       = ["xg_total", "vaep_total", "vaep_offensive", "vaep_defensive"]
    metric_labels = ["xG", "VAEP Total", "Off. VAEP", "Def. VAEP"]

    def ts(col: str, tid: str) -> float:
        return round(_team_stat(team_summary, col, tid), 3)

    home_vals = [ts(m, home_id) for m in metrics]
    away_vals = [ts(m, away_id) for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=home_name,
        x=metric_labels,
        y=home_vals,
        marker_color=HOME_COLOR,
        text=[f"{v:.3f}" for v in home_vals],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_PRIMARY),
        hovertemplate="%{x}: %{y:.3f}<extra>" + home_name + "</extra>",
    ))
    fig.add_trace(go.Bar(
        name=away_name,
        x=metric_labels,
        y=away_vals,
        marker_color=AWAY_COLOR,
        text=[f"{v:.3f}" for v in away_vals],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_PRIMARY),
        hovertemplate="%{x}: %{y:.3f}<extra>" + away_name + "</extra>",
    ))
    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(t=30, b=30, l=45, r=20),
        plot_bgcolor=BG_PRIMARY,
        paper_bgcolor=BG_PRIMARY,
        font=dict(color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, family="Barlow Condensed, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_MUTED),
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(family="Barlow Condensed, sans-serif", size=11, color=TEXT_PRIMARY),
        ),
    )
    return fig


# ── Action Type Distribution ──────────────────────────────────────────────────

def _action_distribution(
    vaep_df: pd.DataFrame,
    team_id: str,
    team_name: str,
    color: str,
) -> go.Figure:
    """Horizontal bar chart of top-10 action type counts for one team."""
    if vaep_df is None or Cols.ACTION_TYPE not in vaep_df.columns:
        return go.Figure()

    team_df = vaep_df[vaep_df[Cols.TEAM_ID].astype(str) == str(team_id)]
    counts  = team_df[Cols.ACTION_TYPE].value_counts().head(10)

    fig = go.Figure(go.Bar(
        y=counts.index.tolist(),
        x=counts.values.tolist(),
        orientation="h",
        marker_color=color,
        text=[str(v) for v in counts.values],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_PRIMARY),
        hovertemplate="%{y}: %{x} actions<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=team_name,
            font=dict(size=13, color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
        ),
        height=340,
        margin=dict(t=45, b=20, l=110, r=45),
        plot_bgcolor=BG_PRIMARY,
        paper_bgcolor=BG_PRIMARY,
        font=dict(color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
        xaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_MUTED),
        ),
        yaxis=dict(
            showgrid=False, autorange="reversed",
            tickfont=dict(family="Barlow Condensed, sans-serif", size=11, color=TEXT_PRIMARY),
        ),
    )
    return fig


# ── Main Render ───────────────────────────────────────────────────────────────

def render() -> None:
    """Render the Match Analysis page."""
    inject_css()

    st.markdown(
        '<h1 style="margin-bottom:0.25rem; color:#E2E8F0; '
        'font-family:Barlow Condensed, sans-serif; font-weight:900; '
        'text-transform:uppercase; letter-spacing:0.06em;">Match Analysis</h1>',
        unsafe_allow_html=True,
    )

    match_ids = list_match_ids()
    if not match_ids:
        st.warning("No processed matches found. Run the pipeline first.", icon="⚠️")
        return

    match_id = st.selectbox(
        "match",
        match_ids,
        format_func=match_label,
        label_visibility="collapsed",
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    vaep_df        = load_vaep(match_id)
    team_summary   = load_team_summary(match_id)
    player_summary = load_player_summary(match_id)
    report         = load_tactical_report(match_id)

    if vaep_df is None:
        st.error("VAEP data not found for this match. Run Phase 6 first.", icon="🔴")
        return

    team_ids  = get_team_ids(match_id)
    home_id   = team_ids[0] if team_ids else "home"
    away_id   = team_ids[1] if len(team_ids) > 1 else "away"
    home_name = get_team_name(match_id, home_id)
    away_name = get_team_name(match_id, away_id)

    # ── Match banner ──────────────────────────────────────────────────────────
    match_banner(home_name, away_name, match_id)

    # ── KPI Intelligence Cards ────────────────────────────────────────────────
    section_header("KPI Intelligence")
    _render_kpi_cards(
        vaep_df, team_summary, player_summary, report,
        home_id, away_id, home_name, away_name,
    )

    st.divider()

    # ── Match Momentum Graph ──────────────────────────────────────────────────
    section_header("Match Momentum")
    st.plotly_chart(
        _momentum_graph(vaep_df, home_id, away_id, home_name, away_name),
        width="stretch",
    )

    st.divider()

    # ── Team Comparison ───────────────────────────────────────────────────────
    section_header("Team Comparison")
    st.plotly_chart(
        _comparison_chart(team_summary, home_id, away_id, home_name, away_name),
        width="stretch",
    )

    st.divider()

    # ── Most Valuable Actions ─────────────────────────────────────────────────
    section_header("Most Valuable Actions")
    if Cols.VAEP_VALUE in vaep_df.columns:
        top_cols = [
            c for c in [
                Cols.MINUTE,
                Cols.PLAYER_NAME,
                Cols.TEAM_NAME,
                Cols.ACTION_TYPE,
                Cols.VAEP_VALUE,
                Cols.START_X,
                Cols.START_Y,
            ]
            if c in vaep_df.columns
        ]
        top_actions = vaep_df.nlargest(10, Cols.VAEP_VALUE)[top_cols].copy()
        if Cols.VAEP_VALUE in top_actions.columns:
            top_actions[Cols.VAEP_VALUE] = top_actions[Cols.VAEP_VALUE].round(4)
        st.dataframe(
            top_actions.reset_index(drop=True),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("VAEP values not available in this dataset.", icon="ℹ️")

    st.divider()

    # ── Action Type Distribution ──────────────────────────────────────────────
    section_header("Action Type Distribution")
    if Cols.ACTION_TYPE in vaep_df.columns and Cols.TEAM_ID in vaep_df.columns:
        ac1, ac2 = st.columns(2)
        with ac1:
            st.plotly_chart(
                _action_distribution(vaep_df, home_id, home_name, HOME_COLOR),
                width="stretch",
            )
        with ac2:
            st.plotly_chart(
                _action_distribution(vaep_df, away_id, away_name, AWAY_COLOR),
                width="stretch",
            )
    else:
        st.info("Action type data not available.", icon="ℹ️")
