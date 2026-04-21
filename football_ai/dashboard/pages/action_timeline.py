"""
Action Timeline page.

Renders the match event stream as a scrollable timeline with VAEP bar
chart, action-type filter, and an interactive scatter plot of actions
on the pitch coloured by VAEP value.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_ai.constants import Cols
from football_ai.dashboard.data_loader import (
    get_team_ids,
    get_team_name,
    list_match_ids,
    load_vaep,
    match_label,
)


def _vaep_timeline(df: pd.DataFrame, title: str) -> go.Figure:
    """VAEP bar chart over match minutes."""
    fig = go.Figure()

    pos_mask = df[Cols.VAEP_VALUE] >= 0
    neg_mask = df[Cols.VAEP_VALUE] < 0

    fig.add_trace(go.Bar(
        x=df.loc[pos_mask, Cols.MINUTE].tolist(),
        y=df.loc[pos_mask, Cols.VAEP_VALUE].tolist(),
        name="Positive VAEP",
        marker_color="#00E5A0",
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=df.loc[neg_mask, Cols.MINUTE].tolist(),
        y=df.loc[neg_mask, Cols.VAEP_VALUE].tolist(),
        name="Negative VAEP",
        marker_color="#FF4C6B",
        opacity=0.85,
    ))
    fig.add_hline(y=0, line_color="#334155", line_width=0.8)

    _tickfont = dict(family="JetBrains Mono, monospace", size=10, color="#64748B")
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Barlow Condensed, sans-serif", size=13, color="#E2E8F0"),
        ),
        xaxis_title="Minute",
        yaxis_title="VAEP value",
        barmode="overlay",
        height=300,
        margin=dict(t=40, b=30),
        plot_bgcolor="#0A0E1A",
        paper_bgcolor="#0A0E1A",
        font=dict(color="#E2E8F0", family="Barlow Condensed, sans-serif"),
        xaxis=dict(gridcolor="#1E293B", tickfont=_tickfont),
        yaxis=dict(gridcolor="#1E293B", tickfont=_tickfont),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def _pitch_scatter(df: pd.DataFrame, title: str) -> go.Figure:
    """Scatter of actions on pitch coloured by VAEP value."""
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    # Pitch background
    fig.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80,
                  fillcolor="#0C2A1A", line=dict(color="#334155", width=2))
    # Centre line
    fig.add_shape(type="line", x0=60, y0=0, x1=60, y1=80,
                  line=dict(color="#334155", width=1))

    vaep_vals = df[Cols.VAEP_VALUE].tolist()
    fig.add_trace(go.Scatter(
        x=df[Cols.START_X].tolist(),
        y=df[Cols.START_Y].tolist(),
        mode="markers",
        marker=dict(
            size=8,
            color=vaep_vals,
            colorscale="RdYlGn",
            cmid=0,
            showscale=True,
            colorbar=dict(title="VAEP", thickness=12),
            opacity=0.85,
            line=dict(color="white", width=0.3),
        ),
        hovertemplate=(
            "%{customdata[0]}  %{customdata[1]}<br>"
            "Min: %{customdata[2]}  VAEP: %{customdata[3]:.4f}<extra></extra>"
        ),
        customdata=list(zip(
            df.get(Cols.PLAYER_NAME, pd.Series(["?"] * len(df))).tolist(),
            df.get(Cols.ACTION_TYPE, pd.Series(["?"] * len(df))).tolist(),
            df.get(Cols.MINUTE, pd.Series([0] * len(df))).tolist(),
            vaep_vals,
        )),
    ))

    _tickfont = dict(family="JetBrains Mono, monospace", size=9, color="#475569")
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Barlow Condensed, sans-serif", size=13, color="#E2E8F0"),
        ),
        xaxis=dict(
            range=[0, 120], showgrid=False, zeroline=False,
            constrain="domain", tickfont=_tickfont,
        ),
        yaxis=dict(
            range=[0, 80], showgrid=False, zeroline=False,
            scaleanchor="x", scaleratio=1, tickfont=_tickfont,
        ),
        height=380,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="#0A0E1A",
        font=dict(color="#E2E8F0", family="Barlow Condensed, sans-serif"),
    )
    return fig


def render() -> None:
    st.title("⏱️ Action Timeline")

    match_ids = list_match_ids()
    if not match_ids:
        st.warning("No processed matches found. Run the pipeline first.", icon="⚠️")
        return

    match_id = st.selectbox("Select match", match_ids, format_func=match_label)
    vaep_df  = load_vaep(match_id)

    if vaep_df is None or vaep_df.empty:
        st.error("VAEP data not found for this match. Run Phase 6 first.")
        return

    team_ids  = get_team_ids(match_id)
    home_id   = team_ids[0] if team_ids else "home"
    away_id   = team_ids[1] if len(team_ids) > 1 else "away"
    home_name = get_team_name(match_id, home_id)
    away_name = get_team_name(match_id, away_id)

    # ── Filters ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        team_filter = st.multiselect(
            "Filter by team",
            [home_name, away_name],
            default=[home_name, away_name],
        )
    with col2:
        action_types_available = (
            sorted(vaep_df[Cols.ACTION_TYPE].unique().tolist())
            if Cols.ACTION_TYPE in vaep_df.columns
            else []
        )
        selected_types = st.multiselect(
            "Filter by action type",
            action_types_available,
            default=action_types_available,
        )
    with col3:
        _has_vaep = Cols.VAEP_VALUE in vaep_df.columns
        min_vaep = float(vaep_df[Cols.VAEP_VALUE].min()) if _has_vaep else -1.0
        max_vaep = float(vaep_df[Cols.VAEP_VALUE].max()) if _has_vaep else 1.0
        vaep_range = st.slider(
            "VAEP range",
            min_value=round(min_vaep, 2),
            max_value=round(max_vaep, 2),
            value=(round(min_vaep, 2), round(max_vaep, 2)),
            step=0.01,
        )

    # ── Apply filters ─────────────────────────────────────────────────────────
    df = vaep_df.copy()

    # Team filter
    name_to_id = {home_name: home_id, away_name: away_id}
    selected_ids = [name_to_id[n] for n in team_filter if n in name_to_id]
    if selected_ids and Cols.TEAM_ID in df.columns:
        df = df[df[Cols.TEAM_ID].astype(str).isin([str(t) for t in selected_ids])]

    # Action type filter
    if selected_types and Cols.ACTION_TYPE in df.columns:
        df = df[df[Cols.ACTION_TYPE].isin(selected_types)]

    # VAEP range filter
    if Cols.VAEP_VALUE in df.columns:
        df = df[
            (df[Cols.VAEP_VALUE] >= vaep_range[0]) &
            (df[Cols.VAEP_VALUE] <= vaep_range[1])
        ]

    st.caption(f"Showing {len(df)} actions after filters")
    st.divider()

    # ── VAEP timeline (both teams) ────────────────────────────────────────────
    if Cols.MINUTE in df.columns and Cols.VAEP_VALUE in df.columns:
        st.subheader("VAEP over match time")
        tab1, tab2, tab3 = st.tabs([
            "Both teams", home_name, away_name
        ])
        with tab1:
            st.plotly_chart(
                _vaep_timeline(df, "All selected actions — VAEP by minute"),
                width="stretch",
            )
        _has_team = Cols.TEAM_ID in df.columns
        with tab2:
            home_df = (
                df[df[Cols.TEAM_ID].astype(str) == str(home_id)] if _has_team else df
            )
            st.plotly_chart(
                _vaep_timeline(home_df, f"{home_name} — VAEP by minute"),
                width="stretch",
            )
        with tab3:
            away_df = (
                df[df[Cols.TEAM_ID].astype(str) == str(away_id)] if _has_team else df
            )
            st.plotly_chart(
                _vaep_timeline(away_df, f"{away_name} — VAEP by minute"),
                width="stretch",
            )

    st.divider()

    # ── Pitch scatter ─────────────────────────────────────────────────────────
    st.subheader("Actions on pitch (coloured by VAEP)")
    if Cols.START_X in df.columns and Cols.VAEP_VALUE in df.columns:
        st.plotly_chart(
            _pitch_scatter(df, "Action locations coloured by VAEP"),
            width="stretch",
        )

    st.divider()

    # ── Event table ───────────────────────────────────────────────────────────
    st.subheader("Event log")
    table_cols = [
        c for c in [
            Cols.MINUTE, Cols.TEAM_NAME, Cols.PLAYER_NAME,
            Cols.ACTION_TYPE, Cols.RESULT, Cols.VAEP_VALUE,
            Cols.VAEP_OFFENSIVE, Cols.VAEP_DEFENSIVE,
            Cols.START_X, Cols.START_Y,
        ] if c in df.columns
    ]
    display = df[table_cols].copy()
    for col in [Cols.VAEP_VALUE, Cols.VAEP_OFFENSIVE, Cols.VAEP_DEFENSIVE]:
        if col in display.columns:
            display[col] = display[col].round(4)

    st.dataframe(
        display.sort_values(Cols.MINUTE).reset_index(drop=True)
        if Cols.MINUTE in display.columns
        else display.reset_index(drop=True),
        width="stretch",
        height=350,
    )
