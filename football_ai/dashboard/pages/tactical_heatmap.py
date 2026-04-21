"""
Tactical Heatmap page.

Renders zone-level weakness and pressure maps on a pitch graphic
using Plotly scatter/heatmap overlaid on a green pitch background.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from football_ai.constants import Cols
from football_ai.dashboard.data_loader import (
    get_team_ids,
    get_team_name,
    list_match_ids,
    load_pressure,
    load_vaep,
    load_weaknesses,
    match_label,
)

# Pitch dimensions (StatsBomb standard)
_PITCH_L = 120.0
_PITCH_W = 80.0

# Risk level colours
_RISK_COLORS = {
    "critical": "#D32F2F",
    "high":     "#FF5722",
    "medium":   "#FFC107",
    "low":      "#4CAF50",
}


def _pitch_layout(title: str) -> dict:
    """Return Plotly layout dict for a pitch-coloured figure."""
    return dict(
        title=dict(
            text=title,
            font=dict(size=13, color="#E2E8F0", family="Barlow Condensed, sans-serif"),
        ),
        xaxis=dict(
            range=[0, _PITCH_L], showgrid=False, zeroline=False,
            title="x (metres)", constrain="domain",
            tickfont=dict(family="JetBrains Mono, monospace", size=9, color="#475569"),
        ),
        yaxis=dict(
            range=[0, _PITCH_W], showgrid=False, zeroline=False,
            title="y (metres)", scaleanchor="x", scaleratio=1,
            tickfont=dict(family="JetBrains Mono, monospace", size=9, color="#475569"),
        ),
        plot_bgcolor="#0C2A1A",
        paper_bgcolor="#0A0E1A",
        font=dict(color="#E2E8F0", family="Barlow Condensed, sans-serif"),
        height=420,
        margin=dict(t=50, b=30, l=30, r=30),
        legend=dict(
            orientation="h", yanchor="top", y=-0.12,
            font=dict(color="#E2E8F0"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )


def _pitch_lines() -> list[dict]:
    """Return Plotly shapes for pitch markings (white lines)."""
    style = dict(type="line", line=dict(color="white", width=1.5), layer="below")
    shapes = [
        # Boundary
        dict(**style, x0=0, y0=0,       x1=_PITCH_L, y1=0),
        dict(**style, x0=0, y0=_PITCH_W, x1=_PITCH_L, y1=_PITCH_W),
        dict(**style, x0=0, y0=0,       x1=0,        y1=_PITCH_W),
        dict(**style, x0=_PITCH_L, y0=0, x1=_PITCH_L, y1=_PITCH_W),
        # Half-way line
        dict(**style, x0=60, y0=0, x1=60, y1=_PITCH_W),
        # Penalty areas
        dict(**style, x0=0,  y0=18, x1=18, y1=18),
        dict(**style, x0=0,  y0=62, x1=18, y1=62),
        dict(**style, x0=0,  y0=18, x1=0,  y1=62),
        dict(**style, x0=18, y0=18, x1=18, y1=62),
        dict(**style, x0=102, y0=18, x1=120, y1=18),
        dict(**style, x0=102, y0=62, x1=120, y1=62),
        dict(**style, x0=102, y0=18, x1=102, y1=62),
        # Goal lines
        dict(**style, x0=0,   y0=36, x1=-2,  y1=36),
        dict(**style, x0=0,   y0=44, x1=-2,  y1=44),
        dict(**style, x0=120, y0=36, x1=122, y1=36),
        dict(**style, x0=120, y0=44, x1=122, y1=44),
    ]
    return shapes


def _weakness_map(df, title: str) -> go.Figure:
    """Render weakness zones as coloured scatter on pitch."""
    fig = go.Figure()

    if df is not None and not df.empty and "zone_x" in df.columns:
        for risk, color in _RISK_COLORS.items():
            subset = df[df["risk_level"] == risk] if "risk_level" in df.columns else df
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset["zone_x"].tolist(),
                y=subset["zone_y"].tolist(),
                mode="markers",
                marker=dict(
                    size=14,
                    color=color,
                    opacity=0.85,
                    line=dict(color="white", width=0.5),
                ),
                name=risk.capitalize(),
                hovertemplate=(
                    f"Risk: {risk}<br>"
                    "x: %{x:.1f}<br>"
                    "y: %{y:.1f}<br>"
                    "Opp. VAEP: %{customdata:.4f}<extra></extra>"
                ),
                customdata=(
                    subset["mean_opponent_vaep"].tolist()
                    if "mean_opponent_vaep" in subset.columns
                    else [0.0] * len(subset)
                ),
            ))

    fig.update_layout(**_pitch_layout(title))
    fig.update_layout(shapes=_pitch_lines())
    return fig


def _pressure_heatmap(pressure_df, team_name: str) -> go.Figure:
    """Render opponent threat as a bubble chart on pitch."""
    fig = go.Figure()

    if pressure_df is not None and not pressure_df.empty and "zone_x" in pressure_df.columns:
        threat = pressure_df["threat"].clip(lower=0) if "threat" in pressure_df.columns \
            else np.ones(len(pressure_df))
        max_threat = float(threat.max()) or 1.0

        fig.add_trace(go.Scatter(
            x=pressure_df["zone_x"].tolist(),
            y=pressure_df["zone_y"].tolist(),
            mode="markers",
            marker=dict(
                size=(threat / max_threat * 30 + 5).tolist(),
                color=threat.tolist(),
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(title="Threat", thickness=12),
                opacity=0.80,
                line=dict(color="white", width=0.5),
            ),
            name="Opponent threat",
            hovertemplate=(
                "x: %{x:.1f}  y: %{y:.1f}<br>"
                "Threat: %{marker.color:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(**_pitch_layout(f"Opponent pressure on {team_name}"))
    fig.update_layout(shapes=_pitch_lines())
    return fig


def _vaep_zone_heatmap(vaep_df, team_id: str, team_name: str) -> go.Figure:
    """Render mean VAEP per pitch zone as a heatmap for a team's own actions."""
    fig = go.Figure()

    if vaep_df is None or Cols.ZONE_ID not in vaep_df.columns:
        fig.update_layout(**_pitch_layout(f"VAEP zones — {team_name}"))
        return fig

    own_df = vaep_df[vaep_df[Cols.TEAM_ID].astype(str) == str(team_id)]
    if own_df.empty or Cols.VAEP_VALUE not in own_df.columns:
        fig.update_layout(**_pitch_layout(f"VAEP zones — {team_name}"))
        return fig

    from football_ai.config import settings
    from football_ai.utils import zone_to_coords

    zone_vaep = own_df.groupby(Cols.ZONE_ID)[Cols.VAEP_VALUE].mean()

    xs, ys, vs = [], [], []
    for zid, val in zone_vaep.items():
        cx, cy = zone_to_coords(
            int(zid),
            settings.pitch.zones_x,
            settings.pitch.zones_y,
            settings.pitch.length,
            settings.pitch.width,
        )
        xs.append(cx)
        ys.append(cy)
        vs.append(float(val))

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            size=16,
            color=vs,
            colorscale="RdYlGn",
            cmid=0,
            showscale=True,
            colorbar=dict(title="Mean VAEP", thickness=12),
            opacity=0.80,
            line=dict(color="white", width=0.5),
        ),
        hovertemplate="x: %{x:.1f}  y: %{y:.1f}<br>Mean VAEP: %{marker.color:.4f}<extra></extra>",
    ))

    fig.update_layout(**_pitch_layout(f"Own-action VAEP zones — {team_name}"))
    fig.update_layout(shapes=_pitch_lines())
    return fig


def render() -> None:
    st.title("🗺️ Tactical Heatmap")

    match_ids = list_match_ids()
    if not match_ids:
        st.warning("No processed matches found. Run the pipeline first.", icon="⚠️")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        match_id = st.selectbox("Select match", match_ids, format_func=match_label)
    with col2:
        map_type = st.selectbox(
            "Map type",
            ["Weakness zones", "Opponent pressure", "Own-action VAEP"],
        )

    team_ids   = get_team_ids(match_id)
    home_id    = team_ids[0] if team_ids else "home"
    away_id    = team_ids[1] if len(team_ids) > 1 else "away"
    home_name  = get_team_name(match_id, home_id)
    away_name  = get_team_name(match_id, away_id)

    vaep_df = load_vaep(match_id)

    st.divider()
    fc1, fc2 = st.columns(2)

    if map_type == "Weakness zones":
        home_weak = load_weaknesses(match_id, home_id)
        away_weak = load_weaknesses(match_id, away_id)

        with fc1:
            st.subheader(f"{home_name} weaknesses")
            if home_weak is not None and not home_weak.empty:
                st.caption(
                    f"Critical: {(home_weak['risk_level']=='critical').sum()}  |  "
                    f"High: {(home_weak['risk_level']=='high').sum()}"
                    if 'risk_level' in home_weak.columns else ""
                )
            st.plotly_chart(
                _weakness_map(home_weak, f"{home_name} — weakness zones"),
                width="stretch",
            )

        with fc2:
            st.subheader(f"{away_name} weaknesses")
            if away_weak is not None and not away_weak.empty:
                st.caption(
                    f"Critical: {(away_weak['risk_level']=='critical').sum()}  |  "
                    f"High: {(away_weak['risk_level']=='high').sum()}"
                    if 'risk_level' in away_weak.columns else ""
                )
            st.plotly_chart(
                _weakness_map(away_weak, f"{away_name} — weakness zones"),
                width="stretch",
            )

    elif map_type == "Opponent pressure":
        home_press = load_pressure(match_id, home_id)
        away_press = load_pressure(match_id, away_id)

        with fc1:
            st.plotly_chart(
                _pressure_heatmap(home_press, home_name),
                width="stretch",
            )
        with fc2:
            st.plotly_chart(
                _pressure_heatmap(away_press, away_name),
                width="stretch",
            )

    else:  # Own-action VAEP
        with fc1:
            st.plotly_chart(
                _vaep_zone_heatmap(vaep_df, home_id, home_name),
                width="stretch",
            )
        with fc2:
            st.plotly_chart(
                _vaep_zone_heatmap(vaep_df, away_id, away_name),
                width="stretch",
            )

    st.divider()
    st.caption(
        "Pitch coordinates follow the StatsBomb standard: x ∈ [0, 120] m, y ∈ [0, 80] m. "
        "Teams always attack left → right."
    )
