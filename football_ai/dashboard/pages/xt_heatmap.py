"""
xT Heatmap page.

Per-team expected-threat heatmap rendered on a proper StatsBomb pitch
using mplsoccer.  Threat is derived from the ``xt_delta`` column of the
VAEP DataFrame: positive values represent zones where the team generated
threat, negative values where they conceded it.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from football_ai.constants import PITCH_LENGTH, PITCH_WIDTH, Cols
from football_ai.dashboard.data_loader import (
    get_team_ids,
    get_team_name,
    list_match_ids,
    load_vaep,
    match_label,
)
from football_ai.dashboard.style import (
    AWAY_COLOR,
    HOME_COLOR,
    inject_css,
    section_header,
)

# ── Optional mplsoccer ────────────────────────────────────────────────────────
try:
    from mplsoccer import Pitch  # type: ignore[import]
    _HAS_MPLSOCCER = True
except ImportError:
    _HAS_MPLSOCCER = False


# ── Constants ─────────────────────────────────────────────────────────────────
_GRID_X = 12   # bins along pitch length
_GRID_Y = 8    # bins along pitch width


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_xt_grid(
    vaep_df: pd.DataFrame,
    team_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin xt_delta values on a spatial grid for one team.

    Returns
    -------
    (xi, yi, zi) arrays suitable for plt.pcolormesh / mplsoccer.heatmap.
    xi, yi are bin-centre coordinates; zi is the mean xt_delta per cell.
    """
    empty = (
        np.zeros((_GRID_Y, _GRID_X)),
        np.linspace(0, PITCH_LENGTH, _GRID_X),
        np.linspace(0, PITCH_WIDTH, _GRID_Y),
    )
    if vaep_df is None or vaep_df.empty:
        return empty
    if Cols.XT_DELTA not in vaep_df.columns or Cols.START_X not in vaep_df.columns:
        return empty

    team_mask = vaep_df[Cols.TEAM_ID].astype(str) == str(team_id)
    df = vaep_df[team_mask].copy()
    if df.empty:
        return empty

    x_edges = np.linspace(0, PITCH_LENGTH, _GRID_X + 1)
    y_edges = np.linspace(0, PITCH_WIDTH,  _GRID_Y + 1)

    zi = np.zeros((_GRID_Y, _GRID_X))
    counts = np.zeros_like(zi)

    for _, row in df.iterrows():
        x = float(row[Cols.START_X])
        y = float(row[Cols.START_Y])
        xt = float(row[Cols.XT_DELTA]) if pd.notna(row[Cols.XT_DELTA]) else 0.0
        xi_idx = int(np.clip(np.digitize(x, x_edges) - 1, 0, _GRID_X - 1))
        yi_idx = int(np.clip(np.digitize(y, y_edges) - 1, 0, _GRID_Y - 1))
        zi[yi_idx, xi_idx] += xt
        counts[yi_idx, xi_idx] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        zi = np.where(counts > 0, zi / counts, 0.0)

    xi_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    yi_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    return zi, xi_centres, yi_centres


def _draw_xt_heatmap(
    vaep_df: pd.DataFrame | None,
    team_id: str,
    team_name: str,
    color: str,
) -> plt.Figure:
    """Render the xT heatmap for one team on a pitch background."""
    zi, xi, yi = _build_xt_grid(vaep_df, team_id)

    x_edges = np.linspace(0, PITCH_LENGTH, _GRID_X + 1)
    y_edges = np.linspace(0, PITCH_WIDTH,  _GRID_Y + 1)

    # Symmetric colormap bounds
    abs_max = float(np.abs(zi).max()) if zi.any() else 0.01
    vmax = max(abs_max, 1e-6)
    vmin = -vmax

    if _HAS_MPLSOCCER:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            pitch = Pitch(
                pitch_type="statsbomb",
                pitch_color="#0A0E1A",
                line_color="#334155",
                linewidth=1.2,
            )
            fig, ax = pitch.draw(figsize=(10, 6.5))

        # Build arrays of bin-centre coords + values for mplsoccer heatmap
        # Use pcolormesh manually for full control
        xm, ym = np.meshgrid(x_edges, y_edges)
        pcm = ax.pcolormesh(
            xm, ym, zi,
            cmap="RdYlGn",
            vmin=vmin, vmax=vmax,
            alpha=0.75,
            zorder=2,
        )
        cb = fig.colorbar(pcm, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("Mean xT delta (positive = threat created)", color="#E2E8F0", fontsize=9)
        cb.ax.yaxis.set_tick_params(color="#E2E8F0")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#E2E8F0", fontsize=8)

    else:
        # Fallback: plain matplotlib
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.patch.set_facecolor("#0A0E1A")
        ax.set_facecolor("#0C2A1A")
        xm, ym = np.meshgrid(x_edges, y_edges)
        pcm = ax.pcolormesh(xm, ym, zi, cmap="RdYlGn", vmin=vmin, vmax=vmax, alpha=0.75)
        fig.colorbar(pcm, ax=ax, fraction=0.025, pad=0.02)
        ax.set_xlim(0, PITCH_LENGTH)
        ax.set_ylim(0, PITCH_WIDTH)
        ax.set_xlabel("Pitch length (m)", color="#E2E8F0")
        ax.set_ylabel("Pitch width (m)", color="#E2E8F0")
        ax.tick_params(colors="#E2E8F0")

    fig.suptitle(
        f"{team_name}  —  xT Threat Map",
        color="#E2E8F0",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def _xt_summary_stats(
    vaep_df: pd.DataFrame | None,
    team_id: str,
) -> dict[str, float]:
    """Return summary xT stats for one team."""
    out = {"total": 0.0, "positive": 0.0, "negative": 0.0, "peak": 0.0}
    if vaep_df is None or vaep_df.empty or Cols.XT_DELTA not in vaep_df.columns:
        return out
    mask = vaep_df[Cols.TEAM_ID].astype(str) == str(team_id)
    vals = vaep_df[mask][Cols.XT_DELTA].dropna()
    if vals.empty:
        return out
    out["total"]    = round(float(vals.sum()), 4)
    out["positive"] = round(float(vals[vals > 0].sum()), 4)
    out["negative"] = round(float(vals[vals < 0].sum()), 4)
    out["peak"]     = round(float(vals.max()), 4)
    return out


# ── Public render ─────────────────────────────────────────────────────────────

def render() -> None:
    inject_css()
    st.title("xT Threat Heatmap")

    match_ids = list_match_ids()
    if not match_ids:
        st.warning("No processed matches found. Run the pipeline first.", icon="⚠️")
        return

    match_id = st.selectbox("Select match", match_ids, format_func=match_label)
    vaep_df  = load_vaep(match_id)
    team_ids = get_team_ids(match_id)

    if not team_ids:
        st.error("No team data found for this match.")
        return

    home_id   = team_ids[0]
    away_id   = team_ids[1] if len(team_ids) > 1 else team_ids[0]
    home_name = get_team_name(match_id, home_id)
    away_name = get_team_name(match_id, away_id)

    if vaep_df is not None and Cols.XT_DELTA not in vaep_df.columns:
        st.warning(
            "The loaded VAEP data does not contain an `xt_delta` column. "
            "Re-run Phase 6 (VAEP computation) to generate xT values.",
            icon="⚠️",
        )
        return

    # ── KPI row ───────────────────────────────────────────────────────────────
    section_header("xT Summary")
    h_stats = _xt_summary_stats(vaep_df, home_id)
    a_stats = _xt_summary_stats(vaep_df, away_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{home_name}  xT Total", h_stats["total"])
    c2.metric(f"{home_name}  xT Created", h_stats["positive"])
    c3.metric(f"{away_name}  xT Total", a_stats["total"])
    c4.metric(f"{away_name}  xT Created", a_stats["positive"])

    st.divider()

    # ── Side-by-side heatmaps ─────────────────────────────────────────────────
    section_header("Threat Maps — per team")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(home_name)
        fig_h = _draw_xt_heatmap(vaep_df, home_id, home_name, HOME_COLOR)
        st.pyplot(fig_h)
        plt.close(fig_h)

    with col2:
        st.subheader(away_name)
        fig_a = _draw_xt_heatmap(vaep_df, away_id, away_name, AWAY_COLOR)
        st.pyplot(fig_a)
        plt.close(fig_a)

    st.divider()

    # ── Top xT zones table ────────────────────────────────────────────────────
    section_header("Highest-Threat Actions")
    if vaep_df is not None and Cols.XT_DELTA in vaep_df.columns:
        display_cols = [c for c in [
            Cols.MINUTE, Cols.PLAYER_NAME, Cols.TEAM_NAME,
            Cols.ACTION_TYPE, Cols.XT_DELTA, Cols.START_X, Cols.START_Y,
        ] if c in vaep_df.columns]
        top_xt = vaep_df.nlargest(15, Cols.XT_DELTA)[display_cols].copy()
        if Cols.XT_DELTA in top_xt.columns:
            top_xt[Cols.XT_DELTA] = top_xt[Cols.XT_DELTA].round(4)
        st.dataframe(top_xt.reset_index(drop=True), width="stretch", hide_index=True)
