"""
Formation Visualizer page.

Draws detected formations on a StatsBomb pitch using player average
positions derived from the VAEP DataFrame.  Home and away are shown
side-by-side with player name annotations.
"""

from __future__ import annotations

import warnings

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from football_ai.constants import PITCH_LENGTH, PITCH_WIDTH, Cols
from football_ai.dashboard.data_loader import (
    get_team_ids,
    get_team_name,
    list_match_ids,
    load_tactical_report,
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


# ── Private helpers ───────────────────────────────────────────────────────────

def _player_avg_positions(
    vaep_df: pd.DataFrame | None,
    team_id: str,
) -> dict[str, tuple[float, float]]:
    """
    Compute average (start_x, start_y) per player for one team.

    Returns
    -------
    dict mapping short player name to (mean_x, mean_y).
    """
    if vaep_df is None or vaep_df.empty:
        return {}
    if Cols.START_X not in vaep_df.columns or Cols.START_Y not in vaep_df.columns:
        return {}

    mask = vaep_df[Cols.TEAM_ID].astype(str) == str(team_id)
    keep_cols = [c for c in [Cols.PLAYER_NAME, Cols.PLAYER_ID, Cols.START_X, Cols.START_Y]
                 if c in vaep_df.columns]
    df   = vaep_df[mask][keep_cols].copy()
    if df.empty:
        return {}

    grouped = df.groupby(Cols.PLAYER_NAME, as_index=False).agg(
        mean_x=(Cols.START_X, "mean"),
        mean_y=(Cols.START_Y, "mean"),
        count=(Cols.START_X, "count"),
    )
    # Keep players with at least 3 actions for meaningful positions
    grouped = grouped[grouped["count"] >= 3]

    positions: dict[str, tuple[float, float]] = {}
    for _, row in grouped.iterrows():
        name = str(row[Cols.PLAYER_NAME])
        short = name.split()[-1] if name else name   # last name only
        positions[short] = (float(row["mean_x"]), float(row["mean_y"]))

    return positions


def _draw_formation(
    positions: dict[str, tuple[float, float]],
    team_name: str,
    formation: str,
    color: str,
    flip: bool = False,
) -> plt.Figure:
    """
    Draw player positions on a pitch background.

    Parameters
    ----------
    positions : dict  {short_name: (x, y)}
    flip      : If True, mirror x-axis (used for away team so they attack right→left).
    """
    if _HAS_MPLSOCCER:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            pitch = Pitch(
                pitch_type="statsbomb",
                pitch_color="#0A0E1A",
                line_color="#334155",
                linewidth=1.2,
                half=False,
            )
            fig, ax = pitch.draw(figsize=(8, 5.5))
    else:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        fig.patch.set_facecolor("#0A0E1A")
        ax.set_facecolor("#0C2A1A")
        ax.set_xlim(0, PITCH_LENGTH)
        ax.set_ylim(0, PITCH_WIDTH)
        ax.set_aspect("equal")
        # Draw simple pitch lines
        for rect in [
            plt.Rectangle((0, 0), PITCH_LENGTH, PITCH_WIDTH,
                           fill=False, edgecolor="#334155", linewidth=1.2),
        ]:
            ax.add_patch(rect)
        ax.axvline(PITCH_LENGTH / 2, color="#334155", linewidth=1.0, linestyle="--")

    if not positions:
        ax.text(
            PITCH_LENGTH / 2, PITCH_WIDTH / 2,
            "No position data",
            ha="center", va="center",
            color="#78909C", fontsize=12,
        )
    else:
        for short_name, (x, y) in positions.items():
            px = PITCH_LENGTH - x if flip else x

            # Player dot
            ax.scatter(
                px, y,
                s=300, c=color, zorder=5,
                edgecolors="white", linewidths=1.2,
                alpha=0.9,
            )
            # Player name label
            ax.text(
                px, y + 2.8,
                short_name,
                ha="center", va="bottom",
                fontsize=7.5,
                color="white",
                fontweight="bold",
                zorder=6,
                path_effects=[
                    pe.withStroke(linewidth=2, foreground="#0A0E1A"),
                ],
            )

    ax.set_title(
        f"{team_name}  [{formation}]",
        color="#E2E8F0",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    fig.tight_layout()
    return fig


def _formation_stats(
    vaep_df: pd.DataFrame | None,
    team_id: str,
) -> dict[str, int]:
    """Count key action types for a team."""
    out: dict[str, int] = {"passes": 0, "shots": 0, "dribbles": 0, "total": 0}
    if vaep_df is None or Cols.ACTION_TYPE not in vaep_df.columns:
        return out
    mask = vaep_df[Cols.TEAM_ID].astype(str) == str(team_id)
    df   = vaep_df[mask]
    out["total"]    = len(df)
    out["passes"]   = int((df[Cols.ACTION_TYPE] == "pass").sum())
    out["shots"]    = int((df[Cols.ACTION_TYPE] == "shot").sum())
    out["dribbles"] = int((df[Cols.ACTION_TYPE] == "dribble").sum())
    return out


# ── Public render ─────────────────────────────────────────────────────────────

def render() -> None:
    inject_css()
    st.title("Formation Visualizer")

    match_ids = list_match_ids()
    if not match_ids:
        st.warning("No processed matches found. Run the pipeline first.", icon="⚠️")
        return

    match_id = st.selectbox("Select match", match_ids, format_func=match_label)
    vaep_df  = load_vaep(match_id)
    report   = load_tactical_report(match_id)
    team_ids = get_team_ids(match_id)

    if not team_ids:
        st.error("No team data found for this match.")
        return

    home_id   = team_ids[0]
    away_id   = team_ids[1] if len(team_ids) > 1 else team_ids[0]
    home_name = get_team_name(match_id, home_id)
    away_name = get_team_name(match_id, away_id)

    home_formation = "unknown"
    away_formation = "unknown"
    if report:
        home_formation = report.get("home_formation", "unknown")
        away_formation = report.get("away_formation", "unknown")

    # ── Formation badge row ───────────────────────────────────────────────────
    section_header("Detected Formations")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(
            f'<span class="team-badge home">{home_name}</span>'
            f'<br><span style="font-size:1.6rem; font-weight:800; color:#ECEFF1;">'
            f'{home_formation}</span>',
            unsafe_allow_html=True,
        )
    with b2:
        st.markdown(
            f'<span class="team-badge away">{away_name}</span>'
            f'<br><span style="font-size:1.6rem; font-weight:800; color:#ECEFF1;">'
            f'{away_formation}</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Pitch visualizations ──────────────────────────────────────────────────
    section_header("Average Player Positions")
    pos_home = _player_avg_positions(vaep_df, home_id)
    pos_away = _player_avg_positions(vaep_df, away_id)

    col1, col2 = st.columns(2)
    with col1:
        fig_h = _draw_formation(pos_home, home_name, home_formation, HOME_COLOR, flip=False)
        st.pyplot(fig_h)
        plt.close(fig_h)

    with col2:
        fig_a = _draw_formation(pos_away, away_name, away_formation, AWAY_COLOR, flip=True)
        st.pyplot(fig_a)
        plt.close(fig_a)

    st.divider()

    # ── Action stats per team ─────────────────────────────────────────────────
    section_header("Action Breakdown")
    h_stats = _formation_stats(vaep_df, home_id)
    a_stats = _formation_stats(vaep_df, away_id)

    cols = st.columns(4)
    cols[0].metric(f"{home_name}  Total Actions", h_stats["total"])
    cols[1].metric(f"{home_name}  Passes",        h_stats["passes"])
    cols[2].metric(f"{away_name}  Total Actions", a_stats["total"])
    cols[3].metric(f"{away_name}  Passes",        a_stats["passes"])

    st.divider()

    # ── Player position table ─────────────────────────────────────────────────
    section_header("Player Average Positions")
    all_rows = []
    for name, (x, y) in pos_home.items():
        all_rows.append({
            "Player": name, "Team": home_name,
            "Avg X": round(x, 1), "Avg Y": round(y, 1),
        })
    for name, (x, y) in pos_away.items():
        all_rows.append({
            "Player": name, "Team": away_name,
            "Avg X": round(x, 1), "Avg Y": round(y, 1),
        })

    if all_rows:
        st.dataframe(pd.DataFrame(all_rows), width="stretch", hide_index=True)
    else:
        st.info("Not enough action data to compute player positions.")
