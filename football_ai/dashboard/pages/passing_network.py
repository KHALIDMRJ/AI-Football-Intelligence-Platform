"""
Passing Network Visualization page — ASE 1.

Renders interactive passing networks for each team overlaid on a
football pitch.  Player nodes are sized by involvement (weighted degree).
Directed edges are scaled by pass frequency.

Dependencies
------------
    mplsoccer  — professional pitch backgrounds (optional; falls back to
                 a manual Plotly pitch if not installed)
    networkx   — graph construction and degree calculation
    matplotlib — figure rendering (displayed via st.pyplot)
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

try:
    import networkx as nx  # type: ignore[import]
    _HAS_NETWORKX = True
except ImportError:
    nx = None  # type: ignore[assignment]
    _HAS_NETWORKX = False
import pandas as pd
import streamlit as st

from football_ai.constants import Cols
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
    TEXT_PRIMARY,
    inject_css,
    match_banner,
    section_header,
)

# ── Optional dependency: mplsoccer ────────────────────────────────────────────
try:
    from mplsoccer import Pitch as _MplPitch  # type: ignore[import]
    _HAS_MPLSOCCER = True
except ImportError:
    _HAS_MPLSOCCER = False

# Action types treated as pass-like events when building the network
_PASS_TYPES: frozenset[str] = frozenset({
    "pass", "cross", "corner", "free_kick", "throw_in", "goal_kick",
})


# ── Network construction ──────────────────────────────────────────────────────

def _build_pass_network(
    vaep_df: pd.DataFrame,
    team_id: str,
    min_passes: int = 2,
) -> tuple[dict[str, tuple[float, float]], dict[tuple[str, str], int]]:
    """
    Build a weighted, directed passing network from VAEP action data.

    Strategy
    --------
    For each pass-type action by the given team, the "receiver" is
    approximated as the player who performs the next action for the
    same team.  This is a standard estimation used in passing-network
    literature when receiver IDs are not recorded.

    Parameters
    ----------
    vaep_df    : VAEP DataFrame (per-action rows, sorted by time).
    team_id    : Team to build the network for.
    min_passes : Minimum edge weight to include in the network.

    Returns
    -------
    positions : player_name → (avg_start_x, avg_start_y)
    edges     : (from_player, to_player) → pass count
    """
    if vaep_df is None or vaep_df.empty:
        return {}, {}

    required_cols = [Cols.TEAM_ID, Cols.PLAYER_NAME, Cols.START_X, Cols.START_Y]
    if not all(c in vaep_df.columns for c in required_cols):
        return {}, {}

    # Sort by period → minute → second → original index for chronological order
    sort_cols = [c for c in [Cols.PERIOD, Cols.MINUTE, "second", Cols.INDEX]
                 if c in vaep_df.columns]
    df = vaep_df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else vaep_df.copy()

    # ── Average positions per player (own team only) ──────────────────────────
    team_mask = df[Cols.TEAM_ID].astype(str) == str(team_id)
    team_df   = df[team_mask]

    positions: dict[str, tuple[float, float]] = {}
    for player, grp in team_df.groupby(Cols.PLAYER_NAME):
        if pd.isna(player) or str(player) in ("nan", "", "None"):
            continue
        positions[str(player)] = (
            float(grp[Cols.START_X].mean()),
            float(grp[Cols.START_Y].mean()),
        )

    # ── Edge weights: consecutive same-team actions after a pass ─────────────
    edges: dict[tuple[str, str], int] = {}
    has_action_type = Cols.ACTION_TYPE in df.columns

    for i in range(len(df) - 1):
        row_curr = df.iloc[i]
        row_next = df.iloc[i + 1]

        # Must be a pass-type action by our team
        if str(row_curr.get(Cols.TEAM_ID, "")) != str(team_id):
            continue
        if has_action_type:
            if str(row_curr.get(Cols.ACTION_TYPE, "")) not in _PASS_TYPES:
                continue

        # Receiver is from same team
        if str(row_next.get(Cols.TEAM_ID, "")) != str(team_id):
            continue

        from_p = str(row_curr.get(Cols.PLAYER_NAME, ""))
        to_p   = str(row_next.get(Cols.PLAYER_NAME, ""))

        if from_p in ("nan", "", "None") or to_p in ("nan", "", "None"):
            continue
        if from_p == to_p:
            continue

        key = (from_p, to_p)
        edges[key] = edges.get(key, 0) + 1

    # Filter by minimum frequency
    edges = {k: v for k, v in edges.items() if v >= min_passes}

    return positions, edges


# ── Network drawing ───────────────────────────────────────────────────────────

def _draw_network(
    positions: dict[str, tuple[float, float]],
    edges: dict[tuple[str, str], int],
    team_name: str,
    color: str,
) -> plt.Figure:
    """
    Draw the passing network on a football pitch.

    Uses mplsoccer for the pitch background when available; falls back
    to a manually drawn Matplotlib pitch otherwise.

    Node size is proportional to the player's weighted degree (total
    pass involvement).  Edge width and opacity are proportional to
    pass count.

    Parameters
    ----------
    positions : player_name → (x, y) average position on pitch.
    edges     : (from_player, to_player) → pass count.
    team_name : Title label.
    color     : Primary colour for nodes and edges.
    """
    # ── Pitch setup ───────────────────────────────────────────────────────────
    if _HAS_MPLSOCCER:
        pitch = _MplPitch(
            pitch_type="statsbomb",
            pitch_color="#0A0E1A",
            line_color="#334155",
            line_alpha=0.65,
            linewidth=1.5,
            goal_type="box",
        )
        # mplsoccer calls fig.set_tight_layout() internally, which raises a
        # PendingDeprecationWarning in matplotlib >= 3.8.  Suppress it here
        # since we cannot modify third-party library code.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
                module="mplsoccer",
            )
            fig, ax = pitch.draw(figsize=(11, 7.5))
        fig.patch.set_facecolor("#0A0E1A")
    else:
        fig, ax = plt.subplots(figsize=(11, 7.5))
        ax.set_facecolor("#0A0E1A")
        fig.patch.set_facecolor("#0A0E1A")
        ax.set_xlim(-3, 123)
        ax.set_ylim(-3, 83)
        _draw_manual_pitch(ax)
        ax.set_aspect("equal")
        ax.axis("off")

    # ── Empty state ───────────────────────────────────────────────────────────
    if not positions:
        ax.text(
            60, 40,
            "No pass data available for this team",
            ha="center", va="center",
            color="white", fontsize=13,
            fontweight="bold",
        )
        ax.set_title(
            f"{team_name} — Passing Network",
            color=TEXT_PRIMARY, fontsize=14, fontweight="bold", pad=12,
        )
        return fig

    # ── Build graph for degree calculation ───────────────────────────────────
    # Use NetworkX when available; otherwise compute degrees manually.
    max_edge_weight = max(edges.values()) if edges else 1

    if _HAS_NETWORKX and nx is not None:
        G = nx.DiGraph()
        for player in positions:
            G.add_node(player)
        for (src, dst), cnt in edges.items():
            G.add_edge(src, dst, weight=cnt)
        degree_map = {
            p: G.degree(p, weight="weight") if p in G else 1
            for p in positions
        }
    else:
        # Manual weighted degree (sum of all edge weights touching each player)
        degree_map: dict[str, float] = {p: 0.0 for p in positions}
        for (src, dst), cnt in edges.items():
            degree_map[src] = degree_map.get(src, 0.0) + cnt
            degree_map[dst] = degree_map.get(dst, 0.0) + cnt

    max_degree = max(degree_map.values(), default=1) or 1

    # ── Draw directed edges ───────────────────────────────────────────────────
    for (src, dst), cnt in sorted(edges.items(), key=lambda x: x[1]):
        if src not in positions or dst not in positions:
            continue
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        lw    = 0.6 + (cnt / max_edge_weight) * 5.0
        alpha = 0.30 + (cnt / max_edge_weight) * 0.60
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                alpha=alpha,
                mutation_scale=14,
                shrinkA=8,
                shrinkB=8,
            ),
        )

    # ── Draw player nodes ─────────────────────────────────────────────────────
    for player, (px, py) in positions.items():
        deg  = float(degree_map.get(player, 1))
        size = 180 + (deg / max_degree) * 900

        ax.scatter(
            px, py,
            s=size,
            c=color,
            zorder=5,
            edgecolors="white",
            linewidths=1.5,
            alpha=0.93,
        )

        # Short last-name label
        parts = player.split()
        label = parts[-1] if len(parts) > 1 else player[:12]
        ax.text(
            px, py + 3.8,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="white",
            fontweight="bold",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="#0A0E1A",
                edgecolor="none",
                alpha=0.65,
            ),
        )

    # ── Title & legend ────────────────────────────────────────────────────────
    ax.set_title(
        f"{team_name} — Passing Network",
        color=TEXT_PRIMARY,
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    legend_handles = [
        mpatches.Patch(color=color, label="Player node  (size ∝ pass involvement)"),
        plt.Line2D([0], [0], color=color, lw=2.5,
                   label="Pass direction  (width ∝ frequency)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=8.5,
        facecolor="#1C2333",
        edgecolor="none",
        labelcolor="white",
        framealpha=0.85,
    )

    fig.tight_layout(pad=0.8)
    return fig


def _draw_manual_pitch(ax: plt.Axes) -> None:
    """Draw minimal StatsBomb pitch markings when mplsoccer is unavailable."""
    kw = dict(color="#334155", linewidth=1.5, alpha=0.65)
    # Outline
    ax.plot([0, 0, 120, 120, 0], [0, 80, 80, 0, 0], **kw)
    # Halfway
    ax.plot([60, 60], [0, 80], **kw)
    # Penalty areas
    ax.plot([0, 18, 18, 0], [18, 18, 62, 62], **kw)
    ax.plot([120, 102, 102, 120], [18, 18, 62, 62], **kw)
    # Goals
    ax.plot([-2, 0, 0, -2], [36, 36, 44, 44], **kw)
    ax.plot([120, 122, 122, 120], [36, 36, 44, 44], **kw)
    # Centre circle (approx.)
    circle = plt.Circle((60, 40), 10, color="#334155", fill=False, linewidth=1.5, alpha=0.5)
    ax.add_patch(circle)
    ax.plot(60, 40, "o", color="#334155", ms=3, alpha=0.5)


# ── Network statistics panel ──────────────────────────────────────────────────

def _network_stats(
    positions: dict[str, tuple[float, float]],
    edges: dict[tuple[str, str], int],
) -> None:
    """Display compact summary statistics for a passing network."""
    if not positions and not edges:
        st.caption("No passing data available.")
        return

    total  = sum(edges.values())
    unique = len(edges)
    top3   = sorted(edges.items(), key=lambda x: x[1], reverse=True)[:3]

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Players tracked",    len(positions))
    mc2.metric("Unique connections", unique)
    mc3.metric("Pass sequences",     total)

    if top3:
        pairs_html = "  &nbsp;|&nbsp;  ".join(
            f"<b>{src.split()[-1]} → {dst.split()[-1]}</b> ({cnt})"
            for (src, dst), cnt in top3
        )
        st.markdown(
            f"<small style='color:#78909C;'>Top combinations: {pairs_html}</small>",
            unsafe_allow_html=True,
        )


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    """Render the Passing Network Visualization page."""
    inject_css()

    st.markdown(
        '<h1 style="margin-bottom:0.25rem; color:#ECEFF1;">🔗 Passing Network</h1>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Player average positions and weighted directed pass connections "
        "overlaid on a StatsBomb pitch (x ∈ [0,120], y ∈ [0,80])."
    )

    missing = []
    if not _HAS_MPLSOCCER:
        missing.append("`mplsoccer`")
    if not _HAS_NETWORKX:
        missing.append("`networkx`")
    if missing:
        st.info(
            f"{', '.join(missing)} not installed — some features use fallbacks. "
            f"Install with: `pip install {' '.join(p.strip('`') for p in missing)}`",
            icon="ℹ️",
        )

    match_ids = list_match_ids()
    if not match_ids:
        st.warning("No processed matches found. Run the pipeline first.", icon="⚠️")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 1])
    with ctrl1:
        match_id = st.selectbox(
            "match",
            match_ids,
            format_func=match_label,
            label_visibility="collapsed",
        )
    with ctrl2:
        min_passes = st.number_input(
            "Min. passes",
            min_value=1,
            max_value=15,
            value=2,
            help="Minimum number of passes to draw a connection.",
        )
    with ctrl3:
        view = st.selectbox("View", ["Both teams", "Home only", "Away only"])

    vaep_df = load_vaep(match_id)
    if vaep_df is None:
        st.error("VAEP data not found. Run Phase 6 first.", icon="🔴")
        return

    team_ids  = get_team_ids(match_id)
    home_id   = team_ids[0] if team_ids else "home"
    away_id   = team_ids[1] if len(team_ids) > 1 else "away"
    home_name = get_team_name(match_id, home_id)
    away_name = get_team_name(match_id, away_id)

    match_banner(home_name, away_name, match_id)

    # ── Build networks ────────────────────────────────────────────────────────
    with st.spinner("Building passing networks…"):
        home_pos, home_edges, away_pos, away_edges = {}, {}, {}, {}
        if view in ("Both teams", "Home only"):
            home_pos, home_edges = _build_pass_network(
                vaep_df, home_id, int(min_passes)
            )
        if view in ("Both teams", "Away only"):
            away_pos, away_edges = _build_pass_network(
                vaep_df, away_id, int(min_passes)
            )

    # ── Render ────────────────────────────────────────────────────────────────
    if view == "Both teams":
        nc1, nc2 = st.columns(2)

        with nc1:
            section_header(home_name)
            fig_home = _draw_network(home_pos, home_edges, home_name, HOME_COLOR)
            st.pyplot(fig_home, width="stretch")
            plt.close(fig_home)
            _network_stats(home_pos, home_edges)

        with nc2:
            section_header(away_name)
            fig_away = _draw_network(away_pos, away_edges, away_name, AWAY_COLOR)
            st.pyplot(fig_away, width="stretch")
            plt.close(fig_away)
            _network_stats(away_pos, away_edges)

    elif view == "Home only":
        section_header(home_name)
        fig_home = _draw_network(home_pos, home_edges, home_name, HOME_COLOR)
        st.pyplot(fig_home, width="stretch")
        plt.close(fig_home)
        _network_stats(home_pos, home_edges)

    else:  # Away only
        section_header(away_name)
        fig_away = _draw_network(away_pos, away_edges, away_name, AWAY_COLOR)
        st.pyplot(fig_away, width="stretch")
        plt.close(fig_away)
        _network_stats(away_pos, away_edges)

    st.divider()
    st.caption(
        "Nodes: player average action positions. "
        "Edges: consecutive same-team actions after a pass — "
        "arrow direction shows pass flow, width shows frequency."
    )
