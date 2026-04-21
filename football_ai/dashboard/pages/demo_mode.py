"""
Professional Demo Mode page.

A curated single-screen dashboard designed for presentation to professors
and recruiters.  Minimal controls, maximum visual impact:

  1. Match identity banner  (auto-selects the first available match)
  2. Four headline KPI cards (xG, VAEP advantage, top player, possession)
  3. Match momentum graph   (full-width, diverging bar + cumulative lines)
  4. Passing networks       (home & away side-by-side on pitch backgrounds)
  5. Top performers table   (clean ranked leaderboard, no raw IDs)
  6. Technology stack strip (one-line summary of models used)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
from football_ai.dashboard.pages.match_analysis import (
    _momentum_graph,
    _possession_pct,
    _team_stat,
    _top_player_name,
)
from football_ai.dashboard.pages.passing_network import (
    _build_pass_network,
    _draw_network,
)
from football_ai.dashboard.style import (
    ACCENT_GREEN,
    AWAY_COLOR,
    BG_PRIMARY,
    GOLD,
    GRID_COLOR,
    HOME_COLOR,
    NEON_GREEN,
    TEXT_MUTED,
    TEXT_PRIMARY,
)

# ── Demo-specific CSS ─────────────────────────────────────────────────────────

_DEMO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Demo hero banner ──────────────────────────────────────────── */
.demo-hero {
    background: linear-gradient(135deg, #060A14 0%, #0D1528 40%, #0A1020 100%);
    border: 1px solid rgba(0,229,160,0.2);
    border-radius: 18px;
    padding: 32px 40px 24px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.demo-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent 10%, #00E5A0 50%, transparent 90%);
    opacity: 0.5;
}
.demo-hero::after {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,229,160,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.demo-badge {
    display: inline-block;
    background: linear-gradient(90deg, #00E5A0, #00B8D4);
    color: #0A0E1A;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.62rem;
    font-weight: 900;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 12px;
}
.demo-match-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.4rem;
    font-weight: 900;
    color: #E2E8F0;
    margin: 0 0 6px 0;
    line-height: 1.1;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.demo-match-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
    letter-spacing: 0.08em;
    margin: 0;
}

/* ── KPI strip ─────────────────────────────────────────────────── */
.demo-kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 28px;
}
.demo-kpi {
    background: rgba(28, 35, 51, 0.85);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 20px 22px;
    border-top: 3px solid var(--kpi-accent, #00E5A0);
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}
.demo-kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background: linear-gradient(135deg, rgba(0,229,160,0.03) 0%, transparent 50%);
    pointer-events: none;
}
.demo-kpi-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.62rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #475569;
    margin-bottom: 8px;
}
.demo-kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #E2E8F0;
    line-height: 1.1;
}
.demo-kpi-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #334155;
    margin-top: 6px;
}

/* ── Section label ─────────────────────────────────────────────── */
.demo-section {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: #334155;
    margin: 28px 0 12px 2px;
    border-left: 3px solid #00E5A0;
    padding-left: 12px;
    position: relative;
}
.demo-section::after {
    content: '';
    position: absolute;
    bottom: -4px; left: 0;
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,229,160,0.15), transparent 60%);
}

/* ── Tech strip ────────────────────────────────────────────────── */
.demo-tech-strip {
    background: rgba(10, 14, 26, 0.9);
    border-radius: 12px;
    padding: 16px 22px;
    margin-top: 32px;
    border: 1px solid #1E293B;
    display: flex;
    gap: 28px;
    flex-wrap: wrap;
    align-items: center;
}
.demo-tech-item {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #475569;
}
.demo-tech-item strong {
    color: #00E5A0;
}
</style>
"""


# ── KPI helpers ───────────────────────────────────────────────────────────────

def _demo_kpi(
    label: str,
    value: str,
    sub: str = "",
    accent: str = NEON_GREEN,
) -> str:
    """Return HTML for a single demo KPI tile."""
    return (
        f'<div class="demo-kpi" style="--kpi-accent:{accent};">'
        f'  <div class="demo-kpi-label">{label}</div>'
        f'  <div class="demo-kpi-value">{value}</div>'
        f'  <div class="demo-kpi-sub">{sub}</div>'
        f'</div>'
    )


def _build_kpi_html(
    vaep_df: pd.DataFrame | None,
    team_summary: pd.DataFrame | None,
    player_summary: pd.DataFrame | None,
    home_id: str,
    away_id: str,
    home_name: str,
    away_name: str,
) -> str:
    """Compute and render the 4 headline KPI tiles."""
    home_xg   = _team_stat(team_summary, "xg_total",   home_id)
    away_xg   = _team_stat(team_summary, "xg_total",   away_id)
    home_vaep = _team_stat(team_summary, "vaep_total", home_id)
    away_vaep = _team_stat(team_summary, "vaep_total", away_id)
    home_poss = _possession_pct(vaep_df, home_id) if vaep_df is not None else 50.0

    # xG advantage
    xg_leader = home_name if home_xg >= away_xg else away_name
    xg_diff   = abs(home_xg - away_xg)
    kpi1 = _demo_kpi(
        "xG Advantage",
        xg_leader,
        sub=f"+{xg_diff:.2f} xG ({home_xg:.2f} vs {away_xg:.2f})",
        accent=HOME_COLOR if home_xg >= away_xg else AWAY_COLOR,
    )

    # VAEP leader
    vaep_leader = home_name if home_vaep >= away_vaep else away_name
    vaep_diff   = abs(home_vaep - away_vaep)
    kpi2 = _demo_kpi(
        "VAEP Leader",
        vaep_leader,
        sub=f"+{vaep_diff:.3f} VAEP edge",
        accent=HOME_COLOR if home_vaep >= away_vaep else AWAY_COLOR,
    )

    # Top player across both teams
    top_name = _top_player_name(player_summary, home_id)
    top_vaep  = 0.0
    _has_vaep_total = (
        player_summary is not None
        and not player_summary.empty
        and "vaep_total" in player_summary.columns
    )
    if _has_vaep_total:
        top_row  = player_summary.nlargest(1, "vaep_total")
        if "player_name" in top_row.columns:
            top_name = str(top_row["player_name"].iloc[0])
        top_vaep = float(top_row["vaep_total"].iloc[0])
    kpi3 = _demo_kpi(
        "Top Performer",
        top_name,
        sub=f"{top_vaep:.3f} VAEP total",
        accent=GOLD,
    )

    # Possession
    away_poss = round(100.0 - home_poss, 1)
    kpi4 = _demo_kpi(
        "Possession",
        f"{home_poss:.0f}% — {away_poss:.0f}%",
        sub=f"{home_name} vs {away_name}",
        accent=ACCENT_GREEN,
    )

    return (
        '<div class="demo-kpi-strip">'
        + kpi1 + kpi2 + kpi3 + kpi4
        + '</div>'
    )


# ── Chart builders ────────────────────────────────────────────────────────────

def _demo_momentum(
    vaep_df: pd.DataFrame | None,
    home_id: str,
    away_id: str,
    home_name: str,
    away_name: str,
) -> go.Figure:
    """Full-width momentum graph with polished demo styling."""
    fig = _momentum_graph(vaep_df, home_id, away_id, home_name, away_name)
    fig.update_layout(
        height=340,
        title=None,
        margin=dict(t=10, b=40, l=55, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right",  x=1,
            font=dict(color=TEXT_PRIMARY, size=11, family="Barlow Condensed, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_PRIMARY,
        font=dict(color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
    )
    return fig


def _demo_top_players(
    player_summary: pd.DataFrame | None,
    n: int = 8,
) -> pd.DataFrame:
    """Return a clean top-n leaderboard DataFrame for display."""
    if player_summary is None or player_summary.empty:
        return pd.DataFrame()
    want = [
        ("player_name",    "Player"),
        ("team_name",      "Team"),
        ("action_count",   "Actions"),
        ("vaep_total",     "VAEP Total"),
        ("vaep_offensive", "Off. VAEP"),
        ("vaep_defensive", "Def. VAEP"),
        ("xg_total",       "xG"),
    ]
    present = [(field, label) for field, label in want if field in player_summary.columns]
    if not present:
        return pd.DataFrame()
    fields  = [field for field, _ in present]
    labels  = [label for _, label in present]
    top = player_summary.nlargest(n, "vaep_total")[fields].copy()
    top.columns = labels
    for col in ["VAEP Total", "Off. VAEP", "Def. VAEP", "xG"]:
        if col in top.columns:
            top[col] = top[col].round(4)
    top.insert(0, "Rank", range(1, len(top) + 1))
    return top.reset_index(drop=True)


# ── Section label helper ──────────────────────────────────────────────────────

def _section(title: str) -> None:
    st.markdown(f'<div class="demo-section">{title}</div>', unsafe_allow_html=True)


# ── Public render ─────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown(_DEMO_CSS, unsafe_allow_html=True)

    match_ids = list_match_ids()
    if not match_ids:
        st.warning("No processed matches found. Run the pipeline first.", icon="⚠️")
        return

    # Match selector — minimal, compact
    with st.container():
        if len(match_ids) > 1:
            match_id = st.selectbox(
                "Match",
                match_ids,
                format_func=match_label,
                label_visibility="collapsed",
            )
        else:
            match_id = match_ids[0]

    vaep_df    = load_vaep(match_id)
    team_sum   = load_team_summary(match_id)
    player_sum = load_player_summary(match_id)
    report     = load_tactical_report(match_id)
    team_ids   = get_team_ids(match_id)

    if not team_ids:
        st.error("No team data found for this match.")
        return

    home_id   = team_ids[0]
    away_id   = team_ids[1] if len(team_ids) > 1 else team_ids[0]
    home_name = get_team_name(match_id, home_id)
    away_name = get_team_name(match_id, away_id)
    home_form = (report or {}).get("home_formation", "")
    away_form = (report or {}).get("away_formation", "")

    # ── Hero banner ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="demo-hero">
            <div class="demo-badge">Professional Demo</div>
            <h1 class="demo-match-title">
                <span style="color:{HOME_COLOR};">{home_name}</span>
                <span style="color:#334155; font-size:1.3rem; margin:0 18px;">VS</span>
                <span style="color:{AWAY_COLOR};">{away_name}</span>
            </h1>
            <p class="demo-match-sub">
                MATCH &middot; {match_id}
                {"&nbsp;&nbsp;|&nbsp;&nbsp;" + home_form + " vs " + away_form
                 if home_form and away_form else ""}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Headline KPIs ─────────────────────────────────────────────────────────
    kpi_html = _build_kpi_html(
        vaep_df, team_sum, player_sum,
        home_id, away_id, home_name, away_name,
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── Match Momentum ────────────────────────────────────────────────────────
    _section("Match Momentum — VAEP per Minute")
    fig_mom = _demo_momentum(vaep_df, home_id, away_id, home_name, away_name)
    st.plotly_chart(fig_mom, use_container_width=True)

    # ── Passing Networks ──────────────────────────────────────────────────────
    _section("Passing Networks")
    pn_col1, pn_col2 = st.columns(2)

    for col, tid, tname, color in [
        (pn_col1, home_id, home_name, HOME_COLOR),
        (pn_col2, away_id, away_name, AWAY_COLOR),
    ]:
        with col:
            st.markdown(
                f"<div style='font-family:Barlow Condensed, sans-serif; "
                f"font-size:0.78rem; font-weight:700; color:{color}; "
                f"margin-bottom:6px; text-transform:uppercase; letter-spacing:0.08em;'>"
                f"{tname}</div>",
                unsafe_allow_html=True,
            )
            pos, edges = _build_pass_network(vaep_df, tid, min_passes=2)
            fig_net = _draw_network(pos, edges, tname, color)
            st.pyplot(fig_net)
            plt.close(fig_net)

    # ── Top Performers ────────────────────────────────────────────────────────
    _section("Top Performers")
    top_df = _demo_top_players(player_sum, n=8)
    if not top_df.empty:
        bar_col, tbl_col = st.columns([1, 1])

        with bar_col:
            bar_colors = [GOLD] + [NEON_GREEN] * max(0, len(top_df) - 1)
            fig_bar = go.Figure(go.Bar(
                x=top_df["VAEP Total"].tolist(),
                y=top_df["Player"].tolist(),
                orientation="h",
                marker_color=bar_colors[:len(top_df)],
                text=[f"{v:.3f}" for v in top_df["VAEP Total"].tolist()],
                textposition="outside",
                textfont=dict(
                    color=TEXT_PRIMARY, size=10,
                    family="JetBrains Mono, monospace",
                ),
                hovertemplate="%{y}: %{x:.3f} VAEP<extra></extra>",
            ))
            fig_bar.update_layout(
                xaxis=dict(
                    title="VAEP Total",
                    tickfont=dict(
                        color=TEXT_MUTED,
                        family="JetBrains Mono, monospace",
                        size=10,
                    ),
                    showgrid=True,
                    gridcolor=GRID_COLOR,
                ),
                yaxis=dict(
                    tickfont=dict(
                        color=TEXT_PRIMARY,
                        family="Barlow Condensed, sans-serif",
                        size=11,
                    ),
                    autorange="reversed",
                    showgrid=False,
                ),
                height=max(260, len(top_df) * 36),
                margin=dict(t=10, b=40, l=10, r=70),
                paper_bgcolor=BG_PRIMARY,
                plot_bgcolor=BG_PRIMARY,
                font=dict(
                    color=TEXT_PRIMARY,
                    family="Barlow Condensed, sans-serif",
                ),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with tbl_col:
            st.dataframe(top_df, use_container_width=True, hide_index=True)

    else:
        st.info("Player summary not available. Run Phase 6.")

    # ── Technology stack strip ────────────────────────────────────────────────
    st.markdown(
        """
        <div class="demo-tech-strip">
            <div class="demo-tech-item">
                <strong>VAEP</strong> &mdash; Value Added by Every Player
            </div>
            <div class="demo-tech-item">
                <strong>xG</strong> &mdash; Expected Goals (logistic regression)
            </div>
            <div class="demo-tech-item">
                <strong>xT</strong> &mdash; Expected Threat (Markov chain)
            </div>
            <div class="demo-tech-item">
                <strong>XGBoost</strong> &mdash; P(score) / P(concede) models
            </div>
            <div class="demo-tech-item">
                <strong>StatsBomb</strong> &mdash; 120 &times; 80 m coordinates
            </div>
            <div class="demo-tech-item">
                <strong>mplsoccer</strong> &mdash; pitch rendering
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
