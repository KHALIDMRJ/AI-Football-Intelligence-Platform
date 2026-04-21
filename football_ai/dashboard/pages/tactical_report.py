"""
Tactical Report UI page.

Renders the full tactical JSON report as polished dashboard panels:
  - Match overview banner
  - Key insights (top insights from VAEP/xG/formation)
  - Weaknesses panel (critical and high-risk zones)
  - Strengths panel
  - Top players ranked table
  - Formation summary
  - Key conclusions
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
    load_weaknesses,
    match_label,
)
from football_ai.dashboard.style import (
    AWAY_COLOR,
    BG_PRIMARY,
    GOLD,
    GRID_COLOR,
    HOME_COLOR,
    NEON_GREEN,
    TEXT_MUTED,
    TEXT_PRIMARY,
    inject_css,
    kpi_card,
    match_banner,
    section_header,
)

# ── Private helpers ───────────────────────────────────────────────────────────

def _safe(d: dict[str, Any], key: str, default: Any = "N/A") -> Any:
    return d.get(key, default) if d else default


def _fmt(val: Any, decimals: int = 4) -> str:
    if val is None or val == "N/A":
        return "N/A"
    try:
        return str(round(float(val), decimals))
    except (TypeError, ValueError):
        return str(val)


def _insight_bullets(
    report: dict[str, Any],
    home_name: str,
    away_name: str,
    home_id: str,
    away_id: str,
) -> list[str]:
    """Generate human-readable insight bullets from the report dict."""
    bullets: list[str] = []

    xg_h = report.get("xg_home", 0.0) or 0.0
    xg_a = report.get("xg_away", 0.0) or 0.0
    if xg_h > xg_a:
        bullets.append(
            f"{home_name} dominated chance creation with "
            f"{xg_h:.2f} xG vs {xg_a:.2f} xG."
        )
    elif xg_a > xg_h:
        bullets.append(
            f"{away_name} dominated chance creation with "
            f"{xg_a:.2f} xG vs {xg_h:.2f} xG."
        )
    else:
        bullets.append(f"Both teams created equal xG ({xg_h:.2f}).")

    vaep_h = report.get("vaep_home", 0.0) or 0.0
    vaep_a = report.get("vaep_away", 0.0) or 0.0
    leader  = home_name if vaep_h >= vaep_a else away_name
    diff    = abs(vaep_h - vaep_a)
    bullets.append(
        f"{leader} led in VAEP by {diff:.3f} "
        f"({vaep_h:.3f} vs {vaep_a:.3f})."
    )

    hf = report.get("home_formation", "?")
    af = report.get("away_formation", "?")
    bullets.append(f"Formations: {home_name} played {hf}, {away_name} played {af}.")

    # Top player from top_players list
    top_players = report.get("top_players", [])
    if top_players:
        tp = top_players[0]
        tp_name = tp.get("player_name", "Unknown")
        tp_vaep = _fmt(tp.get("vaep_total"), 3)
        bullets.append(f"Top performer: {tp_name} with {tp_vaep} VAEP total.")

    # Weakness bullets from sub-reports
    for key, team_name in [("home_report", home_name), ("away_report", away_name)]:
        sub = report.get(key) or {}
        n_crit = sub.get("n_critical_zones", 0) or 0
        n_high = sub.get("n_high_risk_zones", 0) or 0
        if n_crit > 0 or n_high > 0:
            bullets.append(
                f"{team_name} had {n_crit} critical and {n_high} high-risk defensive zones."
            )

    return bullets


def _top_players_fig(top_players: list[dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart of top players by VAEP."""
    names  = [p.get("player_name", "?") for p in top_players[:10]]
    values = [float(p.get("vaep_total", 0.0) or 0.0) for p in top_players[:10]]

    bar_colors = [GOLD] + [HOME_COLOR] * max(0, len(names) - 1)

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=bar_colors[:len(names)],
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=10),
        hovertemplate="%{y}: %{x:.3f} VAEP<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(
            title="VAEP Total",
            tickfont=dict(color=TEXT_MUTED, family="JetBrains Mono, monospace", size=10),
            showgrid=True,
            gridcolor=GRID_COLOR,
        ),
        yaxis=dict(
            tickfont=dict(color=TEXT_PRIMARY, size=10, family="Barlow Condensed, sans-serif"),
            autorange="reversed",
            showgrid=False,
        ),
        height=max(280, len(names) * 36),
        margin=dict(t=10, b=40, l=10, r=80),
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_PRIMARY,
        font=dict(color=TEXT_PRIMARY, family="Barlow Condensed, sans-serif"),
    )
    return fig


def _weakness_panel(
    report: dict[str, Any],
    match_id: str,
    team_id: str,
    team_name: str,
    color: str,
) -> None:
    """Render one team's weakness sub-panel."""
    sub = (report.get("home_report") or report.get("away_report") or {})
    # Determine which sub-report belongs to this team
    for key in ["home_report", "away_report"]:
        r = report.get(key) or {}
        if str(r.get("team_id", "")) == str(team_id):
            sub = r
            break

    n_crit = sub.get("n_critical_zones", 0) or 0
    n_high = sub.get("n_high_risk_zones", 0) or 0
    wx     = sub.get("weakest_zone_x")
    wy     = sub.get("weakest_zone_y")

    st.markdown(
        f'<span class="team-badge" style="background:{color};">{team_name}</span>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Critical Zones", n_crit)
    c2.metric("High-Risk Zones", n_high)
    c3.metric(
        "Weakest Zone",
        f"({wx:.0f}, {wy:.0f})" if wx is not None and wy is not None else "N/A",
    )

    # Weakness DataFrame from disk
    weak_df = load_weaknesses(match_id, team_id)
    if weak_df is not None and not weak_df.empty:
        display_cols = [c for c in [
            "zone_x", "zone_y", "risk_level", "mean_opponent_vaep",
        ] if c in weak_df.columns]
        st.dataframe(
            weak_df[display_cols].sort_values("mean_opponent_vaep", ascending=False)
                    .head(8)
                    .reset_index(drop=True),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No weakness data on disk (run Phase 7).")


def _top_players_table(top_players: list[dict[str, Any]]) -> None:
    """Render a styled ranked table of top players."""
    if not top_players:
        st.info("No player data in tactical report.")
        return

    rows = []
    for rank, p in enumerate(top_players, 1):
        rows.append({
            "Rank":        rank,
            "Player":      p.get("player_name", "?"),
            "Team":        p.get("team_name", "?"),
            "VAEP Total":  round(float(p.get("vaep_total", 0.0) or 0.0), 4),
            "Off. VAEP":   round(float(p.get("vaep_offensive", 0.0) or 0.0), 4),
            "Def. VAEP":   round(float(p.get("vaep_defensive", 0.0) or 0.0), 4),
        })

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _conclusions(
    report: dict[str, Any],
    home_name: str,
    away_name: str,
) -> list[str]:
    """Build key tactical conclusion sentences."""
    concs: list[str] = []

    vaep_h = report.get("vaep_home", 0.0) or 0.0
    vaep_a = report.get("vaep_away", 0.0) or 0.0
    xg_h   = report.get("xg_home", 0.0) or 0.0
    xg_a   = report.get("xg_away", 0.0) or 0.0

    dom = home_name if vaep_h >= vaep_a else away_name
    concs.append(f"{dom} controlled the match by VAEP metrics.")

    if xg_h > 0.5 or xg_a > 0.5:
        hi = home_name if xg_h > xg_a else away_name
        concs.append(f"{hi} created the more dangerous chances (xG).")

    home_sub = report.get("home_report") or {}
    away_sub = report.get("away_report") or {}
    hf = home_sub.get("formation", report.get("home_formation", "?"))
    af = away_sub.get("formation", report.get("away_formation", "?"))
    concs.append(f"Tactically, {home_name} deployed a {hf} vs {away_name}'s {af}.")

    total = report.get("total_actions", 0)
    concs.append(f"The match produced {total} analysed SPADL actions in total.")

    return concs


# ── Public render ─────────────────────────────────────────────────────────────

def render() -> None:
    inject_css()
    st.title("Tactical Report")

    match_ids = list_match_ids()
    if not match_ids:
        st.warning("No processed matches found. Run the pipeline first.", icon="⚠️")
        return

    match_id = st.selectbox("Select match", match_ids, format_func=match_label)
    report   = load_tactical_report(match_id)

    if report is None:
        st.error(
            "Tactical report not found.  Run Phase 7:\n"
            "```\npython scripts/run_pipeline.py --input data/raw/fus_FAR.csv\n```",
        )
        return

    team_ids = get_team_ids(match_id)
    home_id   = str(report.get("home_team_id", team_ids[0] if team_ids else ""))
    away_id   = str(report.get("away_team_id", team_ids[1] if len(team_ids) > 1 else ""))
    home_name = report.get("home_team_name") or get_team_name(match_id, home_id)
    away_name = report.get("away_team_name") or get_team_name(match_id, away_id)

    # ── Match banner ──────────────────────────────────────────────────────────
    match_banner(home_name, away_name, match_id)

    # ── Top-line KPIs ─────────────────────────────────────────────────────────
    section_header("Match Overview")
    xg_h   = float(report.get("xg_home",   0.0) or 0.0)
    xg_a   = float(report.get("xg_away",   0.0) or 0.0)
    vaep_h = float(report.get("vaep_home",  0.0) or 0.0)
    vaep_a = float(report.get("vaep_away",  0.0) or 0.0)
    total  = int(report.get("total_actions", 0) or 0)
    hf     = report.get("home_formation", "?")
    af     = report.get("away_formation", "?")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        kpi_card("xG Home", f"{xg_h:.3f}",
                 delta=f"vs Away {xg_a:.3f}",
                 delta_positive=(xg_h > xg_a) if xg_h != xg_a else None,
                 accent=HOME_COLOR),
        unsafe_allow_html=True,
    )
    c2.markdown(
        kpi_card("xG Away", f"{xg_a:.3f}",
                 delta=f"vs Home {xg_h:.3f}",
                 delta_positive=(xg_a > xg_h) if xg_h != xg_a else None,
                 accent=AWAY_COLOR),
        unsafe_allow_html=True,
    )
    c3.markdown(
        kpi_card("VAEP Home", f"{vaep_h:.3f}",
                 delta=f"vs Away {vaep_a:.3f}",
                 delta_positive=(vaep_h > vaep_a) if vaep_h != vaep_a else None,
                 accent=HOME_COLOR),
        unsafe_allow_html=True,
    )
    c4.markdown(
        kpi_card("VAEP Away", f"{vaep_a:.3f}",
                 delta=f"vs Home {vaep_h:.3f}",
                 delta_positive=(vaep_a > vaep_h) if vaep_h != vaep_a else None,
                 accent=AWAY_COLOR),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            f"<div style='text-align:center; color:{TEXT_MUTED};"
            f" font-size:0.82rem; margin:6px 0;'>"
            f"Total SPADL actions: "
            f"<strong style='color:{TEXT_PRIMARY};'>{total}</strong>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Formations: <strong style='color:{HOME_COLOR};'>{hf}</strong> vs "
            f"<strong style='color:{AWAY_COLOR};'>{af}</strong>"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Key insights ──────────────────────────────────────────────────────────
    section_header("Key Insights")
    bullets = _insight_bullets(report, home_name, away_name, home_id, away_id)
    for bullet in bullets:
        st.markdown(
            f"<div style='padding:5px 0 5px 12px; border-left:3px solid {NEON_GREEN};"
            f" color:{TEXT_PRIMARY}; margin-bottom:4px; "
            f"font-family:Barlow Condensed, sans-serif;'>{bullet}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Weaknesses ────────────────────────────────────────────────────────────
    section_header("Defensive Weaknesses")
    wc1, wc2 = st.columns(2)
    with wc1:
        _weakness_panel(report, match_id, home_id, home_name, HOME_COLOR)
    with wc2:
        _weakness_panel(report, match_id, away_id, away_name, AWAY_COLOR)

    st.divider()

    # ── Top players ───────────────────────────────────────────────────────────
    section_header("Top Performers")
    top_players = report.get("top_players", [])

    # Enrich from player_summary if report has limited fields
    if not top_players:
        ps = load_player_summary(match_id)
        if ps is not None and not ps.empty and "vaep_total" in ps.columns:
            top_players = ps.nlargest(10, "vaep_total").rename(columns={
                Cols.PLAYER_ID: "player_id",
                Cols.TEAM_NAME: "team_name",
            }).to_dict(orient="records")

    tp_col, bar_col = st.columns([1, 1])
    with tp_col:
        _top_players_table(top_players)
    with bar_col:
        if top_players:
            fig = _top_players_fig(top_players)
            st.plotly_chart(fig, width="stretch")

    st.divider()

    # ── Most valuable actions ─────────────────────────────────────────────────
    most_val = report.get("most_valuable_actions", [])
    if most_val:
        section_header("Most Valuable Actions")
        rows = []
        for a in most_val:
            rows.append({
                "Minute":      a.get("minute", "?"),
                "Player":      a.get("player_name", "?"),
                "Team":        a.get("team_name", "?"),
                "Type":        a.get("action_type", "?"),
                "VAEP":        round(float(a.get("vaep_value", 0.0) or 0.0), 4),
                "X":           a.get("start_x", "?"),
                "Y":           a.get("start_y", "?"),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.divider()

    # ── Key conclusions ───────────────────────────────────────────────────────
    section_header("Key Conclusions")
    concs = _conclusions(report, home_name, away_name)
    for i, conc in enumerate(concs, 1):
        st.markdown(
            f"<div style='padding:6px 0 6px 14px; "
            f"border-left:3px solid {HOME_COLOR if i % 2 == 1 else AWAY_COLOR};"
            f" color:{TEXT_PRIMARY}; margin-bottom:6px; font-size:0.9rem;"
            f" font-family:Barlow Condensed, sans-serif;'>"
            f"<strong>{i}.</strong> {conc}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Raw report expander ───────────────────────────────────────────────────
    with st.expander("Raw tactical report JSON", expanded=False):
        st.json(report)
