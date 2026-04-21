"""
Streamlit dashboard — Football AI Intelligence Platform.

Run with:
    streamlit run football_ai/dashboard/app.py --server.port 8501

Pages
-----
    Overview              — platform status, match inventory, cross-match leaderboard
    Match Analysis        — KPI cards, momentum graph, xG/VAEP comparison, top actions
    Passing Network       — weighted passing network overlaid on pitch (mplsoccer)
    xT Heatmap            — per-team expected threat heatmap on pitch
    Player Analysis       — individual player VAEP profile with radar chart
    Player Comparison     — compare two players side-by-side with radar and bars
    Rankings              — ranked player tables by any VAEP metric
    Formation Visualizer  — detected formations with average player positions on pitch
    Tactical Heatmap      — zone-level weakness, pressure, and VAEP maps on pitch
    Tactical Report       — full polished tactical report UI
    Action Timeline       — full event stream with VAEP bar chart and pitch scatter
"""

from __future__ import annotations

import streamlit as st

from football_ai.config import settings
from football_ai.logger import setup_logging

# Initialise logging once — subsequent imports share the configured handlers
setup_logging()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=settings.project_name,
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
from football_ai.dashboard.style import inject_css  # noqa: E402

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown(
        f"""
        <div style="padding:14px 0 4px 0;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="
                    width:36px; height:36px;
                    background: linear-gradient(135deg, #00E5A0, #00B8D4);
                    border-radius:10px;
                    display:flex; align-items:center; justify-content:center;
                    font-size:1.2rem;
                    box-shadow: 0 0 16px rgba(0,229,160,0.25);
                ">⚽</div>
                <div>
                    <div style="
                        font-family: 'Barlow Condensed', sans-serif;
                        font-size:1.1rem; font-weight:800;
                        color:#E2E8F0;
                        letter-spacing:0.06em;
                        text-transform:uppercase;
                    ">Football AI</div>
                    <div style="
                        font-family: 'JetBrains Mono', monospace;
                        font-size:0.62rem; color:#475569;
                        letter-spacing:0.04em;
                    ">v{settings.version} · {settings.environment}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="height:1px; background:linear-gradient(90deg, '
        'rgba(0,229,160,0.3), transparent); margin:10px 0 14px;"></div>',
        unsafe_allow_html=True,
    )

    # Section label + Demo
    st.markdown(
        "<div style='"
        "font-family: Barlow Condensed, sans-serif;"
        "font-size:0.58rem; font-weight:800; letter-spacing:0.18em;"
        "text-transform:uppercase; color:#00E5A0; margin-bottom:4px;"
        "'>PRESENTATION</div>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        [
            "▸  Demo Mode",
            "─────────────",
            "◈  Overview",
            "◉  Match Analysis",
            "◎  Player Analysis",
            "⊜  Player Comparison",
            "▴  Rankings",
            "─────────────",
            "⬡  Passing Network",
            "▦  xT Heatmap",
            "⊞  Formation Visualizer",
            "▣  Tactical Heatmap",
            "▧  Tactical Report",
            "▬  Action Timeline",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div style="height:1px; background:linear-gradient(90deg, '
        'rgba(0,229,160,0.15), transparent); margin:14px 0 10px;"></div>',
        unsafe_allow_html=True,
    )

    with st.expander("Pipeline", expanded=False):
        st.code(
            "python scripts/run_pipeline.py \\\n"
            "  --input data/raw/fus_FAR.csv",
            language="bash",
        )

    st.markdown(
        """
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size:0.62rem; color:#334155;
            margin-top:auto; padding-top:24px;
            line-height:1.8;
        ">
            <span style="color:#00E5A0;">VAEP</span> ·
            <span style="color:#00B8D4;">xG</span> ·
            <span style="color:#FFB300;">xT</span> ·
            <span style="color:#A78BFA;">ML</span><br>
            StatsBomb coordinates
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Page routing ──────────────────────────────────────────────────────────────
from football_ai.dashboard.pages import (  # noqa: E402
    action_timeline,
    demo_mode,
    formation_visualizer,
    match_analysis,
    overview,
    passing_network,
    player_analysis,
    player_comparison,
    rankings,
    tactical_heatmap,
    tactical_report,
    xt_heatmap,
)

# Strip icon prefixes and map to modules
_PAGES: dict[str, object] = {
    "Demo Mode":            demo_mode,
    "Overview":             overview,
    "Match Analysis":       match_analysis,
    "Passing Network":      passing_network,
    "xT Heatmap":           xt_heatmap,
    "Player Analysis":      player_analysis,
    "Player Comparison":    player_comparison,
    "Rankings":             rankings,
    "Formation Visualizer": formation_visualizer,
    "Tactical Heatmap":     tactical_heatmap,
    "Tactical Report":      tactical_report,
    "Action Timeline":      action_timeline,
}

# Extract page name from radio label (strip icon prefix)
if page and not page.startswith("─"):
    # Remove icon prefix (e.g., "▸  Demo Mode" -> "Demo Mode")
    page_name = page.split("  ", 1)[-1] if "  " in page else page
    if page_name in _PAGES:
        _PAGES[page_name].render()  # type: ignore[attr-defined]
