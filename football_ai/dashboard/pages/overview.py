"""
Overview page — platform status, match inventory, and cross-match leaderboard.

Redesigned with tactical war room aesthetic — neon accents, monospace stats,
dark premium surfaces.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_ai.dashboard.data_loader import (
    get_team_ids,
    get_team_name,
    list_match_ids,
    load_player_summary,
    load_team_summary,
    pipeline_status,
)
from football_ai.dashboard.style import (
    BG_PRIMARY,
    GOLD,
    GRID_COLOR,
    NEON_AMBER,
    NEON_GREEN,
    NEON_RED,
    TEXT_MUTED,
    TEXT_PRIMARY,
    inject_css,
    kpi_card,
    section_header,
    status_pill,
)


def render() -> None:
    """Render the Overview page."""
    inject_css()

    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="padding:16px 0 6px 0;">
            <h1 style="
                margin:0;
                font-family: 'Barlow Condensed', sans-serif;
                font-size:2.2rem;
                font-weight:900;
                color:#E2E8F0;
                text-transform:uppercase;
                letter-spacing:0.06em;
            ">
                Football AI Intelligence Platform
            </h1>
            <p style="
                margin:8px 0 0 0;
                font-family: 'JetBrains Mono', monospace;
                font-size:0.78rem;
                color:#475569;
                letter-spacing:0.04em;
            ">
                <span style="color:#00E5A0;">VAEP</span> &middot;
                <span style="color:#00B8D4;">xG</span> &middot;
                <span style="color:#FFB300;">xT</span> &middot;
                <span style="color:#A78BFA;">Machine Learning</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Pipeline status cards ─────────────────────────────────────────────────
    section_header("System Status")
    status = pipeline_status()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        done = bool(status["matches_processed"])
        st.markdown(
            kpi_card(
                "Matches Processed",
                str(status["matches_processed"]),
                delta=(
                    status_pill("Ready", "ok") if done
                    else status_pill("Run pipeline", "pending")
                ),
                delta_positive=done,
                accent=NEON_GREEN if done else NEON_AMBER,
            ),
            unsafe_allow_html=True,
        )
    with c2:
        done = bool(status["vaep_computed"])
        st.markdown(
            kpi_card(
                "VAEP Engine",
                "ONLINE" if done else "OFFLINE",
                delta=status_pill("Phase 6", "ok") if done else status_pill("Pending", "pending"),
                delta_positive=done,
                accent=NEON_GREEN if done else NEON_RED,
            ),
            unsafe_allow_html=True,
        )
    with c3:
        done = bool(status["tactical_computed"])
        st.markdown(
            kpi_card(
                "Tactical Intel",
                "ONLINE" if done else "OFFLINE",
                delta=status_pill("Phase 7", "ok") if done else status_pill("Pending", "pending"),
                delta_positive=done,
                accent=NEON_GREEN if done else NEON_RED,
            ),
            unsafe_allow_html=True,
        )
    with c4:
        done = bool(status["models_trained"])
        st.markdown(
            kpi_card(
                "ML Models",
                "TRAINED" if done else "UNTRAINED",
                delta=status_pill("Phase 5", "ok") if done else status_pill("Pending", "pending"),
                delta_positive=done,
                accent=NEON_GREEN if done else NEON_RED,
            ),
            unsafe_allow_html=True,
        )

    if status["matches_processed"] == 0:
        st.warning(
            "No match data found.  Run the pipeline first:\n"
            "```\npython scripts/run_pipeline.py --input data/raw/fus_FAR.csv\n```",
            icon="⚠️",
        )
        return

    st.divider()

    # ── Match inventory ───────────────────────────────────────────────────────
    section_header("Processed Matches")
    match_ids = list_match_ids()

    rows = []
    for mid in match_ids:
        team_ids = get_team_ids(mid)
        home = get_team_name(mid, team_ids[0]) if team_ids else "—"
        away = get_team_name(mid, team_ids[1]) if len(team_ids) > 1 else "—"
        ts   = load_team_summary(mid)
        vaep_total = (
            float(ts["vaep_total"].sum())
            if ts is not None and "vaep_total" in ts.columns
            else 0.0
        )
        xg_total = (
            float(ts["xg_total"].sum())
            if ts is not None and "xg_total" in ts.columns
            else 0.0
        )
        rows.append({
            "Match ID":        mid,
            "Home":            home,
            "Away":            away,
            "Total VAEP":      round(vaep_total, 3),
            "Total xG":        round(xg_total, 3),
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── Cross-match player leaderboard ────────────────────────────────────────
    section_header("Top Players — All Matches")
    all_players: list[pd.DataFrame] = []
    for mid in match_ids:
        ps = load_player_summary(mid)
        if ps is not None and not ps.empty:
            ps = ps.copy()
            ps["match_id"] = mid
            all_players.append(ps)

    if all_players:
        combined = pd.concat(all_players, ignore_index=True)
        if "vaep_total" in combined.columns:
            top = combined.nlargest(10, "vaep_total")
            display_cols = [
                c for c in [
                    "player_name", "team_name", "match_id",
                    "action_count", "vaep_total", "vaep_offensive",
                    "vaep_defensive", "xg_total",
                ]
                if c in top.columns
            ]
            st.dataframe(
                top[display_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

            # Bar chart with design system colours
            bar_colors = [GOLD] + [NEON_GREEN] * 9
            fig = go.Figure(go.Bar(
                x=top["player_name"].tolist(),
                y=top["vaep_total"].tolist(),
                marker_color=bar_colors[:len(top)],
                text=[f"{v:.3f}" for v in top["vaep_total"].tolist()],
                textposition="outside",
                textfont=dict(
                    size=11,
                    color=TEXT_PRIMARY,
                    family="JetBrains Mono, monospace",
                ),
                hovertemplate="%{x}: %{y:.3f} VAEP<extra></extra>",
            ))
            fig.update_layout(
                title=dict(
                    text="TOP 10 — VAEP TOTAL",
                    font=dict(
                        size=14,
                        color=TEXT_PRIMARY,
                        family="Barlow Condensed, sans-serif",
                    ),
                ),
                xaxis=dict(
                    title="",
                    tickangle=-30,
                    showgrid=False,
                    tickfont=dict(
                        color=TEXT_MUTED, size=10,
                        family="Barlow Condensed, sans-serif",
                    ),
                ),
                yaxis=dict(
                    title="VAEP Total",
                    showgrid=True,
                    gridcolor=GRID_COLOR,
                    tickfont=dict(
                        color=TEXT_MUTED,
                        family="JetBrains Mono, monospace",
                    ),
                ),
                height=400,
                margin=dict(t=55, b=70, l=55, r=20),
                plot_bgcolor=BG_PRIMARY,
                paper_bgcolor=BG_PRIMARY,
                font=dict(color=TEXT_PRIMARY),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Platform modules ──────────────────────────────────────────────────────
    section_header("Platform Modules")
    modules: dict[str, str] = {
        "Data Ingestion":        "CSV / JSON → SPADL normalisation → Parquet store",
        "Possession Detection":  "Automated possession & action-chain segmentation",
        "Feature Engineering":   "105-dimensional feature vectors per game state",
        "xG Model":              "Logistic regression — shot quality estimation",
        "xT Grid":               "Markov-chain — expected threat per pitch zone",
        "P_scores / P_concedes": "XGBoost — VAEP probability models",
        "VAEP Engine":           "V(ai) = V(Si) - V(Si-1) per action",
        "Tactical Intelligence": "Weakness detection · formation analysis · player ranking",
        "FastAPI Backend":       "REST API exposing all analytics",
    }
    for name, desc in modules.items():
        st.markdown(
            f"<span style='color:#00E5A0; font-weight:700; "
            f"font-family:Barlow Condensed, sans-serif;'>{name}</span>"
            f"<span style='color:#64748B; font-family:Barlow Condensed, sans-serif;'>"
            f" — {desc}</span>",
            unsafe_allow_html=True,
        )
