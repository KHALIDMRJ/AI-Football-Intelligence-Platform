"""
Dashboard styling — tactical war room aesthetic.

Design system tokens, shared CSS, and HTML component helpers.
Aesthetic: Dark premium · neon-accented · data-driven · VAR room.

Usage
-----
    from football_ai.dashboard.style import inject_css, HOME_COLOR, AWAY_COLOR, kpi_card
    inject_css()
    st.markdown(
        kpi_card("xG Diff", "+0.42", delta="Home leads", delta_positive=True),
        unsafe_allow_html=True,
    )
"""

from __future__ import annotations

import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# Design Tokens
# ═══════════════════════════════════════════════════════════════════════════════

# ── Backgrounds ──────────────────────────────────────────────────────────────
BG_PRIMARY    = "#0A0E1A"   # Page background — near-black navy
BG_SURFACE    = "#1C2333"   # Card / panel surfaces
BG_ELEVATED   = "#232B3E"   # Elevated surfaces (hover, modals)
BG_OVERLAY    = "#0D1220"   # Overlay panels

# ── Accent palette ───────────────────────────────────────────────────────────
NEON_GREEN    = "#00E5A0"   # Primary accent — football neon
NEON_CYAN     = "#00B8D4"   # Secondary accent — data / info
NEON_AMBER    = "#FFB300"   # Warning / possession
NEON_RED      = "#FF4C6B"   # Danger / negative / away team

# ── Team colours ─────────────────────────────────────────────────────────────
HOME_COLOR    = "#00B8D4"   # Cyan — home team
AWAY_COLOR    = "#FF4C6B"   # Vibrant red — away team

# ── Semantic colours ─────────────────────────────────────────────────────────
ACCENT_AMBER  = NEON_AMBER
ACCENT_PURPLE = "#A78BFA"   # Tactical / formations
ACCENT_TEAL   = "#14B8A6"   # Pressure / pressing
ACCENT_GREEN  = NEON_GREEN  # Positive / formations
PITCH_GREEN   = "#0C2A1A"   # Pitch background (dark)

# ── Medal colours ────────────────────────────────────────────────────────────
GOLD          = "#FFD700"
SILVER        = "#94A3B8"
BRONZE        = "#CD7F32"

# ── Surface / layout ────────────────────────────────────────────────────────
SURFACE_DARK  = BG_PRIMARY
CARD_BG       = BG_SURFACE
GRID_COLOR    = "#1E293B"
BORDER_SUBTLE = "rgba(0,229,160,0.12)"

# ── Text ─────────────────────────────────────────────────────────────────────
TEXT_PRIMARY  = "#E2E8F0"
TEXT_MUTED    = "#64748B"
TEXT_DIM      = "#475569"

# ═══════════════════════════════════════════════════════════════════════════════
# Global CSS
# ═══════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   Google Fonts — Barlow Condensed + JetBrains Mono
═══════════════════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ═══════════════════════════════════════════════════════════════════════════
   CSS Custom Properties — Design Tokens
═══════════════════════════════════════════════════════════════════════════ */
:root {
    --bg-primary: #0A0E1A;
    --bg-surface: #1C2333;
    --bg-elevated: #232B3E;
    --bg-overlay: #0D1220;
    --neon-green: #00E5A0;
    --neon-cyan: #00B8D4;
    --neon-amber: #FFB300;
    --neon-red: #FF4C6B;
    --text-primary: #E2E8F0;
    --text-muted: #64748B;
    --text-dim: #475569;
    --border-subtle: rgba(0,229,160,0.12);
    --grid-color: #1E293B;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-xl: 18px;
    --shadow-card: 0 2px 12px rgba(0,0,0,0.5), 0 0 0 1px rgba(0,229,160,0.06);
    --shadow-glow: 0 0 20px rgba(0,229,160,0.12);
    --font-display: 'Barlow Condensed', 'Impact', sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    --font-body: 'Barlow Condensed', -apple-system, sans-serif;
    --space-1: 4px;
    --space-2: 8px;
    --space-3: 12px;
    --space-4: 16px;
    --space-6: 24px;
    --space-8: 32px;
    --space-12: 48px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Reduced motion support
═══════════════════════════════════════════════════════════════════════════ */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Custom scrollbar
═══════════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--neon-green) 0%, var(--neon-cyan) 100%);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--neon-green); }
* { scrollbar-width: thin; scrollbar-color: var(--neon-green) var(--bg-primary); }

/* ═══════════════════════════════════════════════════════════════════════════
   Global transitions
═══════════════════════════════════════════════════════════════════════════ */
*, *::before, *::after {
    transition-property: background, box-shadow, border-color, transform, opacity;
    transition-duration: 0.22s;
    transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Sidebar — frosted glass tactical panel
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: rgba(10, 14, 26, 0.88) !important;
    backdrop-filter: blur(24px) saturate(1.8);
    -webkit-backdrop-filter: blur(24px) saturate(1.8);
    border-right: 1px solid rgba(0,229,160,0.1);
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,160,0.4), transparent);
}
[data-testid="stSidebar"]::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,184,212,0.2), transparent);
}
[data-testid="stSidebar"] .stRadio label {
    color: var(--text-primary) !important;
    font-family: var(--font-display) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}
[data-testid="stSidebar"] .stCaption { color: var(--text-dim); }

/* ═══════════════════════════════════════════════════════════════════════════
   Main container
═══════════════════════════════════════════════════════════════════════════ */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    font-family: var(--font-body);
}
[data-testid="stHeader"] { display: none; }
.stSelectbox label {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-family: var(--font-display);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem;
    font-weight: 700;
    font-family: var(--font-display);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Scanline overlay — subtle CRT/tactical screen effect
═══════════════════════════════════════════════════════════════════════════ */
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 9999;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.015) 2px,
        rgba(0,0,0,0.015) 4px
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   KPI Intelligence Cards — tactical glow
═══════════════════════════════════════════════════════════════════════════ */
.kpi-row {
    display: flex;
    gap: var(--space-3);
    margin-bottom: var(--space-3);
    flex-wrap: nowrap;
}
.kpi-card {
    flex: 1;
    background: rgba(28, 35, 51, 0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: var(--radius-lg);
    padding: 16px 18px;
    border-left: 3px solid var(--card-accent, var(--neon-green));
    border-top: 1px solid rgba(255,255,255,0.03);
    box-shadow: var(--shadow-card);
    min-width: 0;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(
        135deg,
        rgba(0,229,160,0.03) 0%,
        transparent 50%
    );
    pointer-events: none;
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg,
        var(--card-accent, var(--neon-green)),
        transparent
    );
    opacity: 0.15;
}
.kpi-card:hover {
    box-shadow:
        0 4px 24px rgba(0,0,0,0.6),
        0 0 24px rgba(0,229,160,0.1);
    transform: translateY(-2px);
    border-top-color: rgba(0,229,160,0.08);
}
.kpi-label {
    font-family: var(--font-display);
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-value {
    font-family: var(--font-mono);
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.15;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-delta {
    font-family: var(--font-display);
    font-size: 0.7rem;
    margin-top: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    letter-spacing: 0.02em;
}
.kpi-delta.positive { color: #34D399; }
.kpi-delta.negative { color: var(--neon-red); }
.kpi-delta.neutral  { color: var(--text-muted); }

/* ═══════════════════════════════════════════════════════════════════════════
   Section headers — neon-edged
═══════════════════════════════════════════════════════════════════════════ */
.section-header {
    font-family: var(--font-display);
    font-size: 1.05rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-primary);
    padding: 8px 0 7px 14px;
    border-left: 3px solid var(--neon-green);
    margin: 20px 0 12px 0;
    background: linear-gradient(90deg, rgba(0,229,160,0.08) 0%, transparent 60%);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    position: relative;
}
.section-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg,
        rgba(0,229,160,0.2), transparent 80%);
}
.section-header:hover {
    background: linear-gradient(90deg, rgba(0,229,160,0.12) 0%, transparent 60%);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Match banner — command center header
═══════════════════════════════════════════════════════════════════════════ */
.match-banner {
    text-align: center;
    padding: 22px 28px 16px;
    background: linear-gradient(135deg, #080C16 0%, #111827 50%, #080C16 100%);
    border-radius: var(--radius-xl);
    margin-bottom: 20px;
    border: 1px solid rgba(0,229,160,0.2);
    position: relative;
    overflow: hidden;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.03),
        0 0 40px rgba(0,229,160,0.06);
}
.match-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
        transparent 10%,
        var(--neon-green) 50%,
        transparent 90%);
    opacity: 0.4;
}
.match-banner::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle,
        rgba(0,229,160,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.match-banner:hover {
    border-color: rgba(0,229,160,0.35);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.05),
        0 0 50px rgba(0,229,160,0.1);
}
.match-banner-title {
    font-family: var(--font-display);
    font-size: 1.9rem;
    font-weight: 900;
    margin: 0 0 4px 0;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.match-banner-subtitle {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-dim);
    margin: 0;
    letter-spacing: 0.08em;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Team badges
═══════════════════════════════════════════════════════════════════════════ */
.team-badge {
    display: inline-block;
    padding: 3px 14px;
    border-radius: 20px;
    font-family: var(--font-display);
    font-size: 0.78rem;
    font-weight: 700;
    color: white;
    margin-bottom: 8px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.team-badge.home {
    background: var(--neon-cyan);
    box-shadow: 0 0 12px rgba(0,184,212,0.3);
}
.team-badge.away {
    background: var(--neon-red);
    box-shadow: 0 0 12px rgba(255,76,107,0.3);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Shimmer button — neon edition
═══════════════════════════════════════════════════════════════════════════ */
.shimmer-btn {
    display: inline-block;
    padding: 8px 24px;
    background: linear-gradient(135deg, var(--neon-green) 0%, var(--neon-cyan) 100%);
    color: #0A0E1A;
    border: none;
    border-radius: var(--radius-md);
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 0.85rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
.shimmer-btn::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(
        90deg, transparent, rgba(255,255,255,0.25), transparent
    );
    animation: shimmer 2.2s infinite;
}
.shimmer-btn:hover {
    box-shadow: 0 0 20px rgba(0,229,160,0.35);
    transform: translateY(-1px);
}
@keyframes shimmer {
    0%   { left: -100%; }
    100% { left: 100%; }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Animations — fade-in with stagger
═══════════════════════════════════════════════════════════════════════════ */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 8px rgba(0,229,160,0.15); }
    50%      { box-shadow: 0 0 20px rgba(0,229,160,0.3); }
}
.fade-in {
    animation: fadeInUp 0.45s ease-out both;
}

/* KPI card stagger */
.kpi-row .kpi-card:nth-child(1) { animation: fadeInUp 0.4s ease-out 0.05s both; }
.kpi-row .kpi-card:nth-child(2) { animation: fadeInUp 0.4s ease-out 0.12s both; }
.kpi-row .kpi-card:nth-child(3) { animation: fadeInUp 0.4s ease-out 0.19s both; }
.kpi-row .kpi-card:nth-child(4) { animation: fadeInUp 0.4s ease-out 0.26s both; }
.kpi-row .kpi-card:nth-child(5) { animation: fadeInUp 0.4s ease-out 0.33s both; }
.kpi-row .kpi-card:nth-child(6) { animation: fadeInUp 0.4s ease-out 0.40s both; }
.kpi-row .kpi-card:nth-child(7) { animation: fadeInUp 0.4s ease-out 0.47s both; }
.kpi-row .kpi-card:nth-child(8) { animation: fadeInUp 0.4s ease-out 0.54s both; }

/* ═══════════════════════════════════════════════════════════════════════════
   Data tables — tactical grid
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--grid-color) !important;
}
[data-testid="stDataFrame"] table {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Dividers — subtle neon
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stHorizontalRule"] hr,
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg,
        transparent, var(--grid-color), transparent
    ) !important;
    margin: 16px 0 !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Status pill — live indicator
═══════════════════════════════════════════════════════════════════════════ */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 12px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.status-pill.ok {
    background: rgba(0,229,160,0.12);
    color: var(--neon-green);
    border: 1px solid rgba(0,229,160,0.2);
}
.status-pill.pending {
    background: rgba(255,179,0,0.1);
    color: var(--neon-amber);
    border: 1px solid rgba(255,179,0,0.2);
}
.status-pill.error {
    background: rgba(255,76,107,0.1);
    color: var(--neon-red);
    border: 1px solid rgba(255,76,107,0.2);
}
.status-pill .pulse {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: pulseDot 1.6s ease-in-out infinite;
}
@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.7); }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Metric overrides — monospace values
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    color: var(--text-primary) !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--font-display) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Warning / info boxes — dark treatment
═══════════════════════════════════════════════════════════════════════════ */
.stAlert {
    font-family: var(--font-display) !important;
    border-radius: var(--radius-md) !important;
}
</style>
"""

# ── Count-up animation JS ────────────────────────────────────────────────────
_COUNTUP_JS = """
<script>
(function() {
    function animateCountUp() {
        document.querySelectorAll('.kpi-value[data-countup]').forEach(function(el) {
            if (el.dataset.animated) return;
            el.dataset.animated = '1';
            var raw = el.dataset.countup;
            var isFloat = raw.indexOf('.') !== -1;
            var target = parseFloat(raw);
            if (isNaN(target)) return;
            var prefix = el.dataset.prefix || '';
            var suffix = el.dataset.suffix || '';
            var decimals = isFloat ? (raw.split('.')[1] || '').length : 0;
            var duration = 1100;
            var start = performance.now();
            var startVal = 0;
            function step(now) {
                var elapsed = now - start;
                var progress = Math.min(elapsed / duration, 1);
                var eased = 1 - Math.pow(1 - progress, 3);
                var current = startVal + (target - startVal) * eased;
                el.textContent = prefix + current.toFixed(decimals) + suffix;
                if (progress < 1) requestAnimationFrame(step);
            }
            requestAnimationFrame(step);
        });
    }
    animateCountUp();
    var obs = new MutationObserver(function() { animateCountUp(); });
    obs.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""


def inject_css() -> None:
    """Inject the shared dashboard CSS. Call once at the top of each page render."""
    st.markdown(_CSS, unsafe_allow_html=True)


def kpi_card(
    label: str,
    value: str,
    delta: str = "",
    delta_positive: bool | None = None,
    accent: str = NEON_GREEN,
) -> str:
    """
    Return an HTML string for a single KPI card.

    Parameters
    ----------
    label          : Short uppercase label shown above the value.
    value          : Primary metric value (displayed large).
    delta          : Small sub-text below the value (optional).
    delta_positive : True -> green, False -> red, None -> muted grey.
    accent         : Left-border accent colour.
    """
    delta_class = "neutral"
    if delta_positive is True:
        delta_class = "positive"
    elif delta_positive is False:
        delta_class = "negative"

    delta_html = (
        f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    )

    return (
        f'<div class="kpi-card" style="--card-accent:{accent};">'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="kpi-value">{value}</div>'
        f'  {delta_html}'
        f'</div>'
    )


def section_header(title: str) -> None:
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def match_banner(home_name: str, away_name: str, match_id: str) -> None:
    """Render a full-width match header banner."""
    st.markdown(
        f'<div class="match-banner">'
        f'  <div class="match-banner-title">'
        f'    <span style="color:{HOME_COLOR};">{home_name}</span>'
        f'    <span style="color:{TEXT_DIM}; font-size:1rem; margin:0 16px;">VS</span>'
        f'    <span style="color:{AWAY_COLOR};">{away_name}</span>'
        f'  </div>'
        f'  <p class="match-banner-subtitle">MATCH &middot; {match_id}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def status_pill(label: str, status: str = "ok") -> str:
    """Return an HTML status pill. status: 'ok', 'pending', or 'error'."""
    pulse = '<span class="pulse"></span>' if status == "ok" else ""
    return f'<span class="status-pill {status}">{pulse}{label}</span>'


def plotly_layout_defaults() -> dict:
    """Return a dict of Plotly layout defaults matching the design system."""
    return dict(
        plot_bgcolor=BG_PRIMARY,
        paper_bgcolor=BG_PRIMARY,
        font=dict(
            family="Barlow Condensed, sans-serif",
            color=TEXT_PRIMARY,
            size=12,
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_MUTED),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_MUTED),
        ),
        legend=dict(
            font=dict(size=11, color=TEXT_PRIMARY),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=40, b=40, l=50, r=20),
    )
