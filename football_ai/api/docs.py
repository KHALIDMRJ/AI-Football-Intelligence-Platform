"""
Custom Swagger UI — Phase 10 tactical war-room aesthetic.

Serves a redesigned /docs page with:
  * Sticky header (brand + version, live search, theme toggle persisted to localStorage)
  * Stats bar populated from /openapi.json (endpoint count, tag count, model version)
  * Two complete CSS variable sets (dark default, light) with 0.3s transitions
  * Per-method badge + left-border colours
  * Per-tag left-border colours
  * Live endpoint search filtering .opblock entries (and hiding empty tag groups)
  * Customised Authorize button ("🔑 Authorize JWT")
  * Footer attribution

CDN-only (unpkg + Google Fonts), no external CSS frameworks.
"""

# This module is ~98% a triple-quoted HTML/CSS/JS template. The long lines
# inside that template are inline data URIs, font URLs and CSS rules that
# cannot be wrapped without breaking what they emit. They are string content —
# not Python source — so per-line ignore directives cannot reach them.
# Disable line-length only for this template file rather than adding a
# project-wide ignore in pyproject.toml.
# ruff: noqa: E501

from __future__ import annotations

from fastapi.responses import HTMLResponse


def custom_swagger_ui_html(
    *,
    openapi_url: str,
    title: str,
    version: str,
) -> HTMLResponse:
    """Render the custom Swagger UI page.

    Args:
        openapi_url: Path to the OpenAPI JSON document (typically ``/openapi.json``).
        title: Display name for the API (rendered in the header).
        version: Application version (rendered as a chip in the header).
    """
    html = (
        _SWAGGER_TEMPLATE
        .replace("{{OPENAPI_URL}}", openapi_url)
        .replace("{{TITLE}}", title)
        .replace("{{VERSION}}", version)
    )
    return HTMLResponse(content=html, status_code=200)


_SWAGGER_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{{TITLE}} — API Docs</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ccircle cx='12' cy='12' r='10' fill='%2310B981'/%3E%3C/svg%3E" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
<script>
  // Apply persisted theme BEFORE Swagger UI loads, so there is no flash of
  // the wrong palette and so getAttribute("data-theme") is authoritative by
  // the time any other script runs.
  (function () {
    var saved = localStorage.getItem("fai_theme");
    if (saved !== "dark" && saved !== "light") { saved = "dark"; }
    document.documentElement.setAttribute("data-theme", saved);
  })();
</script>
<style>
/* ============================================================
   THEME TOKENS — two complete palettes.
   ============================================================ */
:root[data-theme="dark"] {
  --bg:        #0a0e1a;
  --bg-card:   #111827;
  --bg-input:  #1f2937;
  --border:    #1f2937;
  --text:      #f1f5f9;
  --text-sub:  #94a3b8;
  --accent:    #10b981;
  --accent2:   #6366f1;
  --code-bg:   #0d1117;
}
:root[data-theme="light"] {
  --bg:        #f8fafc;
  --bg-card:   #ffffff;
  --bg-input:  #f1f5f9;
  --border:    #e2e8f0;
  --text:      #0f172a;
  --text-sub:  #64748b;
  --accent:    #059669;
  --accent2:   #4f46e5;
  --code-bg:   #1e293b;
}

/* ============================================================
   GLOBAL — typography, transitions.
   ============================================================ */
html, body {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0;
  padding: 0;
  transition: background 0.3s ease, color 0.3s ease;
}
*, *::before, *::after {
  transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
}

/* ============================================================
   HEADER — sticky, three-column layout.
   ============================================================ */
.fai-header {
  position: sticky;
  top: 0;
  z-index: 100;
  display: grid;
  grid-template-columns: minmax(280px, 1fr) minmax(320px, 2fr) auto;
  align-items: center;
  gap: 24px;
  padding: 14px 28px;
  background: var(--bg-card);
  border-bottom: 1px solid var(--border);
  box-shadow: 0 2px 16px rgba(0, 0, 0, 0.25);
}
.fai-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 700;
  font-size: 1.05rem;
  color: var(--text);
  letter-spacing: -0.01em;
}
.fai-brand .fai-logo {
  font-size: 1.35rem;
  filter: drop-shadow(0 0 6px var(--accent));
}
.fai-version {
  display: inline-block;
  margin-left: 6px;
  padding: 2px 10px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--accent) 18%, transparent);
  color: var(--accent);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent);
}
.fai-search-wrap {
  position: relative;
}
.fai-search-wrap::before {
  content: "🔎";
  position: absolute;
  left: 14px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 0.85rem;
  opacity: 0.55;
  pointer-events: none;
}
#fai-search {
  width: 100%;
  padding: 10px 14px 10px 38px;
  background: var(--bg-input);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-family: inherit;
  font-size: 0.92rem;
  outline: none;
}
#fai-search::placeholder { color: var(--text-sub); }
#fai-search:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 22%, transparent);
}
#fai-theme {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 9px 14px;
  background: var(--bg-input);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-family: inherit;
  font-size: 0.86rem;
  font-weight: 500;
  cursor: pointer;
}
#fai-theme:hover {
  border-color: var(--accent);
  color: var(--accent);
}

/* ============================================================
   STATS BAR — derived from /openapi.json.
   ============================================================ */
.fai-stats-bar {
  padding: 10px 28px;
  background: var(--bg);
  color: var(--text-sub);
  border-bottom: 1px solid var(--border);
  font-size: 0.82rem;
  font-weight: 500;
  letter-spacing: 0.02em;
}
.fai-stats-bar strong {
  color: var(--accent);
  font-weight: 700;
}

/* ============================================================
   FOOTER — attribution.
   ============================================================ */
.fai-footer {
  padding: 18px 28px;
  margin-top: 32px;
  background: var(--bg-card);
  border-top: 1px solid var(--border);
  color: var(--text-sub);
  font-size: 0.78rem;
  text-align: center;
  letter-spacing: 0.02em;
}
.fai-footer .accent { color: var(--accent); font-weight: 600; }

/* ============================================================
   SWAGGER UI OVERRIDES — apply theme variables.
   ============================================================ */
.swagger-ui {
  font-family: inherit !important;
  color: var(--text) !important;
}
.swagger-ui .info,
.swagger-ui .scheme-container,
.swagger-ui .topbar { display: none !important; }

.swagger-ui .wrapper {
  padding: 24px 28px !important;
  max-width: 1280px;
}

/* Tag headers */
.swagger-ui .opblock-tag {
  background: var(--bg-card) !important;
  color: var(--text) !important;
  border-radius: 8px;
  border: 1px solid var(--border);
  border-left: 4px solid var(--text-sub);
  margin: 18px 0 8px 0 !important;
  padding: 12px 18px !important;
  font-size: 1.05rem !important;
  font-weight: 600 !important;
}
.swagger-ui .opblock-tag small {
  color: var(--text-sub) !important;
  font-weight: 400;
}

/* Operation blocks */
.swagger-ui .opblock {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-left-width: 4px !important;
  border-radius: 6px !important;
  margin: 0 0 8px 0 !important;
  box-shadow: none !important;
}
.swagger-ui .opblock .opblock-summary {
  padding: 10px 14px !important;
  border-bottom: 1px solid var(--border);
}
.swagger-ui .opblock .opblock-summary-path,
.swagger-ui .opblock .opblock-summary-description {
  color: var(--text) !important;
  font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
  font-size: 0.92rem;
}
.swagger-ui .opblock-summary-method {
  font-weight: 700 !important;
  border-radius: 4px !important;
  min-width: 76px;
  text-align: center;
}

/* Method colours — badge + left border */
.swagger-ui .opblock.opblock-get { border-left-color: #10b981 !important; }
.swagger-ui .opblock.opblock-get .opblock-summary-method { background: #10b981 !important; }
.swagger-ui .opblock.opblock-post { border-left-color: #6366f1 !important; }
.swagger-ui .opblock.opblock-post .opblock-summary-method { background: #6366f1 !important; }
.swagger-ui .opblock.opblock-put { border-left-color: #f59e0b !important; }
.swagger-ui .opblock.opblock-put .opblock-summary-method { background: #f59e0b !important; }
.swagger-ui .opblock.opblock-delete { border-left-color: #ef4444 !important; }
.swagger-ui .opblock.opblock-delete .opblock-summary-method { background: #ef4444 !important; }
.swagger-ui .opblock.opblock-patch { border-left-color: #8b5cf6 !important; }
.swagger-ui .opblock.opblock-patch .opblock-summary-method { background: #8b5cf6 !important; }

/* Tag-group accents — applied via data-tag attribute */
.swagger-ui .opblock-tag[data-tag="auth"]        { border-left-color: #ef4444 !important; }
.swagger-ui .opblock-tag[data-tag="players"]     { border-left-color: #10b981 !important; }
.swagger-ui .opblock-tag[data-tag="matches"]     { border-left-color: #6366f1 !important; }
.swagger-ui .opblock-tag[data-tag="teams"]       { border-left-color: #f59e0b !important; }
.swagger-ui .opblock-tag[data-tag="predictions"] { border-left-color: #8b5cf6 !important; }
.swagger-ui .opblock-tag[data-tag="analysis"]    { border-left-color: #06b6d4 !important; }
.swagger-ui .opblock-tag[data-tag="admin"]       { border-left-color: #94a3b8 !important; }
.swagger-ui .opblock-tag[data-tag="health"]      { border-left-color: #64748b !important; }
.swagger-ui .opblock-tag[data-tag="system"]      { border-left-color: #64748b !important; }

/* Try-it-out panel */
.swagger-ui .opblock-body {
  background: var(--bg-card) !important;
  color: var(--text) !important;
}
.swagger-ui .opblock-section-header {
  background: var(--bg-input) !important;
  box-shadow: none !important;
  border-bottom: 1px solid var(--border);
}
.swagger-ui .opblock-section-header h4,
.swagger-ui .opblock-description-wrapper p,
.swagger-ui .parameter__name,
.swagger-ui .parameter__type,
.swagger-ui table thead tr td,
.swagger-ui table thead tr th,
.swagger-ui .response-col_status,
.swagger-ui .response-col_description__inner div.markdown,
.swagger-ui .response-col_description__inner div.renderedMarkdown {
  color: var(--text) !important;
}

.swagger-ui input[type="text"],
.swagger-ui input[type="email"],
.swagger-ui input[type="password"],
.swagger-ui textarea,
.swagger-ui select {
  background: var(--bg-input) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}

.swagger-ui .highlight-code,
.swagger-ui pre,
.swagger-ui .microlight,
.swagger-ui .responses-inner pre {
  background: var(--code-bg) !important;
  color: #f1f5f9 !important;
  border-radius: 6px;
}

.swagger-ui .response .response-col_status { font-weight: 700; }
.swagger-ui .response-control-media-type__accept-message { color: var(--text-sub) !important; }

/* Try-it-out / Execute button */
.swagger-ui .btn.execute {
  background: var(--accent2) !important;
  border-color: var(--accent2) !important;
  color: #ffffff !important;
}
.swagger-ui .btn.try-out__btn,
.swagger-ui .btn.cancel {
  background: var(--bg-input) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}

/* Authorize button — prominent */
.swagger-ui .btn.authorize,
.swagger-ui .auth-wrapper .btn.authorize {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  padding: 8px 18px !important;
  border-radius: 6px !important;
}
.swagger-ui .btn.authorize svg { fill: #ffffff !important; }
.swagger-ui .auth-container input[type="text"],
.swagger-ui .auth-container input[type="password"] {
  background: var(--bg-input) !important;
  color: var(--text) !important;
}

/* Models — collapsed by default via defaultModelsExpandDepth: -1 */
.swagger-ui section.models {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
}

/* Filter input — Swagger's built-in (we hide it; ours lives in the header) */
.swagger-ui .filter-container { display: none !important; }
</style>
</head>
<body>
<header class="fai-header">
  <div class="fai-brand">
    <span class="fai-logo">⚽</span>
    Football AI Intelligence Platform
    <span class="fai-version">v{{VERSION}}</span>
  </div>
  <div class="fai-search-wrap">
    <input id="fai-search" type="search" autocomplete="off" placeholder="Search endpoints — try /predictions, /players, /admin…" />
  </div>
  <button id="fai-theme" type="button" aria-label="Toggle colour theme">🌙 Dark</button>
</header>
<div class="fai-stats-bar" id="fai-stats">Loading endpoint stats…</div>
<div id="swagger-ui"></div>
<footer class="fai-footer">
  <span class="accent">⚽ Football AI Intelligence Platform</span>
  &nbsp;•&nbsp; FastAPI + XGBoost + VAEP + Claude API
  &nbsp;•&nbsp; Phase 10/12
</footer>

<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
<script>
(function () {
  const STORAGE_KEY = "fai_theme";
  const themeBtn = document.getElementById("fai-theme");

  // ── Theme toggle ──────────────────────────────────────────────────────────
  // The initial attribute was set by the inline <head> script before paint.
  // Here we only wire the button to flip it and persist each change.
  function applyTheme(theme) {
    const next = theme === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem(STORAGE_KEY, next);
    themeBtn.textContent = next === "dark" ? "🌙 Dark" : "☀️ Light";
  }
  applyTheme(document.documentElement.getAttribute("data-theme") || "dark");

  themeBtn.addEventListener("click", function () {
    const current = document.documentElement.getAttribute("data-theme");
    applyTheme(current === "dark" ? "light" : "dark");
  });

  // ── Swagger UI bootstrap ──────────────────────────────────────────────────
  const ui = SwaggerUIBundle({
    url: "{{OPENAPI_URL}}",
    dom_id: "#swagger-ui",
    presets: [
      SwaggerUIBundle.presets.apis,
      SwaggerUIStandalonePreset
    ],
    layout: "BaseLayout",
    docExpansion: "none",
    defaultModelsExpandDepth: -1,
    syntaxHighlight: { theme: "monokai" },
    tryItOutEnabled: true,
    persistAuthorization: true,
    displayRequestDuration: true,
    filter: true,
    deepLinking: true,
    onComplete: function () {
      // Tag the .opblock-tag containers with data-tag so per-tag CSS rules hit.
      document.querySelectorAll(".opblock-tag").forEach(function (el) {
        const id = (el.getAttribute("data-tag") || el.getAttribute("id") || "")
          .replace(/^operations-tag-/, "")
          .toLowerCase();
        if (id) el.setAttribute("data-tag", id);
      });
      // Same for the parent .opblock-tag-section if Swagger UI adds them.
      document.querySelectorAll(".opblock-tag-section").forEach(function (sec) {
        const tagEl = sec.querySelector(".opblock-tag");
        if (tagEl) {
          const tag = tagEl.getAttribute("data-tag") || "";
          sec.setAttribute("data-tag", tag);
        }
      });
      // Override the Authorize button label.
      document.querySelectorAll(".btn.authorize span, .auth-wrapper .btn.authorize span")
        .forEach(function (s) { s.textContent = "Authorize JWT"; });
      const authBtns = document.querySelectorAll(".btn.authorize");
      authBtns.forEach(function (b) {
        // Preserve the lock SVG; just prefix the emoji on the visible label.
        const span = b.querySelector("span");
        if (span && !span.dataset.faiPatched) {
          span.textContent = "🔑 Authorize JWT";
          span.dataset.faiPatched = "1";
        }
      });
    },
  });

  // ── Stats bar — pull /openapi.json once and summarise ────────────────────
  fetch("{{OPENAPI_URL}}")
    .then(function (r) { return r.json(); })
    .then(function (spec) {
      const paths = spec.paths || {};
      let endpointCount = 0;
      const tagSet = new Set();
      Object.keys(paths).forEach(function (p) {
        const ops = paths[p] || {};
        Object.keys(ops).forEach(function (method) {
          if (["get", "post", "put", "patch", "delete", "options", "head"].indexOf(method) === -1) return;
          endpointCount += 1;
          const op = ops[method] || {};
          (op.tags || []).forEach(function (t) { tagSet.add(t); });
        });
      });
      const declaredTags = (spec.tags || []).map(function (t) { return t.name; });
      declaredTags.forEach(function (t) { tagSet.add(t); });
      const version = (spec.info && spec.info.version) || "{{VERSION}}";
      const el = document.getElementById("fai-stats");
      el.innerHTML =
        "<strong>" + endpointCount + "</strong> endpoints  •  " +
        "<strong>" + tagSet.size + "</strong> tags  •  " +
        "Model v<strong>" + version + "</strong>";
    })
    .catch(function () {
      document.getElementById("fai-stats").textContent = "Endpoint stats unavailable.";
    });

  // ── Live endpoint search — case-insensitive, no debounce ─────────────────
  const searchInput = document.getElementById("fai-search");
  function applyFilter() {
    const q = searchInput.value.trim().toLowerCase();
    const sections = document.querySelectorAll(".opblock-tag-section");
    sections.forEach(function (section) {
      let visible = 0;
      section.querySelectorAll(".opblock").forEach(function (op) {
        const pathEl = op.querySelector(".opblock-summary-path");
        const methodEl = op.querySelector(".opblock-summary-method");
        const path = (pathEl ? pathEl.innerText : "").toLowerCase();
        const method = (methodEl ? methodEl.innerText : "").toLowerCase();
        const match = !q || path.indexOf(q) !== -1 || method.indexOf(q) !== -1;
        op.style.display = match ? "" : "none";
        if (match) visible += 1;
      });
      // Hide the entire tag group if nothing matches inside it.
      section.style.display = (q && visible === 0) ? "none" : "";
    });
  }
  searchInput.addEventListener("keyup", applyFilter);
  searchInput.addEventListener("search", applyFilter);
})();
</script>
</body>
</html>
"""
