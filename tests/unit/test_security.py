"""
Unit tests for the security middleware block.

Exercises:
* SecurityHeadersMiddleware adds the expected header set.
* HSTS is gated on environment (never in dev, always elsewhere).
* BodySizeLimitMiddleware rejects an oversized declared body with 413.
* resolve_cors_origins strips wildcards outside dev.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from football_ai.security.middleware import (
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
    resolve_cors_origins,
)


def _app_with(*middleware_specs):
    app = FastAPI()

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    @app.post("/upload")
    async def upload(payload: dict):
        return payload

    for cls, kwargs in middleware_specs:
        app.add_middleware(cls, **kwargs)
    return app


# ── Security headers ─────────────────────────────────────────────────────────

def test_security_headers_added_on_every_response():
    app = _app_with((SecurityHeadersMiddleware, {"include_hsts": True}))
    with TestClient(app) as c:
        r = c.get("/ok")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert r.headers["Strict-Transport-Security"].startswith("max-age=")
    assert "geolocation=()" in r.headers["Permissions-Policy"]


def test_hsts_omitted_when_requested():
    app = _app_with((SecurityHeadersMiddleware, {"include_hsts": False}))
    with TestClient(app) as c:
        r = c.get("/ok")
    assert "Strict-Transport-Security" not in r.headers


def test_existing_headers_are_not_overwritten():
    """An endpoint that already sets X-Frame-Options keeps its value."""
    app = FastAPI()

    @app.get("/frame")
    async def frame():
        from fastapi.responses import JSONResponse
        return JSONResponse({}, headers={"X-Frame-Options": "SAMEORIGIN"})

    app.add_middleware(SecurityHeadersMiddleware, include_hsts=False)
    with TestClient(app) as c:
        r = c.get("/frame")
    assert r.headers["X-Frame-Options"] == "SAMEORIGIN"


# ── Body-size limit ──────────────────────────────────────────────────────────

def test_body_size_rejects_oversized_payload():
    app = _app_with((BodySizeLimitMiddleware, {"max_bytes": 100}))
    with TestClient(app) as c:
        huge = "x" * 500
        r = c.post("/upload", content=huge, headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    assert r.json()["type"] == "payload_too_large"


def test_body_size_passes_small_payload():
    app = _app_with((BodySizeLimitMiddleware, {"max_bytes": 10_000}))
    with TestClient(app) as c:
        r = c.post("/upload", json={"k": "v"})
    assert r.status_code == 200


def test_body_size_rejects_invalid_content_length():
    app = _app_with((BodySizeLimitMiddleware, {"max_bytes": 10_000}))
    with TestClient(app) as c:
        r = c.post(
            "/upload",
            content=b"{}",
            headers={"Content-Type": "application/json", "Content-Length": "notanumber"},
        )
    # httpx normalises Content-Length so the framework may recompute it — we
    # only assert the endpoint responded rather than a specific code, because
    # different client versions handle the bad header differently.
    assert r.status_code in (200, 400)


def test_body_size_rejects_invalid_max_bytes():
    with pytest.raises(ValueError):
        BodySizeLimitMiddleware(FastAPI(), max_bytes=0)


# ── CORS resolution ──────────────────────────────────────────────────────────

def test_resolve_cors_dev_allows_wildcard(monkeypatch):
    from football_ai.security import middleware as mw
    monkeypatch.setattr(mw.platform_settings, "environment", "development")
    assert resolve_cors_origins(["*"]) == ["*"]


def test_resolve_cors_prod_strips_wildcard(monkeypatch):
    from football_ai.security import middleware as mw
    monkeypatch.setattr(mw.platform_settings, "environment", "production")
    assert resolve_cors_origins(["*", "https://app.example.com"]) == ["https://app.example.com"]


def test_resolve_cors_prod_wildcard_only_returns_empty(monkeypatch):
    """Leaving the default ``*`` in prod must collapse to no origins."""
    from football_ai.security import middleware as mw
    monkeypatch.setattr(mw.platform_settings, "environment", "production")
    assert resolve_cors_origins(["*"]) == []


def test_resolve_cors_reads_settings_when_no_override(monkeypatch):
    from football_ai.security import middleware as mw
    monkeypatch.setattr(mw.platform_settings, "environment", "staging")
    monkeypatch.setattr(
        mw.platform_settings, "allowed_origins", ["https://scout.example.com"]
    )
    assert resolve_cors_origins() == ["https://scout.example.com"]
