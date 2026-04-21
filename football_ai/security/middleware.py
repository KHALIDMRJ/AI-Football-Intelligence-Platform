"""
Security middlewares.

:class:`SecurityHeadersMiddleware` adds the browser-facing defaults that
Mozilla's Observatory and OWASP Cheat Sheet flag as must-haves: HSTS,
nosniff, Referrer-Policy, clickjacking protection. The exposition is
opinionated but conservative — nothing that would break the Swagger UI
or the Streamlit dashboard (the main two first-party front-ends).

:class:`BodySizeLimitMiddleware` rejects requests whose ``Content-Length``
exceeds ``max_bytes`` with 413. This protects the upload path (e.g. the
Phase-7 CSV ingest route) from a client shipping a multi-GB payload that
would otherwise be streamed into memory before FastAPI validates it.

:func:`resolve_cors_origins` picks a safe ``allow_origins`` list from
``platform_settings`` based on environment:

* ``development`` or ``test`` → ``["*"]`` (ergonomic local dev)
* anything else → ``settings.allowed_origins`` with ``"*"`` stripped.
  If the user left the default (``["*"]``) in a non-dev environment, we
  fall back to a safe empty list and log a warning — misconfiguration
  is caught loudly instead of silently accepting any origin in prod.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from football_ai.config import platform_settings
from football_ai.logger import get_logger

logger = get_logger(__name__)


# ── Security headers ─────────────────────────────────────────────────────────

# 63072000 s = 2 years, the HSTS value Mozilla recommends for "preload"
# submission. Safe because we control all sub-paths and assume HTTPS in
# any non-dev environment.
_HSTS_MAX_AGE = 63072000

_DEFAULT_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # Only advertise the permissions we actually use. Tight defaults that
    # callers can loosen per-response if a future endpoint needs them.
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Stamp a fixed block of security headers on every response.

    HSTS is only emitted when ``environment != "development"`` so local
    HTTP testing still works (browsers cache HSTS aggressively and a
    dev opt-in ruins the next prod visit from the same laptop).
    """

    def __init__(self, app, *, include_hsts: bool | None = None) -> None:
        super().__init__(app)
        if include_hsts is None:
            include_hsts = platform_settings.environment != "development"
        self._include_hsts = include_hsts

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        for name, value in _DEFAULT_HEADERS.items():
            # Don't clobber if an endpoint has deliberately set its own value.
            response.headers.setdefault(name, value)
        if self._include_hsts:
            response.headers.setdefault(
                "Strict-Transport-Security",
                f"max-age={_HSTS_MAX_AGE}; includeSubDomains",
            )
        return response


# ── Body-size limit ──────────────────────────────────────────────────────────

class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose ``Content-Length`` exceeds ``max_bytes``.

    We only inspect the header — we don't read the body — so a truthful
    client gets a fast 413 and a lying client still gets caught later by
    FastAPI's own body-parsing guards. That's deliberate: reading the
    body here would double the memory footprint of every legitimate
    request for a guard that's cheap to do at the edge (Nginx / Envoy).
    """

    def __init__(self, app, *, max_bytes: int) -> None:
        super().__init__(app)
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                n = int(content_length)
            except ValueError:
                return JSONResponse(
                    {"detail": "Invalid Content-Length header.", "status": 400,
                     "type": "invalid_request"},
                    status_code=400,
                )
            if n > self._max_bytes:
                return JSONResponse(
                    {
                        "detail": (
                            f"Request body too large: {n} bytes "
                            f"(max {self._max_bytes})."
                        ),
                        "status": 413,
                        "type": "payload_too_large",
                    },
                    status_code=413,
                )
        return await call_next(request)


# ── CORS resolution ──────────────────────────────────────────────────────────

def resolve_cors_origins(origins: Iterable[str] | None = None) -> list[str]:
    """Return the effective CORS ``allow_origins`` list.

    Args:
        origins: Override to skip the settings lookup (used by tests).

    In dev/test we accept ``*`` so developers can hit the API from any
    port; anywhere else we refuse wildcard origins and log a warning
    when the user left the default in place — a misconfiguration is
    better caught by an empty origin list (CORS calls fail loudly)
    than by a silent "allow anything" in production.
    """
    env = platform_settings.environment
    src = list(origins) if origins is not None else list(platform_settings.allowed_origins)
    if env in ("development", "test"):
        return src or ["*"]

    safe = [o for o in src if o and o != "*"]
    if not safe:
        logger.warning(
            "CORS: no explicit allowed_origins configured for env=%s; "
            "defaulting to an empty list (no cross-origin calls will succeed).",
            env,
        )
    return safe
