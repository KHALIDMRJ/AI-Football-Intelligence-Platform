"""
Request-context middleware.

Responsibilities per request
----------------------------
1. Read or generate ``X-Request-ID``; bind it to the context vars.
2. Time the request; record latency into the Prometheus histogram.
3. Emit exactly one structured access log line when the response fires
   (success or exception).
4. Echo ``X-Request-ID`` back on the response so clients (and upstream
   proxies) can correlate their side of the call.

The middleware deliberately does NOT read the auth user — that would
double-decode the JWT on every request. Instead, the endpoints call
:func:`bind_user_id` once they have a resolved ``User`` (see the
:func:`get_current_user` dep), and the access log picks it up.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from football_ai.logger import get_logger

from .context import (
    bind_request_id,
    clear_request_context,
    current_user_id,
)
from .metrics import REQUEST_DURATION, REQUESTS_TOTAL

logger = get_logger("football_ai.access")


_HEADER_NAME = "X-Request-ID"
_UNMATCHED = "unmatched"


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = bind_request_id(request.headers.get(_HEADER_NAME))
        started = time.perf_counter()
        status_code = 500  # default for the exception path
        response: Response | None = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - started
            route = _route_template(request)
            method = request.method

            REQUESTS_TOTAL.labels(method=method, route=route, status=str(status_code)).inc()
            REQUEST_DURATION.labels(method=method, route=route).observe(duration)

            logger.info(
                "%s %s -> %d (%.1fms)",
                method, request.url.path, status_code, duration * 1000,
                extra={
                    "http_method": method,
                    "path": request.url.path,
                    "route": route,
                    "status": status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "client_ip": _client_ip(request),
                    "user_id": current_user_id() or "-",
                },
            )

            if response is not None:
                response.headers[_HEADER_NAME] = request_id
            clear_request_context()


def _route_template(request: Request) -> str:
    """Return the FastAPI route template if a route matched, else ``unmatched``."""
    route = request.scope.get("route")
    if route is None:
        return _UNMATCHED
    # Starlette Route objects carry .path; Mount / WebSocketRoute differ but
    # still expose .path. Fall back defensively.
    return getattr(route, "path", _UNMATCHED)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # Only keep the left-most entry (the original client).
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "-"


def register(app: ASGIApp) -> None:
    """Install the middleware onto a FastAPI app instance."""
    # Middlewares in Starlette execute in reverse registration order, so
    # adding this one LAST means it runs FIRST — perfect for stamping
    # request ids before anything else can log.
    app.add_middleware(RequestContextMiddleware)
