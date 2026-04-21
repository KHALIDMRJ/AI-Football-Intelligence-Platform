"""
Security middleware — headers + request size ceiling.

The pieces here are deliberately narrow: they sit between the
observability middleware (which stamps request ids) and the router, and
they refuse malformed traffic before the app's expensive work starts.
"""

from .middleware import (
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
    resolve_cors_origins,
)

__all__ = [
    "SecurityHeadersMiddleware",
    "BodySizeLimitMiddleware",
    "resolve_cors_origins",
]
