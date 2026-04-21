"""
Per-request context variables.

``request_id`` and ``user_id`` are set by :class:`RequestContextMiddleware`
at the start of every HTTP request and cleared at the end. Because
they're :class:`ContextVar`-backed, they survive ``await`` boundaries
inside a single request without callers having to thread the values
through every function signature — logs and metrics read them directly.

Why not just use logging filters alone?
---------------------------------------
A filter on the log record alone doesn't help the metrics layer (which
wants to tag samples with user tier) or background tasks that want to
inherit the originating request id. A :class:`ContextVar` is the one
mechanism that plays nicely with ``asyncio.Task`` and thread pools.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

# Module-level context vars — `default=None` so tests and workers that
# never set them still read sensibly.
_request_id: ContextVar[str | None] = ContextVar("football_ai_request_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("football_ai_user_id", default=None)


def bind_request_id(request_id: str | None = None) -> str:
    """Store a request id for the current async context.

    When ``request_id`` is falsy we generate a new UUID4 — but if the
    caller hands in an incoming ``X-Request-ID`` header value we preserve
    it, so traces correlate across service boundaries.
    """
    rid = request_id or uuid.uuid4().hex
    _request_id.set(rid)
    return rid


def bind_user_id(user_id: str | None) -> None:
    """Attach the authenticated user id to the current context."""
    _user_id.set(user_id)


def current_request_id() -> str | None:
    return _request_id.get()


def current_user_id() -> str | None:
    return _user_id.get()


def clear_request_context() -> None:
    """Reset the context to its default empty state."""
    _request_id.set(None)
    _user_id.set(None)
