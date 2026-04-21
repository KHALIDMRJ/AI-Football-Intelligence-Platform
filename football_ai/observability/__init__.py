"""
Observability layer — structured logging, request context, metrics.

The three modules collaborate through :mod:`~.context`:

* :mod:`~.logging` reads the context to enrich every log record with
  request/user ids, so one request's entire log trail shares a
  correlation id without the endpoint having to pass it around.
* :mod:`~.middleware` sets the context on each incoming HTTP request and
  emits a single structured access log when the response fires.
* :mod:`~.metrics` records counters and histograms; the scrape endpoint
  lives on the infra-health router at ``/metrics``.
"""

from __future__ import annotations

from .context import (
    bind_request_id,
    bind_user_id,
    clear_request_context,
    current_request_id,
    current_user_id,
)
from .logging import JsonLogFormatter, RequestContextFilter, install_json_logging
from .metrics import (
    PREDICTIONS_TOTAL,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    metrics_endpoint,
)
from .middleware import RequestContextMiddleware

__all__ = [
    "bind_request_id",
    "bind_user_id",
    "clear_request_context",
    "current_request_id",
    "current_user_id",
    "JsonLogFormatter",
    "RequestContextFilter",
    "install_json_logging",
    "PREDICTIONS_TOTAL",
    "REQUESTS_TOTAL",
    "REQUEST_DURATION",
    "metrics_endpoint",
    "RequestContextMiddleware",
]
