"""
Prometheus metrics — request, prediction, and cache instruments.

Exposure
--------
The scrape endpoint lives at ``/metrics`` (registered in
``football_ai.api.endpoints.health``). Prometheus text format by default
so any vanilla scrape config works with zero configuration.

Cardinality
-----------
Labels are chosen to be LOW-cardinality:

* ``method`` — HTTP verb (bounded small set).
* ``route``  — the **template** (``/api/v1/players/catalog/{player_id}``)
  NOT the concrete path. Otherwise every UUID creates a new series.
* ``status`` — HTTP status code.

The middleware pulls the route template from the matched FastAPI route
(``request.scope["route"].path``). Unmatched paths land in a bucket
labelled ``"unmatched"`` so misbehaving clients don't shatter the
series table.
"""

from __future__ import annotations

from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

# Own registry so tests can run in isolation without clobbering the
# global default registry (handy when repeatedly importing app factories).
REGISTRY = CollectorRegistry()


REQUESTS_TOTAL = Counter(
    "football_ai_http_requests_total",
    "HTTP requests handled, labeled by method/route/status.",
    labelnames=("method", "route", "status"),
    registry=REGISTRY,
)

REQUEST_DURATION = Histogram(
    "football_ai_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "route"),
    # Buckets tuned for web APIs: sub-5ms through ~10s.
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

PREDICTIONS_TOTAL = Counter(
    "football_ai_predictions_total",
    "Predictions produced by the match-outcome model.",
    labelnames=("outcome",),
    registry=REGISTRY,
)

CACHE_OPS_TOTAL = Counter(
    "football_ai_cache_ops_total",
    "Cache operations by kind/result.",
    labelnames=("op", "result"),
    registry=REGISTRY,
)


async def metrics_endpoint() -> PlainTextResponse:
    """Prometheus scrape handler. Returns the text exposition format."""
    payload = generate_latest(REGISTRY)
    return PlainTextResponse(payload, media_type=CONTENT_TYPE_LATEST)
