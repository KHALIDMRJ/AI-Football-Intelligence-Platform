"""
Background tasks — arq-based, with direct-call fallback for tests.

The task functions in :mod:`football_ai.tasks.jobs` are plain async
functions that take ``ctx`` (the arq context dict, which always carries
at least a sessionmaker) plus their own arguments. This shape lets them
run two ways:

1. **Production**: registered with the arq worker
   (``WorkerSettings.functions``), enqueued via the admin endpoint, and
   executed off the request thread.
2. **Tests**: called directly with an ad-hoc ``ctx`` containing a fresh
   sessionmaker. No Redis, no worker process — just an awaitable.

This dual mode keeps the test surface small (no need to spin up an arq
worker in CI) without sacrificing the production deployment shape.
"""

from __future__ import annotations

from .jobs import (
    backfill_finished_predictions,
    refresh_upcoming_predictions,
    warm_caches,
)

__all__ = [
    "backfill_finished_predictions",
    "refresh_upcoming_predictions",
    "warm_caches",
]
