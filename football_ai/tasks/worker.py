"""
arq worker configuration.

Run with::

    arq football_ai.tasks.worker.WorkerSettings

The worker process imports our DB session factory at startup so jobs
can hand it out via ``ctx``. Cron schedules are intentionally light:

* Predictions backfill — every 30 min (catches finished fixtures fast)
* Predictions refresh  — every 4 h (form data doesn't move that fast)
* Cache warm           — daily at 03:00 UTC (off-peak globally)

These cadences mirror what a small production deployment would set;
heavier setups would push refresh to every hour and add per-league
priorities. For the portfolio, this proves the wiring without
over-engineering scheduling.
"""

from __future__ import annotations

from typing import Any

from arq.connections import RedisSettings
from arq.cron import cron

from football_ai.config import platform_settings
from football_ai.db.session import async_session_factory
from football_ai.logger import get_logger
from football_ai.tasks.jobs import (
    backfill_finished_predictions,
    refresh_upcoming_predictions,
    warm_caches,
)

logger = get_logger(__name__)


async def _startup(ctx: dict[str, Any]) -> None:
    """Hand the session factory to every job via the arq context."""
    ctx["session_factory"] = async_session_factory
    logger.info("arq worker started — session factory wired into ctx")


async def _shutdown(ctx: dict[str, Any]) -> None:
    logger.info("arq worker stopping")


def _redis_settings() -> RedisSettings:
    """Parse ``platform_settings.redis_url`` into an :class:`arq.RedisSettings`."""
    return RedisSettings.from_dsn(platform_settings.redis_url)


class WorkerSettings:
    """arq picks this class up from the CLI invocation."""

    redis_settings = _redis_settings()
    on_startup = _startup
    on_shutdown = _shutdown

    functions = [
        backfill_finished_predictions,
        refresh_upcoming_predictions,
        warm_caches,
    ]

    cron_jobs = [
        cron(backfill_finished_predictions, minute={0, 30}, run_at_startup=False),
        cron(refresh_upcoming_predictions, hour={0, 4, 8, 12, 16, 20}, minute={5}),
        cron(warm_caches, hour={3}, minute={0}),
    ]
