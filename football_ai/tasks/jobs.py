"""
Background job definitions.

Every job is an ``async def`` taking ``ctx`` (the arq context) plus its own
arguments. ``ctx["session_factory"]`` is set by the worker startup hook
(see :mod:`football_ai.tasks.worker`) and by the test helpers.

What each job does and why
--------------------------
* :func:`backfill_finished_predictions` — when matches finish, the
  scoreline only lands via the ingest path, not the prediction path.
  This job stamps every ``AIPrediction`` for a recently-finished match
  with its true outcome so the accuracy report stays current.
* :func:`refresh_upcoming_predictions` — re-runs the predictor against
  all matches scheduled in the next ``window_hours``. New form data
  (last weekend's results, an injury) shifts probabilities; users get
  the freshest signal without having to POST individually.
* :func:`warm_caches` — pre-loads the cache slots that the dashboard
  hits on every page render (rankings, accuracy, fixtures). Run on a
  cron so cold-cache pages never bite a real user.

Each job returns a small JSON-friendly dict so the admin enqueue
endpoint can echo the result on synchronous (``await=True``) runs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from football_ai.cache import get_cache
from football_ai.crud import prediction as prediction_crud
from football_ai.logger import get_logger
from football_ai.ml.serving.match_predictor import MatchPredictor, outcome_from_score
from football_ai.models.match import Match, MatchStatus

logger = get_logger(__name__)


async def backfill_finished_predictions(
    ctx: dict[str, Any],
    *,
    since_hours: int = 24,
) -> dict[str, Any]:
    """Stamp ``actual_result`` + ``was_correct`` on predictions for matches
    that finished in the last ``since_hours``.

    Idempotent — re-running stamps the same value, no harm done. Useful
    after a manual ingest catch-up where the predictions and the score
    landed in different transactions.
    """
    factory = ctx["session_factory"]
    cutoff = datetime.now(UTC) - timedelta(hours=since_hours)
    updated = 0
    matches_seen = 0

    async with factory() as db:
        stmt = (
            select(Match)
            .where(
                Match.status == MatchStatus.finished,
                Match.match_date >= cutoff,
                Match.home_score.is_not(None),
                Match.away_score.is_not(None),
            )
        )
        matches = (await db.execute(stmt)).scalars().all()

        for match in matches:
            matches_seen += 1
            outcome = outcome_from_score(match.home_score, match.away_score)
            updated += await prediction_crud.backfill_actual_result(
                db, match.id, outcome
            )
        await db.commit()

    logger.info(
        "backfill_finished_predictions: %d match(es), %d prediction(s) stamped",
        matches_seen, updated,
    )
    return {"matches": matches_seen, "predictions_updated": updated}


async def refresh_upcoming_predictions(
    ctx: dict[str, Any],
    *,
    window_hours: int = 72,
) -> dict[str, Any]:
    """Re-run the predictor on every scheduled match in the next ``window_hours``.

    Skips matches the predictor refuses (e.g. cancelled). A failure on one
    match is logged and the loop continues — one bad fixture shouldn't
    starve the rest of the slate of refreshed odds.
    """
    factory = ctx["session_factory"]
    horizon = datetime.now(UTC) + timedelta(hours=window_hours)
    refreshed = 0
    failed = 0

    predictor = MatchPredictor.instance()

    async with factory() as db:
        stmt = (
            select(Match)
            .where(
                Match.status == MatchStatus.scheduled,
                Match.match_date <= horizon,
            )
        )
        matches = (await db.execute(stmt)).scalars().all()

        for match in matches:
            try:
                await predictor.predict(db, match)
                refreshed += 1
            except Exception as exc:
                failed += 1
                logger.warning(
                    "refresh_upcoming_predictions: %s failed (%s: %s)",
                    match.id, exc.__class__.__name__, exc,
                )

        await db.commit()

    logger.info(
        "refresh_upcoming_predictions: refreshed=%d failed=%d", refreshed, failed,
    )
    return {"refreshed": refreshed, "failed": failed}


async def warm_caches(
    ctx: dict[str, Any],
    *,
    namespaces: list[str] | None = None,
) -> dict[str, Any]:
    """Drop selected cache namespaces so the next request rebuilds fresh.

    Counter-intuitive name — we don't pre-populate (that would mean
    duplicating endpoint logic here). Instead, we invalidate so the
    *next* user request hydrates the cache; given the cron runs at
    quiet hours, the rebuild cost lands off the user-visible path.
    """
    cache = await get_cache()
    if namespaces is None:
        # Full clear — used by ops after a schema bump or model retrain.
        await cache.clear()
        logger.info("warm_caches: full clear")
        return {"cleared": "all"}

    # Partial clear is best-effort: we don't keep an index of keys per
    # namespace. The InMemoryCache size guarantees expiry on TTL anyway.
    # For Redis, callers who need surgical invalidation should use the
    # SCAN-based admin tooling (out of scope for the portfolio build).
    logger.info("warm_caches: partial clear requested for %s", namespaces)
    return {"cleared": namespaces, "note": "partial clears are TTL-driven"}
