"""
Admin endpoints (v1).

Currently exposes the Phase-6 background-task surface; the Phase-7
external-sync endpoints will land here too.

Routes
------
POST /admin/tasks/{name}     — run or enqueue a background job
GET  /admin/tasks            — list available job names

Why two run modes
-----------------
Jobs ship as plain async functions (see :mod:`football_ai.tasks.jobs`)
so the admin endpoint can call them in two ways depending on the
``run_inline`` query flag:

* ``run_inline=true`` (default in dev / portfolio demo): the function
  is awaited directly and the result is returned in the response.
  Operator gets immediate feedback. Costs the request thread.
* ``run_inline=false``: the function is enqueued onto the arq Redis
  queue; response is the job id. Required when tasks are slow enough
  to time-out a request.

Enqueue silently falls back to inline execution if Redis is not
configured — matches the cache layer's degrade-don't-fail philosophy.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.api.dependencies import get_db, require_role
from football_ai.config import platform_settings
from football_ai.db.session import async_session_factory
from football_ai.external import get_api_football_client, get_quota_tracker
from football_ai.external.api_football import APIFootballClient
from football_ai.external.quota import QuotaTracker
from football_ai.logger import get_logger
from football_ai.models.user import UserRole
from football_ai.services.sync import (
    sync_fixture_events,
    sync_league_fixtures,
    sync_team_squad,
)
from football_ai.tasks import jobs as job_module

logger = get_logger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_role(UserRole.admin))],
)


# Whitelist — never resolve a name to a function via getattr without one.
# Listed explicitly so adding a new job is a deliberate two-line change.
_AVAILABLE_JOBS = {
    "backfill_finished_predictions": job_module.backfill_finished_predictions,
    "refresh_upcoming_predictions": job_module.refresh_upcoming_predictions,
    "warm_caches": job_module.warm_caches,
}


@router.get(
    "/tasks",
    summary="List background tasks (admin)",
)
async def list_tasks() -> dict[str, list[str]]:
    return {"available": sorted(_AVAILABLE_JOBS.keys())}


@router.post(
    "/tasks/{name}",
    summary="Run or enqueue a background task (admin)",
    description=(
        "Executes the named job inline by default (returns the result). "
        "Pass ``run_inline=false`` to push onto the arq queue instead — "
        "if Redis isn't configured the call still runs inline (with a note)."
    ),
    responses={
        404: {"description": "Unknown task name."},
        503: {"description": "Async enqueue requested but Redis unavailable."},
    },
)
async def run_task(
    name: str,
    run_inline: bool = Query(True, description="If false, enqueue on arq."),
    payload: dict[str, Any] | None = Body(default=None),
) -> dict[str, Any]:
    job = _AVAILABLE_JOBS.get(name)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown task '{name}'. See GET /admin/tasks for the list.",
        )

    job_kwargs = payload or {}

    if run_inline:
        ctx = {"session_factory": async_session_factory}
        result = await job(ctx, **job_kwargs)
        return {"task": name, "mode": "inline", "result": result}

    # Async enqueue path — needs Redis.
    return await _enqueue(name, job_kwargs)


# ── Async enqueue helpers ────────────────────────────────────────────────────

async def _enqueue(name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Push a job onto the arq queue. Falls back to inline if Redis is dead."""
    try:
        from arq import create_pool
        from arq.connections import RedisSettings

        settings = RedisSettings.from_dsn(platform_settings.redis_url)
        pool = await create_pool(settings)
        try:
            job = await pool.enqueue_job(name, **kwargs)
            return {
                "task": name,
                "mode": "queued",
                "job_id": job.job_id if job else None,
            }
        finally:
            await pool.aclose()
    except Exception as exc:
        logger.warning(
            "Async enqueue failed (%s) — running inline as a fallback.",
            exc.__class__.__name__,
        )
        ctx = {"session_factory": async_session_factory}
        job = _AVAILABLE_JOBS[name]
        result = await job(ctx, **kwargs)
        return {
            "task": name,
            "mode": "inline_fallback",
            "result": result,
            "note": f"queue unavailable ({exc.__class__.__name__})",
        }


# ── External sync (Phase 7) ──────────────────────────────────────────────────

@router.post(
    "/sync/fixtures",
    summary="Sync fixtures for a league × season from API-Football",
    responses={
        429: {"description": "Daily API quota exhausted."},
        502: {"description": "Upstream provider error."},
        503: {"description": "API-Football integration not configured."},
    },
)
async def sync_fixtures(
    db: Annotated[AsyncSession, Depends(get_db)],
    client: Annotated[APIFootballClient, Depends(get_api_football_client)],
    league: int = Query(..., description="API-Football league id (e.g. 39 = EPL)."),
    season: int = Query(..., description="Season starting year (e.g. 2024)."),
) -> dict[str, Any]:
    result = await sync_league_fixtures(db, client, league=league, season=season)
    return result.to_dict()


@router.post(
    "/sync/squad",
    summary="Sync a team's squad from API-Football",
    responses={
        404: {"description": "Team not found in our DB."},
        409: {"description": "Team has no api_football_id — sync fixtures first."},
        429: {"description": "Daily API quota exhausted."},
        502: {"description": "Upstream provider error."},
        503: {"description": "API-Football integration not configured."},
    },
)
async def sync_squad(
    team_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    client: Annotated[APIFootballClient, Depends(get_api_football_client)],
) -> dict[str, Any]:
    try:
        result = await sync_team_squad(db, client, team_id=team_id)
    except ValueError as exc:
        # Distinguish "missing team" (404) from "missing provider id" (409).
        msg = str(exc)
        if "not found" in msg:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=msg) from exc
        raise HTTPException(status.HTTP_409_CONFLICT, detail=msg) from exc
    return result.to_dict()


@router.post(
    "/sync/events",
    summary="Sync match events for a single fixture from API-Football",
    responses={
        404: {"description": "Match not found in our DB."},
        409: {"description": "Match has no api_football_id — sync fixtures first."},
        429: {"description": "Daily API quota exhausted."},
        502: {"description": "Upstream provider error."},
        503: {"description": "API-Football integration not configured."},
    },
)
async def sync_events(
    match_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    client: Annotated[APIFootballClient, Depends(get_api_football_client)],
) -> dict[str, Any]:
    try:
        result = await sync_fixture_events(db, client, match_id=match_id)
    except ValueError as exc:
        msg = str(exc)
        if "not found" in msg:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=msg) from exc
        raise HTTPException(status.HTTP_409_CONFLICT, detail=msg) from exc
    return result.to_dict()


@router.get(
    "/sync/status",
    summary="External provider quota + configuration snapshot",
    description=(
        "One stop for ops to see whether external integrations are wired up "
        "and how much quota is left for the day."
    ),
)
async def sync_status(
    quota: Annotated[QuotaTracker, Depends(get_quota_tracker)],
    client: Annotated[APIFootballClient, Depends(get_api_football_client)],
) -> dict[str, Any]:
    return {
        "api_football": {
            "configured": client.is_configured(),
            "base_url": platform_settings.api_football_base_url,
            "quota": await quota.snapshot(),
        },
    }
