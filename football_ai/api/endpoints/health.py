"""
Health router — infrastructure probes + Prometheus scrape.

Routes
------
GET /health         — liveness (always 200 when the process is running)
GET /health/ready   — readiness: pings DB + cache; 503 if any dependency is down
GET /health/live    — alias for /health; explicit k8s-style liveness path
GET /metrics        — Prometheus exposition (text format)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from football_ai.api.schemas.responses import HealthResponse
from football_ai.cache.factory import get_cache
from football_ai.config import settings
from football_ai.db.session import async_session_factory
from football_ai.logger import get_logger
from football_ai.observability.metrics import metrics_endpoint

logger = get_logger(__name__)

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Returns 200 when the service is running. "
        "Used by k8s / Railway / Fly liveness probes."
    ),
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.project_name,
        version=settings.version,
        environment=settings.environment,
    )


@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Liveness probe (alias)",
    description="Explicit k8s-style alias for /health.",
    include_in_schema=False,
)
async def health_live() -> HealthResponse:
    return await health()


@router.get(
    "/health/ready",
    summary="Readiness probe (DB + cache)",
    description=(
        "Reports whether the service can do useful work. Returns 503 when "
        "any dependency (DB, cache) is unreachable, so k8s / Railway can "
        "park the pod out of rotation."
    ),
)
async def health_ready(response: Response) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    checks["database"] = await _check_db()
    checks["cache"] = await _check_cache()

    ok = all(c["ok"] for c in checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {"status": "ok" if ok else "degraded", "checks": checks}


@router.get(
    "/metrics",
    summary="Prometheus metrics (text exposition)",
    include_in_schema=False,
)
async def metrics() -> Response:
    return await metrics_endpoint()


# ── Readiness helpers ────────────────────────────────────────────────────────

async def _check_db() -> dict[str, Any]:
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as exc:
        logger.warning("Readiness: DB check failed: %s", exc)
        return {"ok": False, "error": exc.__class__.__name__}


async def _check_cache() -> dict[str, Any]:
    try:
        cache = await get_cache()
        await cache.set("__ready_probe__", b"ok", ttl=5)
        value = await cache.get("__ready_probe__")
        backend = cache.__class__.__name__
        return {"ok": value == b"ok", "backend": backend}
    except Exception as exc:
        logger.warning("Readiness: cache check failed: %s", exc)
        return {"ok": False, "error": exc.__class__.__name__}
