"""
FastAPI application — Football AI Intelligence Platform.

Run with:
    uvicorn football_ai.api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs      (custom Swagger UI — tactical war-room theme)
    http://localhost:8000/redoc     (ReDoc)

Architecture
------------
    main.py                 — app factory + middleware + lifespan + exception handlers
    dependencies.py         — shared FastAPI deps (get_db, get_current_user, get_redis)
    docs.py                 — custom Swagger UI (dark/light toggle, live search)
    endpoints/              — version-agnostic infra endpoints (health probes)
    v1/router.py            — v1 aggregator mounted at /api/v1
    v1/endpoints/           — versioned HTTP endpoints
    schemas/                — Pydantic response DTOs
    services/               — response assemblers on top of the analytics core
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from football_ai.api.docs import custom_swagger_ui_html
from football_ai.api.endpoints import health as infra_health
from football_ai.api.v1.router import api_router as v1_router
from football_ai.cache import close_cache, init_cache
from football_ai.config import settings
from football_ai.core.exceptions import FootballAPIError
from football_ai.logger import get_logger, setup_logging
from football_ai.observability.middleware import RequestContextMiddleware
from football_ai.security import (
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
    resolve_cors_origins,
)

setup_logging()
logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(
        "Starting %s v%s [%s]",
        settings.project_name,
        settings.version,
        settings.environment,
    )
    # Cache backend is picked once at startup (Redis if reachable, else in-memory).
    await init_cache()
    yield
    await close_cache()
    logger.info("Shutting down %s", settings.project_name)


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description=(
            "AI Football Intelligence Platform — advanced tactical analytics "
            "powered by VAEP, xG, xT, and ML-driven player/team insights.\n\n"
            "**Prerequisites:** run `scripts/run_pipeline.py` to process match data "
            "before calling analysis endpoints."
        ),
        docs_url=None,
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Custom Swagger UI ─────────────────────────────────────────────────────
    @app.get("/docs", include_in_schema=False)
    async def custom_docs():
        return custom_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title,
            version=app.version,
        )

    # ── Root welcome ──────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "message": f"Welcome to the {settings.project_name} API",
                "version": settings.version,
                "docs": "/docs",
                "redoc": "/redoc",
                "health": "/health",
                "api": "/api/v1",
            }
        )

    # ── Middleware ────────────────────────────────────────────────────────────
    # Order matters: added LAST runs FIRST on inbound requests.
    # Execution order per request (outermost first):
    #   RequestContextMiddleware   → stamps X-Request-ID, times latency
    #   SecurityHeadersMiddleware  → adds HSTS/nosniff/frame-deny on response
    #   BodySizeLimitMiddleware    → 413 before the app reads the body
    #   CORSMiddleware             → preflight + Origin handling (innermost)
    origins = resolve_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "Retry-After",
        ],
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=5 * 1024 * 1024)  # 5 MiB
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(FootballAPIError)
    async def domain_exception_handler(
        request: Request, exc: FootballAPIError
    ) -> JSONResponse:
        logger.warning(
            "Domain error on %s %s: [%s] %s",
            request.method, request.url.path, exc.error_type, exc.detail,
        )
        # RateLimitError carries Retry-After + X-RateLimit-* so the client
        # can back off intelligently; vanilla domain errors don't.
        extra_headers = getattr(exc, "headers", None) or {}
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "status": exc.status_code,
                "type": exc.error_type,
            },
            headers=extra_headers,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method, request.url.path, exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error.",
                "status": 500,
                "type": "internal_error",
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    # Infra health probes stay at top level (stable across API version bumps).
    app.include_router(infra_health.router)

    # All API traffic is versioned.
    app.include_router(v1_router, prefix="/api/v1")

    return app


app = create_app()
