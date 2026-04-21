"""
v1 API aggregator.

Collects every v1 endpoint router into a single ``APIRouter`` mounted by
``football_ai.api.main`` under ``/api/v1``. Import order below matches the
Swagger UI tag order (Phase 11 styling).
"""

from __future__ import annotations

from fastapi import APIRouter

from football_ai.api.v1.endpoints import (
    admin,
    analysis,
    auth,
    dashboard,
    live,
    matches,
    players,
    predictions,
    tactical,
    teams,
)

api_router = APIRouter()

# Auth first — every other endpoint depends on it (Phase 2).
api_router.include_router(auth.router)

# Core football data (Phase 4 — already live for players/matches/teams/tactical).
api_router.include_router(players.router)
api_router.include_router(matches.router)
api_router.include_router(teams.router)
api_router.include_router(tactical.router)

# AI layer (Phase 5).
api_router.include_router(predictions.router)
api_router.include_router(analysis.router)

# Realtime live feed (Phase 6 — WebSocket fan-out).
api_router.include_router(live.router)

# Aggregated views (Phase 4) + admin (Phase 7).
api_router.include_router(dashboard.router)
api_router.include_router(admin.router)

__all__ = ["api_router"]
