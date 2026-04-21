"""
Aggregated dashboard intelligence endpoints (v1) — Phase 4.

Planned routes
--------------
GET /dashboard/overview          — cross-match KPIs for the authenticated user
GET /dashboard/leagues/{league}  — league-level roll-up (standings, top performers)

Serves the Streamlit UI (football_ai/dashboard) with pre-joined payloads
so the front-end doesn't fan out into many endpoints for a single view.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Endpoints implemented in Phase 4.
