"""
AI analysis endpoints (v1) — Phase 5.

Planned routes
--------------
POST /analysis/player/{player_id}              — Claude-generated scouting report
POST /analysis/match/{match_id}/tactical       — AI tactical breakdown of a finished match

Results are cached 24h (Redis) because scouting reports are expensive to
generate and the underlying stats don't change between matches.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/analysis", tags=["analysis"])

# Endpoints implemented in Phase 5.
