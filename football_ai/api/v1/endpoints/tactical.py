"""
Tactical endpoints (v1) — VAEP/xT/xG tactical intelligence (the analytical core).

Routes
------
GET /tactical/{match_id}   — full tactical intelligence response
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from football_ai.api.schemas.responses import TacticalReportResponse
from football_ai.api.services.tactical_service import TacticalService

router = APIRouter(prefix="/tactical", tags=["tactical"])


def _tactical_service() -> TacticalService:
    return TacticalService()


@router.get(
    "/{match_id}",
    response_model=TacticalReportResponse,
    summary="Full tactical intelligence report",
    description=(
        "Returns the complete tactical intelligence for a match: xG, VAEP, "
        "formations for both teams, tactical weaknesses (up to 10 zones per team), "
        "opponent pressure zones, and the top 10 players by VAEP."
    ),
    responses={
        404: {"description": "Match not found — run the full pipeline first."},
    },
)
async def get_tactical_report(
    match_id: str,
    service: TacticalService = Depends(_tactical_service),
) -> TacticalReportResponse:
    try:
        return service.get_tactical_report(match_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
