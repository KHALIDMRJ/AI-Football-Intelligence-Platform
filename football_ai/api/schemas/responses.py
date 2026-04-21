"""
API response schemas (Pydantic v2).

All endpoints return one of these models.  Using explicit response models:
- guarantees the serialised shape is documented in /docs
- prevents internal DataFrame columns from leaking into responses
- enables FastAPI to validate outgoing data in development mode

Naming convention:
    *Response  — top-level response returned by a router
    *Item      — an element inside a list response
    *Detail    — richer single-resource view
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str
    environment: str


# ── Match ─────────────────────────────────────────────────────────────────────

class MatchItem(BaseModel):
    """Compact match summary for list endpoints."""
    match_id: str
    home_team_id: str
    away_team_id: str
    home_team_name: str
    away_team_name: str
    total_actions: int
    xg_home: float
    xg_away: float
    vaep_home: float
    vaep_away: float


class MatchListResponse(BaseModel):
    count: int
    matches: list[MatchItem]


class TeamVAEPItem(BaseModel):
    team_id: str
    team_name: str
    vaep_total: float
    vaep_offensive: float
    vaep_defensive: float
    xg_total: float


class ActionItem(BaseModel):
    action_id: str
    action_type: str
    player_name: str
    team_name: str
    minute: int
    vaep_value: float
    start_x: float
    start_y: float


class MatchDetailResponse(BaseModel):
    """Full match analysis."""
    match_id: str
    home_team_id: str
    away_team_id: str
    home_team_name: str
    away_team_name: str
    total_actions: int
    xg_home: float
    xg_away: float
    vaep_home: float
    vaep_away: float
    home_formation: str
    away_formation: str
    team_stats: list[TeamVAEPItem] = Field(default_factory=list)
    most_valuable_actions: list[ActionItem] = Field(default_factory=list)


class MatchReportResponse(BaseModel):
    """Full JSON tactical report for a match."""
    match_id: str
    report: dict[str, Any]


# ── Player ────────────────────────────────────────────────────────────────────

class PlayerItem(BaseModel):
    """Compact player entry for list/ranking endpoints."""
    player_id: str
    player_name: str
    team_id: str
    team_name: str
    action_count: int
    vaep_total: float
    vaep_offensive: float
    vaep_defensive: float
    vaep_per_90: float | None = None
    xg_total: float = 0.0
    xt_total: float = 0.0


class PlayerListResponse(BaseModel):
    count: int
    match_id: str | None = None
    players: list[PlayerItem]


class PlayerDetailResponse(BaseModel):
    """Full player profile."""
    player_id: str
    player_name: str
    team_id: str
    team_name: str
    action_count: int
    vaep_total: float
    vaep_offensive: float
    vaep_defensive: float
    vaep_per_90: float | None = None
    xg_total: float = 0.0
    xt_total: float = 0.0
    top_actions: list[ActionItem] = Field(default_factory=list)


class PlayerRankingResponse(BaseModel):
    metric: str
    match_id: str | None = None
    count: int
    players: list[PlayerItem]


# ── Team ──────────────────────────────────────────────────────────────────────

class ZoneWeaknessItem(BaseModel):
    zone_id: int
    zone_x: float
    zone_y: float
    mean_opponent_vaep: float
    action_count: int
    risk_level: str


class TeamDetailResponse(BaseModel):
    team_id: str
    team_name: str
    match_id: str
    action_count: int
    vaep_total: float
    vaep_offensive: float
    vaep_defensive: float
    xg_total: float
    formation: str
    n_critical_zones: int
    weaknesses: list[ZoneWeaknessItem] = Field(default_factory=list)


class TeamListResponse(BaseModel):
    count: int
    match_id: str
    teams: list[TeamDetailResponse]


# ── Tactical ──────────────────────────────────────────────────────────────────

class FormationInfo(BaseModel):
    team_id: str
    team_name: str
    formation: str
    confidence: float


class PressureZoneItem(BaseModel):
    zone_id: int
    zone_x: float
    zone_y: float
    threat: float


class TacticalReportResponse(BaseModel):
    """Full tactical intelligence output for a match."""
    match_id: str
    home_team_name: str
    away_team_name: str
    xg_home: float
    xg_away: float
    vaep_home: float
    vaep_away: float
    formations: list[FormationInfo] = Field(default_factory=list)
    home_weaknesses: list[ZoneWeaknessItem] = Field(default_factory=list)
    away_weaknesses: list[ZoneWeaknessItem] = Field(default_factory=list)
    home_pressure_zones: list[PressureZoneItem] = Field(default_factory=list)
    away_pressure_zones: list[PressureZoneItem] = Field(default_factory=list)
    top_players: list[PlayerItem] = Field(default_factory=list)


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """RFC 7807-style error body."""
    detail: str
    status: int
    type: str = "error"
