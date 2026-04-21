"""
Pydantic DTOs for DB-backed catalogue endpoints (Phase 4).

Kept separate from ``responses.py`` (which holds pipeline/VAEP analytics
shapes) so the two data surfaces can evolve independently. All models
use ``from_attributes=True`` so we can hand a SA row straight to
``model_validate`` without hand-mapping fields.

PagedResponse is a light generic envelope for list endpoints — the
existing pipeline routes return custom envelopes, but DB-backed
endpoints will all grow the same pagination tail, so factor it once.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from football_ai.models.match import EventType, MatchStatus
from football_ai.models.player import PreferredFoot

T = TypeVar("T")


class PagedResponse(BaseModel, Generic[T]):
    """Generic paginated envelope: items + total + page cursor."""

    total: int
    limit: int
    offset: int
    items: list[T]


# ── Teams ────────────────────────────────────────────────────────────────────

class TeamBrief(BaseModel):
    """Compact team summary embedded in match/player responses."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    league: str | None = None
    country: str | None = None
    logo_url: str | None = None


class TeamRegistryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    country: str | None = None
    league: str | None = None
    founded_year: int | None = None
    formation: str | None = None
    playing_style: str | None = None
    logo_url: str | None = None
    api_football_id: int | None = None
    created_at: datetime


# ── Players ──────────────────────────────────────────────────────────────────

class PlayerCatalogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    age: int | None = None
    nationality: str | None = None
    position: str | None = None
    preferred_foot: PreferredFoot | None = None
    jersey_number: int | None = None
    market_value_eur: int | None = None
    api_football_id: int | None = None
    current_team: TeamBrief | None = None


class SquadMemberOut(BaseModel):
    """Roster row — jersey-first ordering, trimmed vs PlayerCatalogOut."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    position: str | None = None
    jersey_number: int | None = None
    age: int | None = None
    nationality: str | None = None


class PlayerStatsOut(BaseModel):
    """One row from ``player_stats`` — season aggregate if match_id is null."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    player_id: uuid.UUID
    match_id: uuid.UUID | None = None
    season: str

    goals: int
    assists: int
    minutes_played: int
    pass_accuracy: float | None = None
    shots_on_target: int
    dribbles_completed: int
    tackles_won: int

    xg: float | None = None
    xa: float | None = None
    rating: float | None = None


# ── Matches ──────────────────────────────────────────────────────────────────

class MatchFixtureOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    league: str
    season: str
    match_date: datetime
    status: MatchStatus

    home_team: TeamBrief
    away_team: TeamBrief

    home_score: int | None = None
    away_score: int | None = None
    venue: str | None = None
    referee: str | None = None
    api_football_id: int | None = None


class MatchEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    match_id: uuid.UUID
    minute: int
    event_type: EventType
    player_id: uuid.UUID | None = None
    team_id: uuid.UUID | None = None
    extra_info: dict[str, Any] | None = None
