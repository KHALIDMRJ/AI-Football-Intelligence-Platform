"""
Pydantic DTOs for the AI prediction + scouting endpoints (Phase 5).

Kept separate from ``catalog.py`` so the prediction surface (which will
keep growing — confidence tiers, calibration plots, model registry views)
can evolve without churn in the catalogue layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from football_ai.models.prediction import PredictedOutcome


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    match_id: uuid.UUID
    model_version: str

    home_win_prob: float
    draw_prob: float
    away_win_prob: float

    predicted_outcome: PredictedOutcome
    confidence_score: float

    key_factors: list[Any] | None = None

    actual_result: PredictedOutcome | None = None
    was_correct: bool | None = None

    created_at: datetime


class PredictionHistoryPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[PredictionOut]


class LeagueAccuracy(BaseModel):
    league: str
    total: int
    correct: int
    accuracy: float = Field(ge=0.0, le=1.0)


class AccuracyReport(BaseModel):
    """Aggregate hit-rate over predictions whose match has finished."""

    total_scored: int  # predictions with a known actual_result
    total_correct: int
    overall_accuracy: float = Field(ge=0.0, le=1.0)
    by_league: list[LeagueAccuracy]
    model_version: str | None = None  # if filter applied


class ScoutingReportOut(BaseModel):
    player_id: uuid.UUID
    prose: str
    model: str
    generated_at: datetime
    token_count: int | None = None
