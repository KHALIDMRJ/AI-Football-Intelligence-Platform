"""
Pydantic v2 domain schemas.

These are the canonical data models used throughout the platform.
Every layer reads and writes using these types, ensuring type safety
and automatic validation at all boundaries.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# ── Raw event (as received from a vendor adapter) ─────────────────────────────

class RawEvent(BaseModel):
    """
    Minimally-validated raw event as it arrives from the ingestion adapter.
    Fields are deliberately broad — normalisation happens in preprocessing.
    """

    model_config = {"extra": "ignore"}

    match_id: str
    index: int
    period: int = Field(ge=1, le=5)
    timestamp: str                     # "HH:MM:SS.mmm"
    minute: int = Field(ge=0)
    second: int = Field(ge=0, le=59)
    event_type: str
    team_id: str | None = None
    team_name: str | None = None
    player_id: str | None = None
    player_name: str | None = None
    player_position: str | None = None
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    duration: float | None = None
    under_pressure: bool = False
    outcome: str | None = None
    body_part: str | None = None
    xg: float | None = None        # if provided by vendor (e.g. StatsBomb)

    @field_validator("start_x", "end_x", mode="before")
    @classmethod
    def clamp_x(cls, v: object) -> float | None:
        if v is None:
            return None
        f = float(v)
        return max(0.0, min(120.0, f))

    @field_validator("start_y", "end_y", mode="before")
    @classmethod
    def clamp_y(cls, v: object) -> float | None:
        if v is None:
            return None
        f = float(v)
        return max(0.0, min(80.0, f))


# ── SPADL action (normalised internal representation) ─────────────────────────

class SPADLAction(BaseModel):
    """
    Canonical SPADL 8-attribute action.
    This is the primary data model for all downstream modules.
    """

    model_config = {"extra": "forbid"}

    match_id: str
    action_id: str                    # stable unique ID
    index: int                        # position in the match event stream
    period: int = Field(ge=1, le=5)
    timestamp: float                  # seconds since kick-off
    minute: int = Field(ge=0)
    second: int = Field(ge=0, le=59)
    team_id: str
    team_name: str
    player_id: str
    player_name: str
    position: str | None = None
    action_type: str                  # from ActionType enum
    body_part: str = "unknown"        # from BodyPart enum
    result: str = "unknown"           # from ActionResult enum
    start_x: float = Field(ge=0.0, le=120.0)
    start_y: float = Field(ge=0.0, le=80.0)
    end_x: float = Field(ge=0.0, le=120.0)
    end_y: float = Field(ge=0.0, le=80.0)
    duration: float = Field(ge=0.0, default=0.0)
    under_pressure: bool = False
    xg: float = Field(ge=0.0, le=1.0, default=0.0)

    # Enriched by preprocessing
    possession_id: int | None = None
    possession_team_id: str | None = None
    chain_id: str | None = None
    chain_index: int | None = None
    zone_id: int | None = None


# ── Game state (computed per action) ─────────────────────────────────────────

class GameState(BaseModel):
    """
    Snapshot of the game immediately after action at ``action_index``.
    Holds the probability estimates that feed into VAEP.
    """

    model_config = {"extra": "forbid"}

    match_id: str
    action_id: str
    action_index: int
    period: int
    timestamp: float
    team_id: str                      # team in possession
    score_home: int = 0
    score_away: int = 0
    p_scores: float = Field(ge=0.0, le=1.0, default=0.0)
    p_concedes: float = Field(ge=0.0, le=1.0, default=0.0)

    @property
    def value(self) -> float:
        """V(Sᵢ) = P_scores − P_concedes"""
        return self.p_scores - self.p_concedes

    @property
    def goal_difference(self) -> int:
        return self.score_home - self.score_away


# ── VAEP result per action ────────────────────────────────────────────────────

class VAEPResult(BaseModel):
    """VAEP value attributed to a single action."""

    model_config = {"extra": "forbid"}

    match_id: str
    action_id: str
    action_index: int
    player_id: str
    player_name: str
    team_id: str
    action_type: str
    vaep_value: float                 # V(Sᵢ) − V(Sᵢ₋₁)
    vaep_offensive: float             # max(0, Δ P_scores)
    vaep_defensive: float             # max(0, −Δ P_concedes)


# ── Player profile ────────────────────────────────────────────────────────────

class PlayerProfile(BaseModel):
    """Aggregated analytics profile for a single player."""

    player_id: str
    player_name: str
    team_id: str
    team_name: str
    position: str | None = None
    minutes_played: float = 0.0
    action_count: int = 0
    vaep_total: float = 0.0
    vaep_offensive: float = 0.0
    vaep_defensive: float = 0.0
    vaep_per_90: float = 0.0
    xg_total: float = 0.0
    xt_total: float = 0.0


# ── Team weakness ────────────────────────────────────────────────────────────

class ZoneWeakness(BaseModel):
    """VAEP-based weakness score for a single pitch zone."""

    team_id: str
    team_name: str
    zone_id: int
    zone_x: float                     # centre x of zone
    zone_y: float                     # centre y of zone
    mean_vaep_against: float          # mean opponent VAEP in this zone
    action_count: int
    risk_level: str                   # "low" | "medium" | "high" | "critical"


# ── Match analysis summary ────────────────────────────────────────────────────

class MatchSummary(BaseModel):
    """Top-level summary object for a processed match."""

    match_id: str
    home_team_id: str
    away_team_id: str
    home_team_name: str
    away_team_name: str
    total_actions: int
    xg_home: float
    xg_away: float
    xt_home: float
    xt_away: float
    vaep_home: float
    vaep_away: float
    top_players: list[PlayerProfile] = Field(default_factory=list)
    weaknesses: list[ZoneWeakness] = Field(default_factory=list)


# ── API request / response schemas ───────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    """Request body for POST /pipeline/run."""

    match_id: str | None = None
    input_path: str | None = None
    skip_training: bool = False
    force_reprocess: bool = False


class PipelineRunResponse(BaseModel):
    """Response for POST /pipeline/run."""

    status: str                       # "started" | "complete" | "error"
    match_id: str | None = None
    message: str
    actions_processed: int = 0
