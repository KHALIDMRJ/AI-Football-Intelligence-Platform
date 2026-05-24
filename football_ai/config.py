"""
Configuration loaders.

Two distinct settings surfaces — one per concern:

* ``settings``          — analytics core (pitch, VAEP, model hyperparams).
                          Source: ``configs/settings.yaml`` (committed to git).
* ``platform_settings`` — infrastructure + secrets (DB URL, Redis URL, JWT
                          secret, external API keys).
                          Source: environment variables + ``.env`` file.

Why two: pitch geometry and VAEP k_actions are stable, reviewable, domain-
tuned values that belong in version control. DB credentials and API keys
belong in env vars (12-factor) and must never be committed.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Sub-models ────────────────────────────────────────────────────────────────

class PathsConfig(BaseModel):
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    data_features: str = "data/features"
    models: str = "models"
    logs: str = "logs"


class PitchConfig(BaseModel):
    length: float = 120.0
    width: float = 80.0
    goal_x: float = 120.0
    goal_y: float = 40.0
    goal_post_top: float = 44.0
    goal_post_bottom: float = 36.0
    zones_x: int = 18
    zones_y: int = 12


class PossessionConfig(BaseModel):
    max_gap_seconds: float = 5.0
    min_chain_length: int = 2
    max_chain_length: int = 5


class VAEPConfig(BaseModel):
    k_actions: int = 10
    min_minutes_for_per90: int = 20


class XGModelConfig(BaseModel):
    type: str = "logistic_regression"
    max_iter: int = 1000
    C: float = 1.0


class XGBoostModelConfig(BaseModel):
    type: str = "xgboost"
    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 20


class CatBoostModelConfig(BaseModel):
    type: str = "catboost"
    iterations: int = 300
    depth: int = 5
    learning_rate: float = 0.05
    loss_function: str = "Logloss"
    eval_metric: str = "AUC"
    early_stopping_rounds: int = 20


class XTModelConfig(BaseModel):
    type: str = "markov"
    zones_x: int = 18
    zones_y: int = 12
    smoothing: float = 1.0


class ModelsConfig(BaseModel):
    xg: XGModelConfig = Field(default_factory=XGModelConfig)
    p_scores: XGBoostModelConfig = Field(default_factory=XGBoostModelConfig)
    p_concedes: CatBoostModelConfig = Field(default_factory=CatBoostModelConfig)
    xt: XTModelConfig = Field(default_factory=XTModelConfig)


class TrainingConfig(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 1


class DashboardConfig(BaseModel):
    port: int = 8501
    theme: str = "light"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


# ── Master settings ────────────────────────────────────────────────────────────

class Settings(BaseModel):
    """Master configuration object loaded from configs/settings.yaml."""

    project_name: str = "Football AI Platform"
    version: str = "0.1.0"
    environment: str = "development"

    paths: PathsConfig = Field(default_factory=PathsConfig)
    pitch: PitchConfig = Field(default_factory=PitchConfig)
    possession: PossessionConfig = Field(default_factory=PossessionConfig)
    vaep: VAEPConfig = Field(default_factory=VAEPConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def root_dir(self) -> Path:
        """Absolute path to the project root (two levels above this file)."""
        return Path(__file__).parent.parent.resolve()

    def abs(self, relative_path: str) -> Path:
        """Convert a config-relative path to an absolute Path."""
        return self.root_dir / relative_path


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _flatten_project(raw: dict[str, Any]) -> dict[str, Any]:
    """
    The YAML has a 'project' block with name/version/environment.
    Flatten those into the top level for Settings.
    """
    result = dict(raw)
    project_block = result.pop("project", {})
    result["project_name"] = project_block.get("name", "Football AI Platform")
    result["version"] = project_block.get("version", "0.1.0")
    result["environment"] = project_block.get("environment", "development")
    return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load and cache settings from configs/settings.yaml.
    The config file path can be overridden via the FOOTBALL_AI_CONFIG env var.
    """
    config_env = os.getenv("FOOTBALL_AI_CONFIG")
    if config_env:
        config_path = Path(config_env)
    else:
        # Walk up from this file to find the project root
        config_path = Path(__file__).parent.parent / "configs" / "settings.yaml"

    if config_path.exists():
        raw = _load_yaml(config_path)
        flat = _flatten_project(raw)
        return Settings(**flat)

    # Fallback: use all defaults
    return Settings()


# Module-level singleton — import this everywhere
settings: Settings = get_settings()


# ── Platform settings (env-driven) ─────────────────────────────────────────────
#
# These are runtime/infra concerns: database URL, Redis URL, JWT secret,
# external API keys. Loaded from environment variables + a .env file.

class PlatformSettings(BaseSettings):
    """
    Infrastructure + secrets configuration for the Football AI platform.

    Reads from process env and ``.env`` (if present).  All fields are
    documented in ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Environment ────────────────────────────────────────────────────────────
    environment: Literal["development", "staging", "production", "test"] = "development"
    log_level: str = "INFO"

    # ── Database ───────────────────────────────────────────────────────────────
    # Default is SQLite via aiosqlite so the platform runs zero-infra for dev.
    # Production (docker-compose, Phase 10) overrides with postgres+asyncpg.
    database_url: str = "sqlite+aiosqlite:///./football_ai.db"
    database_echo: bool = False  # SQL logging

    # ── Redis (Phase 6 — caching + arq queue) ──────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── JWT auth (Phase 3) ─────────────────────────────────────────────────────
    # NEVER commit a real secret. Generate one with:
    #   python -c "import secrets; print(secrets.token_urlsafe(64))"
    secret_key: str = "dev-only-change-me-" + "x" * 40
    jwt_algorithm: str = "HS256"
    access_token_expires_minutes: int = 30
    refresh_token_expires_days: int = 14

    # ── External APIs ─────────────────────────────────────────────────────────
    # API-Football free tier: 100 calls/day. Set to empty string to disable
    # external sync (Phase 7 admin endpoints will return 503).
    api_football_key: str = ""
    api_football_base_url: str = "https://v3.football.api-sports.io"

    # Anthropic (Phase 5 — scouting reports + tactical commentary).
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # Gemini is used only server-side as a chat/comparison proxy. Never expose
    # this through NEXT_PUBLIC_* because browser keys are trivially copied.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # ── CORS / hosts ───────────────────────────────────────────────────────────
    allowed_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ── Rate limiting (Phase 3) ────────────────────────────────────────────────
    # Per-day request quotas by role. Set to -1 for unlimited.
    quota_free_user_per_day: int = 100
    quota_pro_analyst_per_day: int = -1
    quota_club_scout_per_day: int = -1

    # ── Helpers ────────────────────────────────────────────────────────────────
    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def sync_database_url(self) -> str:
        """Synchronous DB URL for Alembic (autogenerate introspection)."""
        url = self.database_url
        return (
            url.replace("+asyncpg", "")
               .replace("+aiosqlite", "")
        )


@lru_cache(maxsize=1)
def get_platform_settings() -> PlatformSettings:
    return PlatformSettings()


platform_settings: PlatformSettings = get_platform_settings()
