"""
Domain exceptions for the Football AI Platform.

Why a hierarchy: the platform crosses three failure domains that must be
distinguishable at the HTTP layer so the right status code and user-facing
message can be chosen without leaking internals.

    FootballAPIError          — base; anything the platform itself raises
    ├── DataNotFoundError     — requested match/player/team is not in the store
    ├── PipelineNotRunError   — tactical report missing because Phase 7 wasn't run
    ├── PredictionError       — match-outcome model failed (feature gap, model miss)
    ├── ExternalAPIError      — API-Football / Anthropic returned non-2xx or timed out
    └── AuthError             — JWT invalid, role insufficient, rate limit hit
"""

from __future__ import annotations


class FootballAPIError(Exception):
    """Base for every domain error raised by this platform."""

    status_code: int = 500
    error_type: str = "internal_error"

    def __init__(self, detail: str, *, status_code: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code


class DataNotFoundError(FootballAPIError):
    """Match / player / team not present in the processed store."""

    status_code = 404
    error_type = "data_not_found"


class PipelineNotRunError(FootballAPIError):
    """Analytical artefact missing because the upstream pipeline has not been executed."""

    status_code = 409
    error_type = "pipeline_not_run"


class PredictionError(FootballAPIError):
    """Match-outcome prediction could not be produced (feature gap, model miss, etc.)."""

    status_code = 422
    error_type = "prediction_failed"


class ExternalAPIError(FootballAPIError):
    """External data provider (API-Football, Anthropic) failed or rate-limited us."""

    status_code = 502
    error_type = "external_api_error"


class AuthError(FootballAPIError):
    """Authentication or authorization failure (JWT, role, quota)."""

    status_code = 401
    error_type = "auth_error"


class RateLimitError(FootballAPIError):
    """Caller exceeded their token-bucket budget for this endpoint.

    Carries the headers dict the HTTP layer should echo back
    (``Retry-After`` + ``X-RateLimit-*``) so clients can back off
    without guesswork.
    """

    status_code = 429
    error_type = "rate_limited"

    def __init__(
        self,
        detail: str,
        *,
        retry_after: float,
        limit: int,
        remaining: int = 0,
    ) -> None:
        super().__init__(detail, status_code=429)
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Retry-After": str(max(1, int(self.retry_after) + 1)),
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
        }
