"""
LLM-backed scouting reports.

Wraps the Anthropic SDK with a graceful-degrade contract: when
``ANTHROPIC_API_KEY`` is unset (the default in dev / CI / portfolio
demo), the service raises ``ExternalAPIError`` (HTTP 503) so callers
get a clear "feature not configured" response rather than a cryptic
500 from the SDK.

Why a service, not inline in the endpoint
-----------------------------------------
Two reasons. (1) Scouting reports will eventually be triggered both
synchronously (from the player profile page) and asynchronously (from a
nightly arq job, Phase 6); the service layer keeps the prompt + parsing
logic in one place. (2) The Anthropic call is the most expensive thing
the platform does — keeping it isolated lets us unit-test the prompt
without spending real tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from football_ai.config import platform_settings
from football_ai.core.exceptions import ExternalAPIError
from football_ai.logger import get_logger
from football_ai.models.player import Player, PlayerStats

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior football scout writing for a professional club's "
    "recruitment department. Your reports are concise, evidence-led, and "
    "stay in the third person. Avoid hype; flag weaknesses honestly. "
    "Cite specific stats from the data block — never invent numbers."
)


@dataclass
class ScoutingReport:
    player_id: str
    prose: str
    model: str
    generated_at: datetime
    token_count: int | None = None


def _build_user_message(player: Player, stats: list[PlayerStats]) -> str:
    """Assemble a structured prompt the model can ground its prose in."""
    season_lines = []
    match_lines = []
    for s in stats:
        line = (
            f"  goals={s.goals} assists={s.assists} mins={s.minutes_played} "
            f"xG={s.xg} xA={s.xa} rating={s.rating}"
        )
        if s.match_id is None:
            season_lines.append(f"- season={s.season}: {line}")
        else:
            match_lines.append(f"- match={s.match_id} season={s.season}: {line}")

    season_block = "\n".join(season_lines) or "  (no season aggregates available)"
    match_block = "\n".join(match_lines[:10]) or "  (no per-match lines available)"

    return (
        f"Generate a scouting report (200–300 words) for the following player.\n\n"
        f"Player: {player.name}\n"
        f"Position: {player.position or 'unknown'}\n"
        f"Age: {player.age if player.age is not None else 'unknown'}\n"
        f"Nationality: {player.nationality or 'unknown'}\n"
        f"Preferred foot: {player.preferred_foot.value if player.preferred_foot else 'unknown'}\n"
        f"\nSEASON AGGREGATES:\n{season_block}\n"
        f"\nRECENT MATCH LINES:\n{match_block}\n"
        f"\nReport structure: opening summary (2 sentences), strengths "
        f"(bullet list, 3 items max), weaknesses (bullet list, 2 items max), "
        f"closing recommendation (1 sentence)."
    )


class ScoutingService:
    """Sync wrapper. The endpoint runs it in a threadpool via FastAPI."""

    def __init__(self, client: Any | None = None) -> None:
        # Allow tests to inject a stub client without touching the env.
        self._client = client
        self._model = platform_settings.anthropic_model

    def is_configured(self) -> bool:
        return bool(platform_settings.anthropic_api_key) or self._client is not None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not platform_settings.anthropic_api_key:
            raise ExternalAPIError(
                "Scouting reports require ANTHROPIC_API_KEY to be configured.",
                status_code=503,
            )
        try:
            import anthropic  # local import — heavy SDK
        except ImportError as exc:  # pragma: no cover
            raise ExternalAPIError(
                "anthropic SDK not installed.", status_code=503
            ) from exc
        self._client = anthropic.Anthropic(
            api_key=platform_settings.anthropic_api_key
        )
        return self._client

    def generate(self, player: Player, stats: list[PlayerStats]) -> ScoutingReport:
        client = self._ensure_client()
        user_msg = _build_user_message(player, stats)
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=600,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as exc:  # network, rate-limit, auth, etc.
            logger.warning("Anthropic call failed: %s", exc)
            raise ExternalAPIError(
                f"Scouting LLM call failed: {exc.__class__.__name__}",
                status_code=502,
            ) from exc

        prose = _extract_text(response)
        token_count = _safe_token_count(response)
        return ScoutingReport(
            player_id=str(player.id),
            prose=prose,
            model=self._model,
            generated_at=datetime.now(UTC),
            token_count=token_count,
        )


def _extract_text(response: Any) -> str:
    """Pull the joined text from Anthropic's content-block response."""
    content = getattr(response, "content", None)
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _safe_token_count(response: Any) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    return int(inp + out) or None
