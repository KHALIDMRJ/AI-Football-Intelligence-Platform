"""
API-Football async client.

Endpoints we exercise
---------------------
* ``GET /fixtures?league=&season=``           — league fixtures for a season
* ``GET /players/squads?team=``               — squad roster
* ``GET /fixtures/events?fixture=``           — minute-by-minute events

The client is intentionally narrow — only the routes the platform syncs.
A wider wrapper would be dead code; the rest of the API surface lives
behind ``raw_get`` for ad-hoc admin calls.

Failure modes
-------------
* No API key set → :class:`ExternalAPIError` 503 ("not configured"). Match the
  scouting-service degradation path so the operator gets the same signal
  for every "you forgot a key" failure.
* Quota exhausted → :class:`ExternalAPIError` 429.
* Upstream non-2xx → :class:`ExternalAPIError` 502 with the upstream body
  truncated to keep logs readable.
* Network / timeout → :class:`ExternalAPIError` 502.

Retries
-------
We retry 5xx and timeouts twice with exponential backoff (0.5s, 1.0s).
4xx is never retried — it's our bug, not theirs.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from football_ai.config import platform_settings
from football_ai.core.exceptions import ExternalAPIError
from football_ai.external.quota import QuotaExceeded, QuotaTracker, get_quota_tracker
from football_ai.logger import get_logger

logger = get_logger(__name__)


_DEFAULT_TIMEOUT = 10.0  # seconds — generous for slow upstream
_RETRY_DELAYS = (0.5, 1.0)  # backoff between retries


class APIFootballClient:
    """Async client over the API-Football REST surface."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        quota: QuotaTracker | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else platform_settings.api_football_key
        self._base_url = (base_url or platform_settings.api_football_base_url).rstrip("/")
        self._quota = quota or get_quota_tracker()
        # An injected client lets tests use httpx-respx without owning the lifecycle.
        self._owned_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        if self._owned_client:
            await self._client.aclose()

    def is_configured(self) -> bool:
        return bool(self._api_key.strip())

    # ── Public endpoint wrappers ─────────────────────────────────────────────

    async def list_fixtures(self, *, league: int, season: int) -> list[dict[str, Any]]:
        """Return ``response`` array from /fixtures."""
        body = await self._get("/fixtures", params={"league": league, "season": season})
        return list(body.get("response", []))

    async def get_squad(self, *, team: int) -> list[dict[str, Any]]:
        """Return players from /players/squads (one team per call)."""
        body = await self._get("/players/squads", params={"team": team})
        squads = body.get("response", [])
        if not squads:
            return []
        # Provider returns one entry per team containing a ``players`` list.
        return list(squads[0].get("players", []))

    async def list_fixture_events(self, *, fixture: int) -> list[dict[str, Any]]:
        body = await self._get("/fixtures/events", params={"fixture": fixture})
        return list(body.get("response", []))

    async def raw_get(self, path: str, **params: Any) -> dict[str, Any]:
        """Escape hatch for ad-hoc admin probes. Quota and auth still enforced."""
        return await self._get(path, params=params)

    # ── Internal request path ────────────────────────────────────────────────

    async def _get(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        if not self.is_configured():
            raise ExternalAPIError(
                "API-Football integration not configured (set API_FOOTBALL_KEY).",
                status_code=503,
            )

        try:
            await self._quota.consume(1)
        except QuotaExceeded as exc:
            raise ExternalAPIError(str(exc), status_code=429) from exc

        url = f"{self._base_url}{path}"
        headers = {
            "x-apisports-key": self._api_key,
            "Accept": "application/json",
        }

        last_exc: Exception | None = None
        for attempt, delay in enumerate((0.0,) + _RETRY_DELAYS):
            if delay:
                await asyncio.sleep(delay)
            try:
                response = await self._client.get(url, params=params, headers=headers)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                logger.warning(
                    "API-Football network error on %s (attempt %d): %s",
                    path, attempt + 1, exc,
                )
                continue

            if response.status_code >= 500:
                last_exc = ExternalAPIError(
                    f"Upstream {response.status_code} from {path}", status_code=502
                )
                logger.warning(
                    "API-Football 5xx on %s (attempt %d): %s",
                    path, attempt + 1, response.status_code,
                )
                continue

            if response.status_code == 429:
                # Provider-side rate limit — surface immediately, no point retrying.
                raise ExternalAPIError(
                    "API-Football rate limit hit upstream.",
                    status_code=429,
                )

            if response.status_code >= 400:
                snippet = (response.text or "")[:200]
                raise ExternalAPIError(
                    f"API-Football {response.status_code} on {path}: {snippet}",
                    status_code=502,
                )

            try:
                return response.json()
            except ValueError as exc:
                raise ExternalAPIError(
                    "API-Football returned non-JSON body.",
                    status_code=502,
                ) from exc

        # All retries exhausted.
        raise ExternalAPIError(
            f"API-Football request to {path} failed after retries: {last_exc}",
            status_code=502,
        )


# ── Module singleton ─────────────────────────────────────────────────────────

_client: APIFootballClient | None = None


def get_api_football_client() -> APIFootballClient:
    """FastAPI dep — overridden in tests with a respx-mocked client."""
    global _client
    if _client is None:
        _client = APIFootballClient()
    return _client


def reset_api_football_client() -> None:
    """Test hook — drop the singleton (useful when monkeypatching settings)."""
    global _client
    _client = None
