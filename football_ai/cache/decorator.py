"""
``@cached`` — endpoint-level result caching with TTL + JSON encoding.

Design notes
------------
* **Key shape**: ``"<namespace>:<func_name>:<arg_hash>"``. We hash kwargs
  (the FastAPI path/query/dep parameters) into a stable digest so that
  two calls with the same query produce the same key, regardless of dep
  iteration order. We deliberately exclude ``Request`` / ``AsyncSession``
  / authenticated user objects from the key — they're per-request and
  would shatter the cache.
* **Encoding**: pydantic models go through ``model_dump_json``; dicts and
  lists go through stdlib ``json``. Any other return type is cached as
  its ``repr`` only when it's bytes — otherwise the wrapper bails and
  the function runs uncached. Endpoint return types in this codebase are
  always pydantic ``BaseModel``s, so the bail-out is a safety net, not a
  hot path.
* **Cache failures don't break endpoints**: a Redis hiccup logs a warning
  and falls through to the live function. Caching is a perf optimisation,
  not a correctness requirement.
"""

from __future__ import annotations

import functools
import hashlib
import json
import uuid
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from football_ai.logger import get_logger

from .backends import Cache
from .factory import get_cache

logger = get_logger(__name__)


# Argument types that must NOT contribute to the cache key. They're
# either per-request (sessions, requests) or unhashable / sensitive
# (the current User row).
_OPAQUE_TYPE_NAMES = frozenset(
    {
        "AsyncSession",
        "Session",
        "Request",
        "Response",
        "WebSocket",
        "BackgroundTasks",
        "User",
        "Cache",
        "InMemoryCache",
        "RedisCache",
    }
)


def cached(
    ttl: int,
    *,
    namespace: str,
    response_model: type[BaseModel] | None = None,
) -> Callable:
    """Cache the JSON-encoded result of an async endpoint.

    Parameters
    ----------
    ttl
        Seconds to keep the cached payload. Pick based on volatility:
        rankings/accuracy → 60s, fixtures → 30s, live data → don't cache.
    namespace
        Logical bucket (e.g. ``"rankings"``). Lets you wipe one feature's
        cache without nuking the rest.
    response_model
        If provided, cached payloads are rehydrated into this pydantic
        model before being returned. Required when the endpoint's return
        annotation is a pydantic generic (FastAPI inspects the return
        type, so we must hand it back the right class).
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                key = _build_key(namespace, func.__name__, kwargs)
            except _Unhashable:
                # Args contain something we can't safely key — skip cache.
                return await func(*args, **kwargs)

            cache: Cache = await get_cache()

            # Read path
            try:
                hit = await cache.get(key)
            except Exception as exc:  # pragma: no cover — infra error
                logger.warning("Cache GET failed (%s) — bypassing cache.", exc)
                hit = None

            if hit is not None:
                try:
                    return _decode(hit, response_model)
                except Exception as exc:  # pragma: no cover — defensive
                    logger.warning("Cache decode failed (%s) — re-running.", exc)

            # Miss → run + write
            result = await func(*args, **kwargs)

            try:
                payload = _encode(result)
                if payload is not None:
                    await cache.set(key, payload, ttl=ttl)
            except Exception as exc:  # pragma: no cover — infra error
                logger.warning("Cache SET failed (%s) — result returned uncached.", exc)

            return result

        return wrapper

    return decorator


# ── Key building ─────────────────────────────────────────────────────────────

class _Unhashable(Exception):
    """Raised when a kwarg can't be safely included in the cache key."""


def _build_key(namespace: str, func_name: str, kwargs: dict[str, Any]) -> str:
    """Stable hash over the keyword arguments (path/query/deps)."""
    parts: list[tuple[str, str]] = []
    for name, value in sorted(kwargs.items()):
        if value is None:
            parts.append((name, "null"))
            continue
        type_name = type(value).__name__
        if type_name in _OPAQUE_TYPE_NAMES:
            # Per-request object — must not leak into the key.
            continue
        try:
            parts.append((name, _stringify(value)))
        except _Unhashable:
            raise

    digest = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:16]
    return f"{namespace}:{func_name}:{digest}"


def _stringify(value: Any) -> str:
    """Convert a kwarg into a stable string for hashing."""
    if isinstance(value, (str, int, float, bool)):
        return f"{type(value).__name__}:{value}"
    if isinstance(value, uuid.UUID):
        return f"uuid:{value}"
    if isinstance(value, (datetime, date)):
        return f"dt:{value.isoformat()}"
    if isinstance(value, (list, tuple)):
        return "list:[" + ",".join(_stringify(v) for v in value) + "]"
    if isinstance(value, dict):
        return "dict:{" + ",".join(
            f"{k}={_stringify(v)}" for k, v in sorted(value.items())
        ) + "}"
    raise _Unhashable(f"cannot key on {type(value).__name__}")


# ── Encoding ─────────────────────────────────────────────────────────────────

def _encode(value: Any) -> bytes | None:
    """Serialise an endpoint return value for storage."""
    if value is None:
        return b"null"
    if isinstance(value, BaseModel):
        return value.model_dump_json().encode("utf-8")
    if isinstance(value, (dict, list, str, int, float, bool)):
        return json.dumps(value, default=_json_default).encode("utf-8")
    return None  # Unsupported — caller skips writing.


def _decode(payload: bytes, model: type[BaseModel] | None) -> Any:
    if payload == b"null":
        return None
    if model is not None:
        return model.model_validate_json(payload)
    return json.loads(payload)


def _json_default(o: Any) -> Any:
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, uuid.UUID):
        return str(o)
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")
