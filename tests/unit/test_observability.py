"""
Unit tests for the observability primitives.

We cover:

* Context vars round-trip (set → get → clear).
* JSON formatter output shape + request_id propagation.
* Metrics registry increments on counter/histogram ops.
"""

from __future__ import annotations

import json
import logging

from football_ai.observability.context import (
    bind_request_id,
    bind_user_id,
    clear_request_context,
    current_request_id,
    current_user_id,
)
from football_ai.observability.logging import JsonLogFormatter, RequestContextFilter
from football_ai.observability.metrics import (
    PREDICTIONS_TOTAL,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
)

# ── Context vars ─────────────────────────────────────────────────────────────

def test_bind_request_id_returns_generated_when_absent():
    clear_request_context()
    rid = bind_request_id()
    assert rid and len(rid) >= 16
    assert current_request_id() == rid
    clear_request_context()


def test_bind_request_id_preserves_incoming_header():
    clear_request_context()
    rid = bind_request_id("trace-abc-123")
    assert rid == "trace-abc-123"
    clear_request_context()


def test_bind_user_id_round_trip():
    clear_request_context()
    bind_user_id("user-42")
    assert current_user_id() == "user-42"
    clear_request_context()
    assert current_user_id() is None


# ── JSON formatter ───────────────────────────────────────────────────────────

def _record(msg: str, **extra) -> logging.LogRecord:
    r = logging.LogRecord(
        name="football_ai.test", level=logging.INFO, pathname="x", lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    for k, v in extra.items():
        setattr(r, k, v)
    return r


def test_json_formatter_emits_required_keys():
    clear_request_context()
    bind_request_id("rid-1")
    bind_user_id("u-7")
    fmt = JsonLogFormatter()
    filt = RequestContextFilter()
    rec = _record("hello world", path="/x", status=200)
    filt.filter(rec)
    payload = json.loads(fmt.format(rec))

    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "rid-1"
    assert payload["user_id"] == "u-7"
    assert payload["path"] == "/x"
    assert payload["status"] == 200
    assert payload["ts"].endswith("Z")
    clear_request_context()


def test_json_formatter_ignores_unserialisable_extras():
    fmt = JsonLogFormatter()
    rec = _record("msg", broken=object())
    payload = json.loads(fmt.format(rec))
    assert "broken" not in payload


# ── Metrics ──────────────────────────────────────────────────────────────────

def test_requests_counter_increments():
    before = REQUESTS_TOTAL.labels(method="GET", route="/x", status="200")._value.get()
    REQUESTS_TOTAL.labels(method="GET", route="/x", status="200").inc()
    after = REQUESTS_TOTAL.labels(method="GET", route="/x", status="200")._value.get()
    assert after == before + 1


def test_duration_histogram_observes():
    # Simply assert we can record without raising.
    REQUEST_DURATION.labels(method="GET", route="/x").observe(0.123)


def test_predictions_counter_has_label():
    PREDICTIONS_TOTAL.labels(outcome="home_win").inc()
    val = PREDICTIONS_TOTAL.labels(outcome="home_win")._value.get()
    assert val >= 1
