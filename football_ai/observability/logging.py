"""
Structured (JSON) logging with request-context enrichment.

Two pieces:

* :class:`RequestContextFilter` — a stdlib :class:`logging.Filter` that
  stamps every record with ``request_id`` and ``user_id`` (read from
  the contextvars in :mod:`~.context`). Attaches to both text and JSON
  handlers.
* :class:`JsonLogFormatter` — emits one JSON object per record. Keys
  chosen to match what most log processors (Loki/Datadog/ELK) expect.

Why both formatters
-------------------
Humans in development want colourful, readable text. Production wants
machine-parseable JSON. :func:`install_json_logging` flips the whole
``football_ai`` hierarchy to JSON without touching unrelated libraries.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .context import current_request_id, current_user_id


class RequestContextFilter(logging.Filter):
    """Inject request-scoped context into every record.

    Safe to attach multiple times — the filter only sets attributes that
    aren't already present, so a handler-specific override (e.g. test
    fakery) still wins.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id() or "-"
        if not hasattr(record, "user_id"):
            record.user_id = current_user_id() or "-"
        return True


class JsonLogFormatter(logging.Formatter):
    """Formatter that serialises each record to a single JSON object.

    The field list is intentionally small — anything exotic belongs in
    the ``extra={}`` block at the log call site and is merged in here.
    """

    # Stdlib attrs we don't want leaking into the JSON blob. Anything not
    # in this blocklist and not a dunder attribute is forwarded as-is.
    _RESERVED = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self._iso_ts(record.created),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "user_id": getattr(record, "user_id", "-"),
        }
        # Attach any caller-supplied `extra={}` keys.
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_") or key in payload:
                continue
            # Log records carry a few internal attrs (e.g. request_id/user_id
            # already added above). Skip non-serialisable values silently.
            try:
                json.dumps(value)
            except TypeError:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _iso_ts(epoch_seconds: float) -> str:
        # Millisecond precision is enough; includes 'Z' so downstream
        # systems parse it as UTC.
        ms = int((epoch_seconds - int(epoch_seconds)) * 1000)
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(epoch_seconds)) + f".{ms:03d}Z"


def install_json_logging(level: str = "INFO") -> None:
    """Swap the ``football_ai`` logger tree over to JSON output.

    Leaves the root logger (and third-party loggers) alone so uvicorn's
    own access log, if enabled, keeps its native format.
    """
    root = logging.getLogger("football_ai")
    root.setLevel(level)

    context_filter = RequestContextFilter()

    # Clear any existing handlers so we don't double-print.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(JsonLogFormatter())
    handler.addFilter(context_filter)
    root.addHandler(handler)
    root.propagate = False
