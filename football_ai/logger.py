"""
Logging setup for the Football AI Platform.

On first import, ``setup_logging()`` must be called once (typically by
``main.py``, ``app.py``, or a script entry point).  All subsequent calls
to ``get_logger(__name__)`` then return a correctly configured child logger.

Configuration is loaded from ``configs/logging.yaml`` if it exists; otherwise
a sensible default is applied.  The ``LOG_LEVEL`` environment variable
overrides the YAML console level at runtime.

Usage
-----
>>> from football_ai.logger import get_logger, setup_logging
>>> setup_logging()
>>> logger = get_logger(__name__)
>>> logger.info("Ready.")
"""

from __future__ import annotations

import logging
import logging.config
import logging.handlers
import os
from pathlib import Path

import yaml

_CONFIGURED = False

# Paths
_ROOT          = Path(__file__).resolve().parent.parent
_LOG_YAML      = _ROOT / "configs" / "logging.yaml"
_LOG_DIR       = _ROOT / "logs"
_DEFAULT_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_ENVIRONMENT   = os.environ.get("ENVIRONMENT", "development")

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATE   = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """
    Configure the Python logging system.

    Loads ``configs/logging.yaml`` when it exists; falls back to a
    programmatic default.  Safe to call multiple times — only configures
    once per process.

    Parameters
    ----------
    level : str, optional
        Override log level (e.g. "DEBUG", "WARNING").
        Defaults to the ``LOG_LEVEL`` environment variable (default "INFO").
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    effective_level = (level or _DEFAULT_LEVEL).upper()

    # Ensure logs/ directory exists
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    if _LOG_YAML.exists():
        _configure_from_yaml(effective_level)
    else:
        _configure_default(effective_level)

    _CONFIGURED = True


def _configure_from_yaml(effective_level: str) -> None:
    """Load logging config from YAML and override the console level."""
    with open(_LOG_YAML, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Override console handler level with runtime env var
    for handler in config.get("handlers", {}).values():
        if handler.get("class") == "logging.StreamHandler":
            handler["level"] = effective_level

    # Override root football_ai logger level too
    if "football_ai" in config.get("loggers", {}):
        config["loggers"]["football_ai"]["level"] = effective_level

    # Ensure the log file path is absolute
    for handler in config.get("handlers", {}).values():
        if "filename" in handler:
            filename = Path(handler["filename"])
            if not filename.is_absolute():
                handler["filename"] = str(_ROOT / filename)
                Path(handler["filename"]).parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)


def _configure_default(effective_level: str) -> None:
    """Programmatic fallback when logging.yaml is absent."""
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(effective_level)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE))

    # File handler
    try:
        fh = logging.handlers.RotatingFileHandler(
            filename=str(_LOG_DIR / "football_ai.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel("INFO")
        fh.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE))
    except OSError:
        fh = None

    pkg_logger = logging.getLogger("football_ai")
    pkg_logger.setLevel(effective_level)
    pkg_logger.handlers.clear()
    pkg_logger.addHandler(handler)
    if fh:
        pkg_logger.addHandler(fh)
    pkg_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the ``football_ai`` hierarchy.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    if not _CONFIGURED:
        # Lazy-init with defaults so library usage never crashes
        setup_logging()
    return logging.getLogger(name)
