"""
External provider clients.

Currently houses the API-Football integration. Anthropic lives under
``football_ai/ml/services/scouting_service.py`` because it's an AI
service, not a data source — the line is "does it bring in match data
that we persist?"; only API-Football does today.
"""

from __future__ import annotations

from .api_football import APIFootballClient, get_api_football_client
from .quota import QuotaTracker, get_quota_tracker

__all__ = [
    "APIFootballClient",
    "QuotaTracker",
    "get_api_football_client",
    "get_quota_tracker",
]
