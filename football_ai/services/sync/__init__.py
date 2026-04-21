"""
External data sync services.

Each module here owns one provider→ORM mapping (fixtures, squads,
events). They share a contract:

* Take a fully-constructed external client + an ``AsyncSession``.
* Upsert by ``api_football_id`` so re-runs are idempotent.
* Return a small :class:`SyncResult` summary the admin endpoint can
  return verbatim.

Why a separate ``services.sync`` namespace
------------------------------------------
``football_ai.services`` historically holds pure business logic (player
service, match service, etc.) backed by parquet. The sync layer is
side-effect heavy (network IO + DB writes) and has its own dependencies
(external client, quota); keeping it under ``services.sync`` avoids
mixing concerns while staying within the existing top-level layout.
"""

from __future__ import annotations

from .base import SyncResult
from .events import sync_fixture_events
from .fixtures import sync_league_fixtures
from .squads import sync_team_squad

__all__ = [
    "SyncResult",
    "sync_league_fixtures",
    "sync_team_squad",
    "sync_fixture_events",
]
