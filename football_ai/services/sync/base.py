"""
Shared types for the sync layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class SyncResult:
    """Outcome of a single sync invocation.

    Returned verbatim by ``/admin/sync/*`` so operators can see what changed
    without grepping logs. ``warnings`` carries soft failures (e.g. one bad
    fixture in a 200-fixture league pull) that didn't merit aborting the
    whole sync.
    """

    target: str                    # e.g. "fixtures:39:2024"
    fetched: int = 0               # rows pulled from upstream
    created: int = 0
    updated: int = 0
    skipped: int = 0               # had nothing changed worth writing
    warnings: list[str] = field(default_factory=list)
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "fetched": self.fetched,
            "created": self.created,
            "updated": self.updated,
            "skipped": self.skipped,
            "warnings": self.warnings,
            "completed_at": self.completed_at.isoformat(),
        }
