"""
SQLAlchemy declarative base + shared column mixins.

Design notes
------------
* One ``Base`` — every ORM class inherits from it, and Alembic sees them all
  via ``football_ai.models.__init__`` which imports each model module.

* UUID primary keys — a UUID is stable across external-ID churn (API-Football
  can re-issue numeric IDs; Transfermarkt IDs look similar). Our internal PK
  must not depend on a provider.

* TimestampMixin — every entity gets ``created_at`` and ``updated_at`` because
  analytics needs "what did we believe about this player as of T?". Server-
  side defaults + ``onupdate`` keep the clock authoritative on the DB.

* SoftDeleteMixin — opt-in on tables where history matters. Example: when a
  player transfers clubs we soft-delete the old Player.current_team_id
  association row (not the player); this keeps season-level reports stable.
  Filtering is explicit at the CRUD layer, not via a global event listener,
  so admin sync code can still see deleted rows.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for every ORM model in the platform."""


class UUIDPKMixin:
    """UUID v4 primary key. Generated Python-side on insert."""

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )


class TimestampMixin:
    """``created_at`` / ``updated_at`` managed by the database clock."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """
    Opt-in soft-delete.

    No global query filter is installed — CRUD helpers must filter explicitly
    (see ``football_ai.crud.*``). Admin + reconciliation code reads deleted
    rows on purpose.
    """

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        server_default="0",
        default=False,
        nullable=False,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
