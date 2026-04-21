"""
Alembic environment.

Loads the live platform settings so the same DATABASE_URL drives both
runtime and migrations. Autogenerate uses the sync-dialect URL (asyncpg
doesn't participate in Alembic's introspection step).

Football-domain reason for ``compare_type=True``: the stat columns (xG,
xA, rating) are Float/Numeric and occasionally we tighten types when
precision requirements change. compare_type catches those silently-
skipped diffs so no precision drift ships.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Register every model so metadata is populated.
from football_ai.config import platform_settings
from football_ai.db.base import Base
from football_ai import models  # noqa: F401 — import side effect registers tables

config = context.config

# Override the dummy URL in alembic.ini with the real one from env.
config.set_main_option("sqlalchemy.url", platform_settings.sync_database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without opening a connection (for review/CI)."""
    context.configure(
        url=platform_settings.sync_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=platform_settings.is_sqlite,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Open a synchronous connection and run migrations against it."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            # SQLite lacks real ALTER TABLE — batch mode recreates tables
            # transparently so migrations behave the same across dialects.
            render_as_batch=platform_settings.is_sqlite,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
