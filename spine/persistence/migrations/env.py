"""Alembic environment configuration for SPINE migrations."""

from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
from alembic import context

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support

# Get the database path from environment or use default
db_path = os.environ.get("SPINE_DB_PATH", ".spine/spine.db")

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = f"sqlite:///{db_path}"
    context.configure(
        url=url,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # Enable sqlite-vec extension
    connectable = engine_from_config(
        {"sqlalchemy.url": f"sqlite:///{db_path}"},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    def process_revision_directives(connection, directives, **kwargs):
        """Process revision directives after migrations."""
        # Enable sqlite-vec after table creation
        connection.execute("SELECT load_extension('vec0')")

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            process_revision_directives=process_revision_directives,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()