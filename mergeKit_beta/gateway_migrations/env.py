"""Alembic environment for the Gateway PostgreSQL schema only."""
from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine, pool

os.environ.setdefault("MERGEKIT_CLI_SCRIPT", "1")

from config import Config
from app.extensions import db
from app.model_gateway import models  # noqa: F401


config = context.config
database_url = os.environ.get("MERGEKIT_MODEL_GATEWAY_DATABASE_URL") or Config.MERGEKIT_MODEL_GATEWAY_DATABASE_URL
if not database_url:
    raise RuntimeError("MERGEKIT_MODEL_GATEWAY_DATABASE_URL is required for Gateway migrations")
config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
target_metadata = db.metadatas["model_gateway"]


def run_migrations_offline():
    context.configure(url=database_url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = create_engine(database_url, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
