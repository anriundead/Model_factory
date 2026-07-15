"""One-time, repeatable copy from the legacy Gateway SQLite tables."""
from __future__ import annotations

import os

from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import MetaData, Table, inspect, select
from sqlalchemy import create_engine

from app.extensions import db


GATEWAY_TABLES = (
    "serving_model_services",
    "serving_api_keys",
    "serving_requests",
    "serving_usage_records",
    "serving_events",
    "research_files",
    "research_jobs",
    "research_chunks",
    "gateway_quota_buckets",
)


def upgrade_gateway_schema(project_root: str, database_url: str) -> None:
    """Upgrade only the Gateway schema, stamping a complete legacy schema once."""
    config = AlembicConfig(os.path.join(project_root, "gateway_alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)
    engine = create_engine(database_url)
    try:
        existing = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    managed = set(db.metadatas["model_gateway"].tables)
    present = existing & managed
    if present and present != managed:
        missing = ", ".join(sorted(managed - present))
        raise RuntimeError(f"Gateway schema is incomplete; missing tables: {missing}")
    if present and "alembic_version" not in existing:
        command.stamp(config, "head")
    command.upgrade(config, "head")
    command.check(config)


def copy_legacy_gateway_rows(source_engine, target_engine) -> dict[str, int]:
    """Copy rows once by primary key; source data is never modified."""
    metadata = db.metadatas["model_gateway"]
    source_tables = set(inspect(source_engine).get_table_names())
    source_metadata = MetaData()
    copied: dict[str, int] = {}
    for name in GATEWAY_TABLES:
        copied[name] = 0
        if name not in source_tables or name not in metadata.tables:
            continue
        table = metadata.tables[name]
        source_table = Table(name, source_metadata, autoload_with=source_engine)
        target_columns = set(table.c.keys())
        with source_engine.connect() as source_conn:
            rows = [
                {column: value for column, value in row.items() if column in target_columns}
                for row in source_conn.execute(select(source_table)).mappings()
            ]
        if not rows:
            continue
        with target_engine.begin() as target_conn:
            existing = set(target_conn.execute(select(table.c.id)).scalars())
            pending = [row for row in rows if row.get("id") not in existing]
            if pending:
                target_conn.execute(table.insert(), pending)
                copied[name] = len(pending)
    return copied
