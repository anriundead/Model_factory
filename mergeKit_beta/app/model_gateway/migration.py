"""One-time, repeatable copy from the legacy Gateway SQLite tables."""
from __future__ import annotations

from sqlalchemy import MetaData, Table, inspect, select

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
)


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
