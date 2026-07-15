import os
import tempfile
import unittest
from datetime import datetime

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class TestGatewayLegacyMigration(unittest.TestCase):
    def test_existing_complete_gateway_schema_is_stamped_then_checked(self):
        from unittest.mock import MagicMock, patch

        from app.model_gateway import migration

        existing = MagicMock()
        existing.get_table_names.return_value = list(migration.GATEWAY_TABLES)
        with patch("app.model_gateway.migration.create_engine"), \
             patch("app.model_gateway.migration.inspect", return_value=existing), \
             patch("app.model_gateway.migration.command.stamp") as stamp, \
             patch("app.model_gateway.migration.command.upgrade") as upgrade, \
             patch("app.model_gateway.migration.command.check") as check:
            migration.upgrade_gateway_schema("/project", "postgresql+psycopg://gateway")

        stamp.assert_called_once()
        upgrade.assert_called_once()
        check.assert_called_once()

    def test_copy_is_idempotent_and_preserves_key_hash(self):
        from sqlalchemy import create_engine, insert, select
        from app.extensions import db
        from app.model_gateway.migration import copy_legacy_gateway_rows
        from app.model_gateway import models  # noqa: F401

        source = create_engine("sqlite:///:memory:")
        target = create_engine("sqlite:///:memory:")
        metadata = db.metadatas["model_gateway"]
        metadata.create_all(source)
        metadata.create_all(target)
        keys = metadata.tables["serving_api_keys"]
        with source.begin() as conn:
            conn.execute(insert(keys).values(
                id="key-1", key_hash="digest-only", prefix="mk_live", last4="1234", owner_label="migration-test", status="active"
            ))

        first = copy_legacy_gateway_rows(source, target)
        second = copy_legacy_gateway_rows(source, target)
        with target.connect() as conn:
            row = conn.execute(select(keys.c.key_hash)).scalar_one()

        self.assertEqual(first["serving_api_keys"], 1)
        self.assertEqual(second["serving_api_keys"], 0)
        self.assertEqual(row, "digest-only")

    def test_copy_accepts_legacy_table_without_new_columns(self):
        from sqlalchemy import Boolean, Column, DateTime, MetaData, String, Table, create_engine, insert, select
        from app.extensions import db
        from app.model_gateway.migration import copy_legacy_gateway_rows
        from app.model_gateway import models  # noqa: F401

        source = create_engine("sqlite:///:memory:")
        target = create_engine("sqlite:///:memory:")
        legacy = MetaData()
        jobs = Table(
            "research_jobs",
            legacy,
            Column("id", String(36), primary_key=True),
            Column("api_key_id", String(36), nullable=False),
            Column("served_model_name", String(128), nullable=False),
            Column("task_type", String(32), nullable=False),
            Column("status", String(32), nullable=False),
            Column("file_ids", db.JSON, nullable=False),
            Column("output_format", String(16), nullable=False),
            Column("require_citations", Boolean, nullable=False),
            Column("expires_at", DateTime, nullable=False),
            Column("created_at", DateTime, nullable=False),
            Column("updated_at", DateTime, nullable=False),
        )
        legacy.create_all(source)
        db.metadatas["model_gateway"].create_all(target)
        now = datetime.utcnow()
        with source.begin() as conn:
            conn.execute(insert(jobs).values(
                id="job-legacy",
                api_key_id="key-legacy",
                served_model_name="qwen-test",
                task_type="summary",
                status="queued",
                file_ids=[],
                output_format="markdown",
                require_citations=True,
                expires_at=now,
                created_at=now,
                updated_at=now,
            ))

        copied = copy_legacy_gateway_rows(source, target)
        target_jobs = db.metadatas["model_gateway"].tables["research_jobs"]
        with target.connect() as conn:
            row = conn.execute(select(target_jobs.c.request_fingerprint).where(target_jobs.c.id == "job-legacy")).one()

        self.assertEqual(copied["research_jobs"], 1)
        self.assertIsNone(row.request_fingerprint)


if __name__ == "__main__":
    unittest.main()
