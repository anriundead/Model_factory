import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class ResearchCleanupTestCase(unittest.TestCase):
    def setUp(self):
        from flask import Flask
        from app.extensions import db
        from app.model_gateway import models as gateway_models  # noqa: F401

        self.root = tempfile.mkdtemp(prefix="research_cleanup_")
        self.runtime = os.path.join(self.root, "runtime")
        os.makedirs(self.runtime)
        self.app = Flask(__name__)
        self.app.config.update(
            SQLALCHEMY_DATABASE_URI=f"sqlite:///{os.path.join(self.root, 'gateway.sqlite')}",
            SQLALCHEMY_BINDS={"model_gateway": f"sqlite:///{os.path.join(self.root, 'gateway.sqlite')}"},
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
        )
        self.db = db
        db.init_app(self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()

    def tearDown(self):
        self.db.session.remove()
        self.db.drop_all()
        self.ctx.pop()
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, relative, content="private"):
        path = os.path.join(self.runtime, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def add_expired_records(self):
        from app.model_gateway.models import ResearchChunk, ResearchFile, ResearchJob, ServingApiKey

        expiry = datetime.utcnow() - timedelta(seconds=1)
        key = ServingApiKey(id="key-1", key_hash="digest", prefix="mk_live", last4="1234", owner_label="cleanup")
        source = ResearchFile(
            id="file-1", api_key_id=key.id, original_name="paper.pdf", source_kind="upload", status="ready",
            quarantine_path=self._write("quarantine/paper.pdf"), parsed_text_path=self._write("parsed/paper.txt"), expires_at=expiry,
        )
        job = ResearchJob(
            id="job-1", api_key_id=key.id, served_model_name="qwen", task_type="summary", status="completed",
            file_ids=[source.id], payload_path=self._write("payloads/job-1.json"), result_path=self._write("results/job-1.json"), expires_at=expiry,
        )
        chunk = ResearchChunk(
            id="chunk-1", file_id=source.id, api_key_id=key.id, ordinal=0, text="private evidence",
            locator={"kind": "page", "value": 1}, expires_at=expiry,
        )
        self.db.session.add_all([key, source, job, chunk])
        self.db.session.commit()
        return source, job, chunk


class TestResearchCleanup(ResearchCleanupTestCase):
    def test_expired_records_and_managed_payloads_are_physically_removed(self):
        from app.model_gateway.cleanup import purge_expired_research_data
        from app.model_gateway.models import ResearchChunk, ResearchFile, ResearchJob

        source, job, chunk = self.add_expired_records()
        source_id, job_id, chunk_id = source.id, job.id, chunk.id
        removed = purge_expired_research_data(self.db.session, self.runtime)

        self.assertEqual(removed, {"files": 1, "jobs": 1, "chunks": 1})
        self.assertIsNone(self.db.session.get(ResearchFile, source_id))
        self.assertIsNone(self.db.session.get(ResearchJob, job_id))
        self.assertIsNone(self.db.session.get(ResearchChunk, chunk_id))
        for relative in ("quarantine/paper.pdf", "parsed/paper.txt", "payloads/job-1.json", "results/job-1.json"):
            self.assertFalse(os.path.exists(os.path.join(self.runtime, relative)))

    def test_expired_source_removes_its_short_lived_vector_sidecar(self):
        from app.model_gateway.cleanup import purge_expired_research_data
        from app.model_gateway.vectors import vector_path

        source, _, _ = self.add_expired_records()
        path = vector_path(self.runtime, source.id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"temporary vector")

        purge_expired_research_data(self.db.session, self.runtime)

        self.assertFalse(os.path.exists(path))

    def test_cleanup_never_unlinks_a_path_outside_research_runtime(self):
        from app.model_gateway.cleanup import purge_expired_research_data
        from app.model_gateway.models import ResearchFile, ServingApiKey

        external = os.path.join(self.root, "must-survive.txt")
        with open(external, "w", encoding="utf-8") as handle:
            handle.write("do not delete")
        key = ServingApiKey(id="key-2", key_hash="digest-2", prefix="mk_live", last4="5678", owner_label="cleanup")
        source = ResearchFile(
            id="file-2", api_key_id=key.id, original_name="bad-path", source_kind="upload", status="ready",
            quarantine_path=external, expires_at=datetime.utcnow() - timedelta(seconds=1),
        )
        self.db.session.add_all([key, source])
        self.db.session.commit()

        purge_expired_research_data(self.db.session, self.runtime)

        self.assertTrue(os.path.isfile(external))
        self.assertIsNone(self.db.session.get(ResearchFile, source.id))


if __name__ == "__main__":
    unittest.main()
