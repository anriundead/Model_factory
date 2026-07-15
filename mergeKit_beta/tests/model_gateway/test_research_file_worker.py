import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class FakeRedis:
    def __init__(self):
        self.acks = []
        self.messages = []

    def xack(self, stream, group, message_id):
        self.acks.append((stream, group, message_id))

    def xadd(self, stream, fields):
        self.messages.append((stream, fields))
        return b"1-0"


class FileWorkerTestCase(unittest.TestCase):
    def setUp(self):
        from flask import Flask
        from app.extensions import db
        from app.model_gateway import models as gateway_models  # noqa: F401

        self.root = tempfile.mkdtemp(prefix="file_worker_")
        db_path = os.path.join(self.root, "gateway.sqlite")
        self.app = Flask(__name__)
        self.app.config.update(
            SQLALCHEMY_DATABASE_URI=f"sqlite:///{db_path}",
            SQLALCHEMY_BINDS={"model_gateway": f"sqlite:///{db_path}"},
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

    def add_received_upload(self, file_id="file-1"):
        from app.model_gateway.models import ResearchFile, ServingApiKey

        key = ServingApiKey(id="key-1", key_hash="digest", prefix="mk_live", last4="1234", owner_label="worker")
        source = ResearchFile(
            id=file_id,
            api_key_id=key.id,
            original_name="report.pdf",
            source_kind="upload",
            status="received",
            quarantine_path="/private/quarantine/report.pdf",
            expires_at=datetime.utcnow() + timedelta(hours=24),
        )
        self.db.session.add_all([key, source])
        self.db.session.commit()


class TestResearchFileWorker(FileWorkerTestCase):
    def test_worker_requires_conda_cxx_runtime_preload(self):
        from app.model_gateway.file_worker import CONDA_CXX_RUNTIME, require_worker_preload

        with patch.dict(os.environ, {"LD_PRELOAD": ""}):
            with self.assertRaisesRegex(RuntimeError, "Conda C\\+\\+ runtime"):
                require_worker_preload()
        with patch.dict(os.environ, {"LD_PRELOAD": CONDA_CXX_RUNTIME}):
            require_worker_preload()

    @patch("app.model_gateway.file_worker.process_research_file", return_value="ready")
    def test_delivery_processes_then_acknowledges(self, process):
        from app.model_gateway.file_worker import consume_file_message
        from app.model_gateway.queue import FILE_CONSUMER_GROUP, FILE_STREAM

        redis = FakeRedis()
        outcome = consume_file_message(self.db.session, redis, "1-0", {b"file_id": b"file-1"})

        self.assertEqual(outcome, "ready")
        process.assert_called_once_with(self.db.session, "file-1")
        self.assertEqual(redis.acks, [(FILE_STREAM, FILE_CONSUMER_GROUP, "1-0")])

    def test_retry_is_acknowledged_then_reconciled_from_database(self):
        from app.model_gateway.file_worker import consume_file_message, reconcile_received_files

        self.add_received_upload()
        redis = FakeRedis()
        with patch("app.model_gateway.file_worker.process_research_file", return_value="retry"):
            self.assertEqual(consume_file_message(self.db.session, redis, "1-0", {"file_id": "file-1"}), "retry")

        requeued = reconcile_received_files(self.db.session, redis)

        self.assertEqual(requeued, ["file-1"])
        self.assertEqual(redis.messages[-1][1], {"file_id": "file-1"})


if __name__ == "__main__":
    unittest.main()
