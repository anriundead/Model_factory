import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class GatewayQuotaTestCase(unittest.TestCase):
    def setUp(self):
        from flask import Flask
        from app.extensions import db
        from app.model_gateway import models as gateway_models  # noqa: F401

        self.root = tempfile.mkdtemp(prefix="gateway_quota_")
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
        from app.model_gateway.models import ServingApiKey
        self.db.session.add(ServingApiKey(id="key-1", key_hash="digest", prefix="mk_live", last4="1234", owner_label="quota"))
        self.db.session.commit()

    def tearDown(self):
        self.db.session.remove()
        self.db.drop_all()
        self.ctx.pop()
        shutil.rmtree(self.root, ignore_errors=True)


class TestGatewayQuotas(GatewayQuotaTestCase):
    def test_window_rejects_usage_after_limit(self):
        from app.model_gateway.quotas import QuotaExceeded, reserve_quota

        now = datetime(2026, 7, 15, 12, 0, 10)
        reserve_quota(self.db.session, "key-1", "chat_requests", amount=1, limit=2, window_seconds=60, now=now)
        reserve_quota(self.db.session, "key-1", "chat_requests", amount=1, limit=2, window_seconds=60, now=now)
        with self.assertRaises(QuotaExceeded) as raised:
            reserve_quota(self.db.session, "key-1", "chat_requests", amount=1, limit=2, window_seconds=60, now=now)

        self.assertEqual(raised.exception.code, "chat_rate_limit_exceeded")
        self.assertGreater(raised.exception.retry_after_seconds, 0)

    def test_releasing_active_job_slot_allows_next_submission(self):
        from app.model_gateway.quotas import QuotaExceeded, release_quota, reserve_quota

        now = datetime(2026, 7, 15, 12, 0, 10)
        reserve_quota(self.db.session, "key-1", "active_research_jobs", amount=1, limit=1, window_seconds=86400, now=now)
        with self.assertRaises(QuotaExceeded):
            reserve_quota(self.db.session, "key-1", "active_research_jobs", amount=1, limit=1, window_seconds=86400, now=now)
        release_quota(self.db.session, "key-1", "active_research_jobs", amount=1, window_seconds=86400, now=now)
        reserve_quota(self.db.session, "key-1", "active_research_jobs", amount=1, limit=1, window_seconds=86400, now=now)


if __name__ == "__main__":
    unittest.main()

