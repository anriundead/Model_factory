import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class ResearchLifecycleTestCase(unittest.TestCase):
    def setUp(self):
        from flask import Flask
        from app.extensions import db
        from app.model_gateway import models as gateway_models  # noqa: F401

        self.db = db
        self.temp_dir = tempfile.mkdtemp(prefix="research_lifecycle_")
        self.db_path = os.path.join(self.temp_dir, "gateway.sqlite")
        self.app = Flask(__name__)
        self.app.config.update(
            SQLALCHEMY_DATABASE_URI=f"sqlite:///{self.db_path}",
            SQLALCHEMY_BINDS={"model_gateway": f"sqlite:///{self.db_path}"},
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
        )
        db.init_app(self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()

    def tearDown(self):
        self.db.session.remove()
        self.db.drop_all()
        self.ctx.pop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def make_job(self, service_status="running", job_status="queued", expires_at=None):
        from app.model_gateway.models import ResearchJob, ServingApiKey, ServingModelService

        service = ServingModelService(
            id="service-1",
            model_path="/models/qwen",
            display_name="Qwen",
            served_model_name="qwen-test",
            status=service_status,
        )
        key = ServingApiKey(
            id="key-1",
            key_hash="digest",
            prefix="mk_live",
            last4="1234",
            owner_label="test",
        )
        job = ResearchJob(
            id="job-1",
            api_key_id=key.id,
            model_service_id=service.id,
            served_model_name=service.served_model_name,
            task_type="summary",
            status=job_status,
            file_ids=["file-1"],
            expires_at=expires_at or datetime.utcnow() + timedelta(hours=1),
        )
        self.db.session.add_all([service, key, job])
        self.db.session.commit()
        return job


class TestResearchLifecycle(ResearchLifecycleTestCase):
    def test_fail_records_reason_and_clears_running_lease(self):
        from app.model_gateway.lifecycle import fail_research_job
        from app.model_gateway.models import ResearchJob

        self.make_job()
        self.assertEqual(
            __import__("app.model_gateway.lifecycle", fromlist=["claim_research_job"]).claim_research_job(
                self.db.session, "job-1", "worker-a", now=datetime.utcnow()
            ),
            "claimed",
        )

        self.assertEqual(
            fail_research_job(self.db.session, "job-1", "worker-a", "citation_validation_failed"),
            "failed",
        )
        job = self.db.session.get(ResearchJob, "job-1")
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error_code, "citation_validation_failed")
        self.assertIsNone(job.lease_owner)
        self.assertIsNotNone(job.finished_at)

    def test_claim_requires_running_model_and_creates_lease(self):
        from app.model_gateway.lifecycle import claim_research_job
        from app.model_gateway.models import ResearchJob

        self.make_job(service_status="stopped")
        outcome = claim_research_job(self.db.session, "job-1", "worker-a", now=datetime.utcnow())
        self.assertEqual(outcome, "paused_model_offline")
        self.assertEqual(self.db.session.get(ResearchJob, "job-1").status, "paused_model_offline")

        from app.model_gateway.models import ServingModelService

        self.db.session.get(ServingModelService, "service-1").status = "running"
        self.db.session.commit()
        outcome = claim_research_job(self.db.session, "job-1", "worker-a", now=datetime.utcnow())
        claimed = self.db.session.get(ResearchJob, "job-1")

        self.assertEqual(outcome, "claimed")
        self.assertEqual(claimed.status, "running")
        self.assertEqual(claimed.lease_owner, "worker-a")
        self.assertIsNotNone(claimed.lease_expires_at)

    def test_reconcile_recovers_only_expired_running_lease(self):
        from app.model_gateway.lifecycle import reconcile_research_jobs
        from app.model_gateway.models import ResearchJob

        self.make_job(job_status="running")
        job = self.db.session.get(ResearchJob, "job-1")
        now = datetime.utcnow()
        job.lease_owner = "dead-worker"
        job.lease_expires_at = now - timedelta(seconds=1)
        self.db.session.commit()

        recovered = reconcile_research_jobs(self.db.session, now=now)
        job = self.db.session.get(ResearchJob, "job-1")

        self.assertEqual(recovered, ["job-1"])
        self.assertEqual(job.status, "queued")
        self.assertIsNone(job.lease_owner)
        self.assertIsNone(job.lease_expires_at)

    def test_reconcile_never_restores_cancel_requested_job(self):
        from app.model_gateway.lifecycle import reconcile_research_jobs
        from app.model_gateway.models import ResearchJob

        self.make_job(job_status="cancel_requested")
        recovered = reconcile_research_jobs(self.db.session, now=datetime.utcnow())
        job = self.db.session.get(ResearchJob, "job-1")

        self.assertEqual(recovered, [])
        self.assertEqual(job.status, "canceled")
        self.assertIsNotNone(job.finished_at)

    def test_reconcile_expires_job_and_removes_payload(self):
        from app.model_gateway.lifecycle import reconcile_research_jobs
        from app.model_gateway.models import ResearchJob

        payload_path = os.path.join(self.temp_dir, "payload.json")
        with open(payload_path, "w", encoding="utf-8") as handle:
            handle.write('{"private":"text"}')
        self.make_job(expires_at=datetime.utcnow() - timedelta(seconds=1))
        job = self.db.session.get(ResearchJob, "job-1")
        job.payload_path = payload_path
        self.db.session.commit()

        self.assertEqual(reconcile_research_jobs(self.db.session, now=datetime.utcnow()), [])
        job = self.db.session.get(ResearchJob, "job-1")
        self.assertEqual(job.status, "expired")
        self.assertIsNone(job.payload_path)
        self.assertFalse(os.path.exists(payload_path))


if __name__ == "__main__":
    unittest.main()
