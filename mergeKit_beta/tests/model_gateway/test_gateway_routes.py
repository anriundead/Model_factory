"""Serving HTTP route tests."""
import os
import json
import shutil
import tempfile
import time
import unittest
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from unittest.mock import patch

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class FakeVllmResponse:
    status_code = 200
    headers = {"content-type": "application/json"}
    text = '{"ok": true}'
    content = b'{"ok": true}'

    def json(self):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }


class FakeStreamingResponse:
    status_code = 200
    headers = {"content-type": "text/event-stream"}
    text = ""

    def iter_content(self, chunk_size=None):
        yield b"data: hello\n\n"


class ServingRoutesTestCase(unittest.TestCase):
    def setUp(self):
        from flask import Flask
        from app.extensions import db
        from app import models as base_models  # noqa: F401
        from app.model_gateway import models as serving_models  # noqa: F401
        from app.model_gateway import register_model_gateway

        self.db = db
        db_file = tempfile.NamedTemporaryFile(prefix="serving_routes_", suffix=".sqlite", delete=False)
        db_file.close()
        self.db_file = db_file.name
        self.app = Flask(__name__)
        self.app.config.update(
            SQLALCHEMY_DATABASE_URI=f"sqlite:///{self.db_file}",
            SQLALCHEMY_BINDS={"model_gateway": f"sqlite:///{self.db_file}"},
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
            MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN="admin-secret",
            MERGEKIT_MODEL_GATEWAY_SYNC_WAIT_SECONDS=60,
            MERGEKIT_MODEL_GATEWAY_RESEARCH_ROOT=tempfile.mkdtemp(),
            MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND="db",
            MODEL_POOL_PATH=tempfile.mkdtemp(),
            LOCAL_MODELS_PATH=tempfile.mkdtemp(),
            MERGE_DIR=tempfile.mkdtemp(),
            LOCAL_MODELS_EXTRA_PATHS=[],
        )
        db.init_app(self.app)
        register_model_gateway(self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()
        self.client = self.app.test_client()

    def tearDown(self):
        roots = [
            self.app.config["MODEL_POOL_PATH"],
            self.app.config["LOCAL_MODELS_PATH"],
            self.app.config["MERGE_DIR"],
        ]
        self.db.session.remove()
        self.db.drop_all()
        self.ctx.pop()
        try:
            os.unlink(self.db_file)
        except OSError:
            pass
        for root in roots:
            shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(self.app.config["MERGEKIT_MODEL_GATEWAY_RESEARCH_ROOT"], ignore_errors=True)

    def make_model_dir(self):
        path = os.path.join(self.app.config["MERGE_DIR"], "task-a", "output")
        os.makedirs(path)
        for name, body in {
            "config.json": "{}",
            "tokenizer.json": "{}",
            "model.safetensors": "placeholder",
        }.items():
            with open(os.path.join(path, name), "w", encoding="utf-8") as f:
                f.write(body)
        return path

    def admin_headers(self, token="admin-secret"):
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def create_running_service_and_key(self):
        from app.model_gateway.auth import hash_secret
        from app.model_gateway.models import ServingApiKey, ServingModelService

        service = ServingModelService(
            model_path=self.make_model_dir(),
            display_name="Qwen demo",
            served_model_name="qwen-demo",
            status="running",
            vllm_port=18001,
            internal_api_key="internal-secret",
        )
        key = ServingApiKey(
            key_hash=hash_secret("mk_live_usersecret"),
            prefix="mk_live",
            last4="cret",
            owner_label="demo-user",
            model_allowlist=["qwen-demo"],
        )
        self.db.session.add_all([service, key])
        self.db.session.commit()
        return service, key

    def mark_research_file_ready(self, file_id):
        from app.model_gateway.models import ResearchFile

        source = self.db.session.get(ResearchFile, file_id)
        source.status = "ready"
        self.db.session.add(source)
        self.db.session.commit()


class TestAdminRoutes(ServingRoutesTestCase):
    def test_admin_can_disable_or_revoke_api_key(self):
        _, key = self.create_running_service_and_key()

        disabled = self.client.post(
            f"/api/model-gateway/admin/api-keys/{key.id}/disable",
            headers=self.admin_headers(),
        )
        revoked = self.client.post(
            f"/api/model-gateway/admin/api-keys/{key.id}/revoke",
            headers=self.admin_headers(),
        )

        self.assertEqual(disabled.status_code, 200)
        self.assertEqual(disabled.get_json()["api_key"]["status"], "disabled")
        self.assertEqual(revoked.status_code, 200)
        self.assertEqual(revoked.get_json()["api_key"]["status"], "revoked")

    def test_admin_routes_require_configured_token(self):
        self.app.config["MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN"] = ""

        resp = self.client.get("/api/model-gateway/admin/model-services")

        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["error"]["code"], "serving_admin_token_not_configured")

    def test_admin_routes_reject_wrong_token(self):
        resp = self.client.get("/api/model-gateway/admin/model-services", headers=self.admin_headers("wrong"))

        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.get_json()["error"]["code"], "unauthorized")

    def test_admin_can_create_service_and_api_key(self):
        from app.model_gateway.models import ServingApiKey, ServingModelService

        model_path = self.make_model_dir()

        service_resp = self.client.post(
            "/api/model-gateway/admin/model-services",
            headers=self.admin_headers(),
            json={
                "model_path": model_path,
                "display_name": "Qwen demo",
                "served_model_name": "qwen-demo",
                "gpu_ids": [0],
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.85,
            },
        )
        key_resp = self.client.post(
            "/api/model-gateway/admin/api-keys",
            headers=self.admin_headers(),
            json={"owner_label": "demo-user", "model_allowlist": ["qwen-demo"]},
        )

        self.assertEqual(service_resp.status_code, 201)
        self.assertEqual(key_resp.status_code, 201)
        self.assertTrue(key_resp.get_json()["api_key"].startswith("mk_live_"))
        self.assertEqual(self.db.session.query(ServingModelService).count(), 1)
        stored_key = self.db.session.query(ServingApiKey).one()
        self.assertNotEqual(stored_key.key_hash, key_resp.get_json()["api_key"])

    def test_admin_create_service_rejects_invalid_numeric_fields_as_400(self):
        resp = self.client.post(
            "/api/model-gateway/admin/model-services",
            headers=self.admin_headers(),
            json={
                "model_path": self.make_model_dir(),
                "display_name": "Qwen demo",
                "served_model_name": "qwen-demo",
                "gpu_ids": [0],
                "tensor_parallel_size": "not-a-number",
                "gpu_memory_utilization": 0.85,
            },
        )

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"]["code"], "invalid_tensor_parallel_size")

    def test_admin_can_start_and_stop_service(self):
        from app.model_gateway.models import ServingModelService

        service = ServingModelService(
            model_path=self.make_model_dir(),
            display_name="Qwen demo",
            served_model_name="qwen-demo",
            status="stopped",
        )
        self.db.session.add(service)
        self.db.session.commit()

        with patch("app.model_gateway.routes.start_service") as start, patch("app.model_gateway.routes.stop_service") as stop:
            service.status = "running"
            start.return_value = service
            start_resp = self.client.post(
                f"/api/model-gateway/admin/model-services/{service.id}/start",
                headers=self.admin_headers(),
            )
            service.status = "stopped"
            stop.return_value = service
            stop_resp = self.client.post(
                f"/api/model-gateway/admin/model-services/{service.id}/stop",
                headers=self.admin_headers(),
            )

        self.assertEqual(start_resp.status_code, 200)
        self.assertEqual(start_resp.get_json()["service"]["status"], "running")
        self.assertEqual(stop_resp.status_code, 200)
        self.assertEqual(stop_resp.get_json()["service"]["status"], "stopped")


class TestResearchRoutes(ServingRoutesTestCase):
    def test_source_ttl_uses_configured_duration(self):
        from app.model_gateway.models import ResearchFile

        self.create_running_service_and_key()
        self.app.config["MERGEKIT_MODEL_GATEWAY_SOURCE_TTL_HOURS"] = 1
        before = datetime.utcnow()
        response = self.client.post(
            "/api/model-gateway/files",
            headers={"Authorization": "Bearer mk_live_usersecret"},
            data={"file": (BytesIO(b"%PDF-1.4\nresearch"), "report.pdf")},
            content_type="multipart/form-data",
        )
        row = self.db.session.get(ResearchFile, response.get_json()["file"]["id"])

        self.assertEqual(response.status_code, 201)
        self.assertLess(abs((row.expires_at - before - timedelta(hours=1)).total_seconds()), 3)

    def test_hourly_research_limit_remains_after_active_slot_is_canceled(self):
        self.create_running_service_and_key()
        self.app.config["MERGEKIT_MODEL_GATEWAY_MAX_ACTIVE_RESEARCH_JOBS"] = 1
        self.app.config["MERGEKIT_MODEL_GATEWAY_RESEARCH_SUBMISSIONS_PER_HOUR"] = 1
        upload = self.client.post(
            "/api/model-gateway/files",
            headers={"Authorization": "Bearer mk_live_usersecret"},
            data={"file": (BytesIO(b"%PDF-1.4\nresearch"), "report.pdf")},
            content_type="multipart/form-data",
        )
        file_id = upload.get_json()["file"]["id"]
        self.mark_research_file_ready(file_id)
        headers = {"Authorization": "Bearer mk_live_usersecret", "Content-Type": "application/json"}
        first = self.client.post(
            "/api/model-gateway/research/jobs", headers=headers,
            json={"model": "qwen-demo", "task_type": "summary", "file_ids": [file_id], "input": "第一项研究"},
        )
        self.client.post(
            f"/api/model-gateway/research/jobs/{first.get_json()['job']['id']}/cancel",
            headers={"Authorization": "Bearer mk_live_usersecret"},
        )
        blocked = self.client.post(
            "/api/model-gateway/research/jobs", headers=headers,
            json={"model": "qwen-demo", "task_type": "summary", "file_ids": [file_id], "input": "第二项研究"},
        )

        self.assertEqual(first.status_code, 202)
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.get_json()["error"]["code"], "research_submission_limit_exceeded")
        self.assertIn("Retry-After", blocked.headers)

    def test_active_research_limit_rejects_then_cancel_releases_slot(self):
        self.create_running_service_and_key()
        self.app.config["MERGEKIT_MODEL_GATEWAY_MAX_ACTIVE_RESEARCH_JOBS"] = 1
        upload = self.client.post(
            "/api/model-gateway/files",
            headers={"Authorization": "Bearer mk_live_usersecret"},
            data={"file": (BytesIO(b"%PDF-1.4\nresearch"), "report.pdf")},
            content_type="multipart/form-data",
        )
        file_id = upload.get_json()["file"]["id"]
        self.mark_research_file_ready(file_id)
        headers = {"Authorization": "Bearer mk_live_usersecret", "Content-Type": "application/json"}
        first = self.client.post(
            "/api/model-gateway/research/jobs", headers=headers,
            json={"model": "qwen-demo", "task_type": "summary", "file_ids": [file_id], "input": "第一项研究"},
        )
        blocked = self.client.post(
            "/api/model-gateway/research/jobs", headers=headers,
            json={"model": "qwen-demo", "task_type": "summary", "file_ids": [file_id], "input": "第二项研究"},
        )
        canceled = self.client.post(
            f"/api/model-gateway/research/jobs/{first.get_json()['job']['id']}/cancel",
            headers={"Authorization": "Bearer mk_live_usersecret"},
        )
        next_job = self.client.post(
            "/api/model-gateway/research/jobs", headers=headers,
            json={"model": "qwen-demo", "task_type": "summary", "file_ids": [file_id], "input": "第三项研究"},
        )

        self.assertEqual(first.status_code, 202)
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.get_json()["error"]["code"], "research_concurrency_limit_exceeded")
        self.assertEqual(canceled.status_code, 200)
        self.assertEqual(next_job.status_code, 202)

    def test_upload_over_daily_byte_quota_is_rejected_without_record(self):
        from app.model_gateway.models import ResearchFile

        self.create_running_service_and_key()
        self.app.config["MERGEKIT_MODEL_GATEWAY_IMPORT_BYTES_PER_DAY"] = 1
        response = self.client.post(
            "/api/model-gateway/files",
            headers={"Authorization": "Bearer mk_live_usersecret"},
            data={"file": (BytesIO(b"%PDF-1.4\nresearch"), "report.pdf")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.get_json()["error"]["code"], "daily_import_quota_exceeded")
        self.assertEqual(self.db.session.query(ResearchFile).count(), 0)

    @patch("app.model_gateway.routes._complete_non_streaming_request")
    def test_chat_request_rate_limit_returns_retry_after(self, _complete):
        self.create_running_service_and_key()
        self.app.config["MERGEKIT_MODEL_GATEWAY_CHAT_REQUESTS_PER_MINUTE"] = 1
        self.app.config["MERGEKIT_MODEL_GATEWAY_SYNC_WAIT_SECONDS"] = 0
        headers = {"Authorization": "Bearer mk_live_usersecret", "Content-Type": "application/json"}
        payload = {"model": "qwen-demo", "messages": [{"role": "user", "content": "hello"}]}

        first = self.client.post("/v1/chat/completions", headers=headers, json=payload)
        blocked = self.client.post("/v1/chat/completions", headers=headers, json=payload)

        self.assertEqual(first.status_code, 202)
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.get_json()["error"]["code"], "chat_rate_limit_exceeded")
        self.assertIn("Retry-After", blocked.headers)

    def test_invalid_chat_request_id_does_not_consume_rate_quota(self):
        self.create_running_service_and_key()
        self.app.config["MERGEKIT_MODEL_GATEWAY_CHAT_REQUESTS_PER_MINUTE"] = 1
        headers = {
            "Authorization": "Bearer mk_live_usersecret",
            "Content-Type": "application/json",
            "X-Request-Id": "not-a-uuid",
        }
        payload = {"model": "qwen-demo", "messages": [{"role": "user", "content": "hello"}]}

        first = self.client.post("/v1/chat/completions", headers=headers, json=payload)
        second = self.client.post("/v1/chat/completions", headers=headers, json=payload)

        self.assertEqual(first.status_code, 400)
        self.assertEqual(first.get_json()["error"]["code"], "invalid_request_id")
        self.assertEqual(second.status_code, 400)
        self.assertEqual(second.get_json()["error"]["code"], "invalid_request_id")

    def test_uploaded_file_status_is_visible_only_to_its_owner(self):
        self.create_running_service_and_key()
        upload = self.client.post(
            "/api/model-gateway/files",
            headers={"Authorization": "Bearer mk_live_usersecret"},
            data={"file": (BytesIO(b"%PDF-1.4\nresearch"), "report.pdf")},
            content_type="multipart/form-data",
        )
        file_id = upload.get_json()["file"]["id"]

        visible = self.client.get(
            f"/api/model-gateway/files/{file_id}",
            headers={"Authorization": "Bearer mk_live_usersecret"},
        )
        hidden = self.client.get(
            f"/api/model-gateway/files/{file_id}",
            headers={"Authorization": "Bearer mk_live_othersecret"},
        )

        self.assertEqual(visible.status_code, 200)
        self.assertEqual(visible.get_json()["file"]["status"], "received")
        self.assertEqual(hidden.status_code, 401)

    @patch("app.model_gateway.routes._enqueue_research_file")
    def test_redis_upload_is_enqueued_after_file_persisted(self, enqueue):
        self.create_running_service_and_key()
        self.app.config["MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND"] = "redis"

        response = self.client.post(
            "/api/model-gateway/files",
            headers={"Authorization": "Bearer mk_live_usersecret"},
            data={"file": (BytesIO(b"%PDF-1.4\nresearch"), "report.pdf")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 201)
        enqueue.assert_called_once()

    @patch("app.model_gateway.routes._enqueue_research_file", side_effect=RuntimeError("redis down"))
    def test_upload_rolls_back_when_redis_delivery_fails(self, _enqueue):
        from app.model_gateway.models import ResearchFile

        self.create_running_service_and_key()
        self.app.config["MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND"] = "redis"

        response = self.client.post(
            "/api/model-gateway/files",
            headers={"Authorization": "Bearer mk_live_usersecret"},
            data={"file": (BytesIO(b"%PDF-1.4\nresearch"), "report.pdf")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.db.session.query(ResearchFile).count(), 0)

    def test_research_job_rejects_file_that_has_not_completed_processing(self):
        self.create_running_service_and_key()
        upload = self.client.post(
            "/api/model-gateway/files",
            headers={"Authorization": "Bearer mk_live_usersecret"},
            data={"file": (BytesIO(b"%PDF-1.4\nresearch"), "report.pdf")},
            content_type="multipart/form-data",
        )

        response = self.client.post(
            "/api/model-gateway/research/jobs",
            headers={"Authorization": "Bearer mk_live_usersecret", "Content-Type": "application/json"},
            json={"model": "qwen-demo", "task_type": "summary", "file_ids": [upload.get_json()["file"]["id"]], "input": "总结"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error"]["code"], "source_not_ready")

    def test_repeated_idempotency_key_returns_original_job_without_requeue(self):
        self.create_running_service_and_key()
        upload = self.client.post(
            "/api/model-gateway/files",
            headers={"Authorization": "Bearer mk_live_usersecret"},
            data={"file": (BytesIO(b"%PDF-1.4\nresearch"), "report.pdf")},
            content_type="multipart/form-data",
        )
        payload = {
            "model": "qwen-demo",
            "task_type": "summary",
            "file_ids": [upload.get_json()["file"]["id"]],
            "input": "总结",
            "output_format": "markdown",
        }
        self.mark_research_file_ready(payload["file_ids"][0])
        headers = {
            "Authorization": "Bearer mk_live_usersecret",
            "Content-Type": "application/json",
            "Idempotency-Key": "research-retry-1",
        }

        first = self.client.post("/api/model-gateway/research/jobs", headers=headers, json=payload)
        second = self.client.post("/api/model-gateway/research/jobs", headers=headers, json=payload)

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202)
        self.assertEqual(first.get_json()["job"]["id"], second.get_json()["job"]["id"])

    def test_idempotency_key_rejects_different_research_payload(self):
        self.create_running_service_and_key()
        upload = self.client.post(
            "/api/model-gateway/files",
            headers={"Authorization": "Bearer mk_live_usersecret"},
            data={"file": (BytesIO(b"%PDF-1.4\nresearch"), "report.pdf")},
            content_type="multipart/form-data",
        )
        payload = {
            "model": "qwen-demo",
            "task_type": "summary",
            "file_ids": [upload.get_json()["file"]["id"]],
            "input": "总结",
        }
        self.mark_research_file_ready(payload["file_ids"][0])
        headers = {
            "Authorization": "Bearer mk_live_usersecret",
            "Content-Type": "application/json",
            "Idempotency-Key": "research-conflict-1",
        }
        self.assertEqual(self.client.post("/api/model-gateway/research/jobs", headers=headers, json=payload).status_code, 202)
        payload["input"] = "改成提取结构化字段"

        conflict = self.client.post("/api/model-gateway/research/jobs", headers=headers, json=payload)

        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.get_json()["error"]["code"], "idempotency_conflict")

    @patch("app.model_gateway.routes._enqueue_research_job")
    def test_redis_queue_backend_enqueues_only_after_job_persisted(self, enqueue):
        self.create_running_service_and_key()
        upload = self.client.post(
            "/api/model-gateway/files",
            headers={"Authorization": "Bearer mk_live_usersecret"},
            data={"file": (BytesIO(b"%PDF-1.4\nresearch"), "report.pdf")},
            content_type="multipart/form-data",
        )
        self.mark_research_file_ready(upload.get_json()["file"]["id"])
        self.app.config["MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND"] = "redis"
        response = self.client.post(
            "/api/model-gateway/research/jobs",
            headers={"Authorization": "Bearer mk_live_usersecret", "Content-Type": "application/json"},
            json={"model": "qwen-demo", "task_type": "summary", "file_ids": [upload.get_json()["file"]["id"]], "input": "总结", "output_format": "markdown"},
        )

        self.assertEqual(response.status_code, 202)
        enqueue.assert_called_once()

    @patch("app.model_gateway.routes._enqueue_research_file")
    @patch("app.model_gateway.routes.validate_public_source_url")
    def test_user_can_enqueue_public_url_source(self, validate_url, enqueue):
        from urllib.parse import urlparse

        self.create_running_service_and_key()
        self.app.config["MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND"] = "redis"
        validate_url.return_value = urlparse("https://example.test/paper.pdf")

        response = self.client.post(
            "/api/model-gateway/sources/url",
            headers={"Authorization": "Bearer mk_live_usersecret", "Content-Type": "application/json"},
            json={"url": "https://example.test/paper.pdf"},
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json()["file"]["source_kind"], "url")
        self.assertEqual(response.get_json()["file"]["status"], "received")
        enqueue.assert_called_once()

    def test_user_can_upload_pdf_and_create_owned_research_job(self):
        self.create_running_service_and_key()

        upload = self.client.post(
            "/api/model-gateway/files",
            headers={"Authorization": "Bearer mk_live_usersecret"},
            data={"file": (BytesIO(b"%PDF-1.4\nresearch"), "report.pdf")},
            content_type="multipart/form-data",
        )

        self.assertEqual(upload.status_code, 201)
        file_id = upload.get_json()["file"]["id"]
        self.mark_research_file_ready(file_id)
        created = self.client.post(
            "/api/model-gateway/research/jobs",
            headers={"Authorization": "Bearer mk_live_usersecret", "Content-Type": "application/json"},
            json={
                "model": "qwen-demo",
                "task_type": "document_qa",
                "file_ids": [file_id],
                "input": "总结这份资料",
                "output_format": "markdown",
                "require_citations": True,
            },
        )

        self.assertEqual(created.status_code, 202)
        body = created.get_json()
        self.assertEqual(body["job"]["status"], "queued")
        self.assertNotIn("总结这份资料", str(body))

        job_id = body["job"]["id"]
        status = self.client.get(
            f"/api/model-gateway/research/jobs/{job_id}",
            headers={"Authorization": "Bearer mk_live_usersecret"},
        )
        canceled = self.client.post(
            f"/api/model-gateway/research/jobs/{job_id}/cancel",
            headers={"Authorization": "Bearer mk_live_usersecret"},
        )

        self.assertEqual(status.status_code, 200)
        self.assertEqual(canceled.status_code, 200)
        self.assertEqual(canceled.get_json()["job"]["status"], "canceled")

    def test_completed_research_job_exposes_answer_and_source_locators_only(self):
        from app.model_gateway.models import ResearchFile, ResearchJob

        self.create_running_service_and_key()
        upload = self.client.post(
            "/api/model-gateway/files",
            headers={"Authorization": "Bearer mk_live_usersecret"},
            data={"file": (BytesIO(b"%PDF-1.4\nresearch"), "report.pdf")},
            content_type="multipart/form-data",
        )
        file_id = upload.get_json()["file"]["id"]
        self.mark_research_file_ready(file_id)
        source = self.db.session.get(ResearchFile, file_id)
        source.source_url = "https://papers.example.org/verified-study"
        self.db.session.commit()
        created = self.client.post(
            "/api/model-gateway/research/jobs",
            headers={"Authorization": "Bearer mk_live_usersecret", "Content-Type": "application/json"},
            json={"model": "qwen-demo", "task_type": "document_qa", "file_ids": [file_id], "input": "结论是什么？"},
        )
        job = self.db.session.get(ResearchJob, created.get_json()["job"]["id"])
        result_path = os.path.join(self.app.config["MERGEKIT_MODEL_GATEWAY_RESEARCH_ROOT"], "result.json")
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump({
                "answer": "效率提升为 42%。[S1]",
                "citations": [1],
                "evidence": [{"file_id": file_id, "locator": {"kind": "page", "value": 2}, "text": "private source text"}],
            }, handle)
        job.status = "completed"
        job.result_path = result_path
        self.db.session.add(job)
        self.db.session.commit()

        response = self.client.get(
            f"/api/model-gateway/research/jobs/{job.id}",
            headers={"Authorization": "Bearer mk_live_usersecret"},
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()["job"]["result"]
        self.assertEqual(result["answer"], "效率提升为 42%。[S1]")
        self.assertEqual(result["citations"], [1])
        self.assertEqual(result["sources"], [{
            "file_id": file_id,
            "locator": {"kind": "page", "value": 2},
            "url": "https://papers.example.org/verified-study",
        }])
        self.assertNotIn("private source text", str(response.get_json()))


class TestOpenAiCompatibleRoutes(ServingRoutesTestCase):
    def test_v1_models_lists_running_allowed_models(self):
        self.create_running_service_and_key()

        resp = self.client.get("/v1/models", headers={"Authorization": "Bearer mk_live_usersecret"})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["object"], "list")
        self.assertEqual(resp.get_json()["data"][0]["id"], "qwen-demo")

    def test_chat_completions_proxies_to_vllm_and_records_usage(self):
        from app.model_gateway.models import ServingRequest, ServingUsageRecord

        self.create_running_service_and_key()

        with patch("app.model_gateway.routes.requests.post", return_value=FakeVllmResponse()) as post:
            resp = self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer mk_live_usersecret", "Content-Type": "application/json"},
                json={"model": "qwen-demo", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 16},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["usage"]["total_tokens"], 7)
        req = self.db.session.query(ServingRequest).one()
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer internal-secret")
        self.assertEqual(post.call_args.kwargs["headers"]["X-Request-Id"], req.id)
        self.assertEqual(resp.headers["X-Request-Id"], req.id)
        self.assertEqual(req.status, "success")
        self.assertEqual(self.db.session.query(ServingUsageRecord).one().total_tokens, 7)

    def test_client_request_id_is_persisted_and_forwarded_to_vllm(self):
        from app.model_gateway.models import ServingRequest

        self.create_running_service_and_key()
        request_id = str(uuid.uuid4())

        with patch("app.model_gateway.routes.requests.post", return_value=FakeVllmResponse()) as post:
            resp = self.client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer mk_live_usersecret",
                    "Content-Type": "application/json",
                    "X-Request-Id": request_id,
                },
                json={"model": "qwen-demo", "messages": [{"role": "user", "content": "ping"}]},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["X-Request-Id"], request_id)
        self.assertIsNotNone(self.db.session.get(ServingRequest, request_id))
        self.assertEqual(post.call_args.kwargs["headers"]["X-Request-Id"], request_id)

    def test_non_streaming_timeout_returns_202_and_background_completion(self):
        from app.model_gateway.models import ServingRequest, ServingUsageRecord

        self.app.config["MERGEKIT_MODEL_GATEWAY_SYNC_WAIT_SECONDS"] = 0.01
        self.create_running_service_and_key()

        def slow_post(*args, **kwargs):
            time.sleep(0.05)
            return FakeVllmResponse()

        with patch("app.model_gateway.routes.requests.post", side_effect=slow_post):
            resp = self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer mk_live_usersecret", "Content-Type": "application/json"},
                json={"model": "qwen-demo", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 16},
            )

        self.assertEqual(resp.status_code, 202)
        body = resp.get_json()
        self.assertEqual(body["object"], "serving.request")
        self.assertEqual(body["status"], "running")
        request_id = body["request_id"]

        deadline = time.time() + 2
        req = None
        while time.time() < deadline:
            self.db.session.remove()
            req = self.db.session.get(ServingRequest, request_id)
            if req and req.status == "success":
                break
            time.sleep(0.02)

        self.assertIsNotNone(req)
        self.assertEqual(req.status, "success")
        self.assertEqual(self.db.session.query(ServingUsageRecord).filter_by(request_id=request_id).one().total_tokens, 7)

    def test_streaming_upstream_error_marks_request_failed(self):
        from app.model_gateway.models import ServingRequest

        self.create_running_service_and_key()

        with patch("app.model_gateway.routes.requests.post", side_effect=RuntimeError("upstream down")):
            resp = self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer mk_live_usersecret", "Content-Type": "application/json"},
                json={"model": "qwen-demo", "messages": [{"role": "user", "content": "ping"}], "stream": True},
            )

        self.assertEqual(resp.status_code, 502)
        req = self.db.session.query(ServingRequest).one()
        self.assertEqual(req.status, "failed")
        self.assertIn("upstream down", req.error_message)

    def test_streaming_success_records_stream_usage(self):
        from app.model_gateway.models import ServingRequest, ServingUsageRecord

        self.create_running_service_and_key()

        with patch("app.model_gateway.routes.requests.post", return_value=FakeStreamingResponse()):
            resp = self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer mk_live_usersecret", "Content-Type": "application/json"},
                json={"model": "qwen-demo", "messages": [{"role": "user", "content": "ping"}], "stream": True},
            )
            body = b"".join(resp.response)

        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"data: hello", body)
        self.assertEqual(resp.headers["X-Request-Id"], self.db.session.query(ServingRequest).one().id)
        self.assertEqual(self.db.session.query(ServingRequest).one().status, "success")
        self.assertEqual(self.db.session.query(ServingUsageRecord).one().usage_source, "stream_usage_unavailable")

    def test_api_key_can_cancel_own_queued_request(self):
        from app.model_gateway.models import ServingRequest

        service, key = self.create_running_service_and_key()
        req = ServingRequest(
            api_key_id=key.id,
            model_service_id=service.id,
            served_model_name="qwen-demo",
            status="queued",
        )
        self.db.session.add(req)
        self.db.session.commit()

        resp = self.client.post(
            f"/v1/requests/{req.id}/cancel",
            headers={"Authorization": "Bearer mk_live_usersecret"},
        )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["request"]["status"], "canceled")
        self.assertEqual(self.db.session.get(ServingRequest, req.id).status, "canceled")

    def test_api_key_can_read_own_request_status_with_usage(self):
        from app.model_gateway.models import ServingRequest, ServingUsageRecord

        service, key = self.create_running_service_and_key()
        req = ServingRequest(
            api_key_id=key.id,
            model_service_id=service.id,
            served_model_name="qwen-demo",
            status="success",
        )
        self.db.session.add(req)
        self.db.session.flush()
        self.db.session.add(ServingUsageRecord(
            request_id=req.id,
            api_key_id=key.id,
            model_service_id=service.id,
            served_model_name="qwen-demo",
            prompt_tokens=3,
            completion_tokens=4,
            total_tokens=7,
            usage_source="vllm_response",
        ))
        self.db.session.commit()

        resp = self.client.get(
            f"/v1/requests/{req.id}",
            headers={"Authorization": "Bearer mk_live_usersecret"},
        )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["request"]["id"], req.id)
        self.assertEqual(body["request"]["status"], "success")
        self.assertEqual(body["usage"]["total_tokens"], 7)
        self.assertEqual(body["usage"]["usage_source"], "vllm_response")

    def test_api_key_cannot_read_other_users_request_status(self):
        from app.model_gateway.auth import hash_secret
        from app.model_gateway.models import ServingApiKey, ServingRequest

        service, key = self.create_running_service_and_key()
        other_key = ServingApiKey(
            key_hash=hash_secret("mk_live_othersecret"),
            prefix="mk_live",
            last4="cret",
            owner_label="other-user",
            model_allowlist=["qwen-demo"],
        )
        req = ServingRequest(
            api_key_id=key.id,
            model_service_id=service.id,
            served_model_name="qwen-demo",
            status="queued",
        )
        self.db.session.add_all([other_key, req])
        self.db.session.commit()

        resp = self.client.get(
            f"/v1/requests/{req.id}",
            headers={"Authorization": "Bearer mk_live_othersecret"},
        )

        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["error"]["code"], "request_not_found")

    def test_cancel_running_request_aborts_vllm_and_marks_canceled(self):
        from app.model_gateway.models import ServingRequest

        service, key = self.create_running_service_and_key()
        req = ServingRequest(
            api_key_id=key.id,
            model_service_id=service.id,
            served_model_name="qwen-demo",
            status="running",
        )
        self.db.session.add(req)
        self.db.session.commit()

        with patch("app.model_gateway.routes.requests.post", return_value=FakeVllmResponse()) as abort:
            resp = self.client.post(
                f"/v1/requests/{req.id}/cancel",
                headers={"Authorization": "Bearer mk_live_usersecret"},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["request"]["status"], "canceled")
        self.assertEqual(abort.call_args.args[0], "http://127.0.0.1:18001/internal/model-gateway/abort/chatcmpl-" + req.id)
        self.assertEqual(abort.call_args.kwargs["headers"]["Authorization"], "Bearer internal-secret")

    def test_api_key_cannot_cancel_other_users_request(self):
        from app.model_gateway.auth import hash_secret
        from app.model_gateway.models import ServingApiKey, ServingRequest

        service, key = self.create_running_service_and_key()
        other_key = ServingApiKey(
            key_hash=hash_secret("mk_live_othersecret"),
            prefix="mk_live",
            last4="cret",
            owner_label="other-user",
            model_allowlist=["qwen-demo"],
        )
        req = ServingRequest(
            api_key_id=key.id,
            model_service_id=service.id,
            served_model_name="qwen-demo",
            status="queued",
        )
        self.db.session.add_all([other_key, req])
        self.db.session.commit()

        resp = self.client.post(
            f"/v1/requests/{req.id}/cancel",
            headers={"Authorization": "Bearer mk_live_othersecret"},
        )

        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["error"]["code"], "request_not_found")
        self.assertEqual(self.db.session.get(ServingRequest, req.id).status, "queued")

    def test_cancel_finished_request_is_rejected(self):
        from app.model_gateway.models import ServingRequest

        service, key = self.create_running_service_and_key()
        req = ServingRequest(
            api_key_id=key.id,
            model_service_id=service.id,
            served_model_name="qwen-demo",
            status="success",
        )
        self.db.session.add(req)
        self.db.session.commit()

        resp = self.client.post(
            f"/v1/requests/{req.id}/cancel",
            headers={"Authorization": "Bearer mk_live_usersecret"},
        )

        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.get_json()["error"]["code"], "request_not_cancelable")


if __name__ == "__main__":
    unittest.main()
