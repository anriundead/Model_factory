"""Gateway binding tests for formal published assets."""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class PublicationGatewayTestCase(unittest.TestCase):
    def setUp(self):
        from flask import Flask

        from app import models as base_models  # noqa: F401
        from app.extensions import db
        from app.model_gateway import models as gateway_models  # noqa: F401
        from app.model_gateway import register_model_gateway
        from app.models import Model

        self.tmp = tempfile.mkdtemp(prefix="publication_gateway_")
        self.published = os.path.join(self.tmp, "published")
        self.core_db = os.path.join(self.tmp, "core.sqlite")
        self.gateway_db = os.path.join(self.tmp, "gateway.sqlite")
        self.app = Flask(__name__)
        self.app.config.update(
            TESTING=True,
            SQLALCHEMY_DATABASE_URI="sqlite:///%s" % self.core_db,
            SQLALCHEMY_BINDS={"model_gateway": "sqlite:///%s" % self.gateway_db},
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
            MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN="admin-secret",
            MODEL_POOL_PATH=os.path.join(self.tmp, "models"),
            LOCAL_MODELS_PATH=os.path.join(self.tmp, "local"),
            MERGE_DIR=os.path.join(self.tmp, "merges"),
            LOCAL_MODELS_EXTRA_PATHS=[],
            PUBLISHED_MODELS_PATH=self.published,
        )
        for root in (self.app.config["MODEL_POOL_PATH"], self.app.config["LOCAL_MODELS_PATH"], self.app.config["MERGE_DIR"]):
            os.makedirs(root)
        db.init_app(self.app)
        register_model_gateway(self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()
        self.db = db
        self.Model = Model
        self.client = self.app.test_client()
        self.published_text = self._published_model("published-text", "ready")
        self.published_vlm = self._published_model("published-vlm", "blocked")

    def tearDown(self):
        self.db.session.remove()
        self.db.drop_all()
        self.ctx.pop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def admin_headers(self):
        return {"Authorization": "Bearer admin-secret", "Content-Type": "application/json"}

    def _published_model(self, publication_id, status):
        from app.model_inspection import inspect_model
        from app.model_publication import build_manifest, commit_staging

        staging = os.path.join(self.published, ".staging", publication_id)
        os.makedirs(staging)
        with open(os.path.join(staging, "config.json"), "w", encoding="utf-8") as handle:
            json.dump({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}, handle)
        for name in ("model.safetensors", "tokenizer.json"):
            with open(os.path.join(staging, name), "wb") as handle:
                handle.write(b"asset")
        with open(os.path.join(staging, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
            json.dump({"weight_map": {"model.weight": "model.safetensors"}}, handle)
        manifest = build_manifest(
            staging,
            {"publication_id": publication_id, "display_name": publication_id, "task_id": "task-%s" % publication_id},
            inspect_model(staging),
            {"structural": {"status": "passed"}},
            {"serving": {
                "backend": "vllm",
                "tested_version": "0.7.0",
                "status": status,
                "reason_code": "unsupported_architecture" if status == "blocked" else None,
            }},
        )
        committed = commit_staging(staging, self.published, manifest, lambda _path, _manifest: None)
        model = self.Model(
            path=os.path.join(self.published, committed["publication_id"]),
            name=committed["display_name"],
            source="published",
            architecture=committed["model"]["model_type"],
            is_vlm=committed["artifact_type"] == "vlm",
        )
        self.db.session.add(model)
        self.db.session.commit()
        return model


class TestPublishableModels(PublicationGatewayTestCase):
    def test_admin_sees_ready_and_blocked_published_models(self):
        response = self.client.get("/api/model-gateway/admin/publishable-models", headers=self.admin_headers())

        self.assertEqual(response.status_code, 200)
        rows = {row["model_id"]: row for row in response.get_json()["models"]}
        self.assertTrue(rows[self.published_text.id]["selectable"])
        self.assertFalse(rows[self.published_vlm.id]["selectable"])
        self.assertEqual(rows[self.published_vlm.id]["blocked_reason_code"], "unsupported_architecture")

    def test_publishable_models_requires_admin_token(self):
        response = self.client.get("/api/model-gateway/admin/publishable-models")

        self.assertEqual(response.status_code, 401)


class TestPublishedServiceLifecycle(PublicationGatewayTestCase):
    def _create(self, **overrides):
        payload = {
            "model_id": self.published_text.id,
            "display_name": "Published text",
            "served_model_name": "published-text",
            "gpu_ids": [0],
        }
        payload.update(overrides)
        return self.client.post("/api/model-gateway/admin/model-services", headers=self.admin_headers(), json=payload)

    def test_create_resolves_path_from_formal_model(self):
        response = self._create()

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["service"]["model_path"], self.published_text.path)

    def test_create_rejects_client_model_path_and_non_selectable_asset(self):
        path = self._create(model_path="/tmp/not-a-formal-asset")
        blocked = self._create(model_id=self.published_vlm.id, served_model_name="published-vlm")

        self.assertEqual(path.status_code, 400)
        self.assertEqual(blocked.status_code, 409)

    def test_running_service_cannot_be_deleted_and_name_is_never_reused(self):
        from app.model_gateway.models import ServingModelService

        created = self._create().get_json()["service"]
        service = self.db.session.get(ServingModelService, created["id"])
        service.status = "running"
        self.db.session.commit()
        active = self.client.delete("/api/model-gateway/admin/model-services/%s" % service.id, headers=self.admin_headers())
        service.status = "stopped"
        self.db.session.commit()
        deleted = self.client.delete("/api/model-gateway/admin/model-services/%s" % service.id, headers=self.admin_headers())
        reused = self._create()

        self.assertEqual(active.status_code, 409)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.get_json()["service"]["status"], "deleted")
        self.assertEqual(reused.status_code, 409)

    def test_formal_blocked_asset_never_reaches_gpu_or_process_launch(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import start_service

        service = ServingModelService(
            model_id=self.published_vlm.id,
            model_path=self.published_vlm.path,
            display_name="Blocked VLM",
            served_model_name="blocked-vlm",
            gpu_ids=[0],
        )
        self.db.session.add(service)
        self.db.session.commit()
        class Config:
            PUBLISHED_MODELS_PATH = self.published
            MODEL_POOL_PATH = self.app.config["MODEL_POOL_PATH"]
            LOCAL_MODELS_PATH = self.app.config["LOCAL_MODELS_PATH"]
            MERGE_DIR = self.app.config["MERGE_DIR"]
            LOCAL_MODELS_EXTRA_PATHS = []

        with mock.patch("app.model_gateway.runtime.validate_gpu_availability") as reserve, mock.patch("app.model_gateway.runtime.subprocess.Popen") as popen:
            with self.assertRaises(ValueError):
                with mock.patch("app.model_gateway.runtime.validate_model_path"):
                    start_service(service.id, config=Config, timeout_s=0)

        reserve.assert_not_called()
        popen.assert_not_called()


class TestPublishedAssetDeleteGuard(PublicationGatewayTestCase):
    def test_soft_deleted_service_releases_asset_without_removing_history(self):
        from app.model_gateway.models import ServingModelService, ServingUsageRecord
        from app.model_publication import PublicationError, delete_registered_published_asset

        service = ServingModelService(
            model_id=self.published_text.id,
            model_path=self.published_text.path,
            display_name="Published text",
            served_model_name="delete-guard-text",
            status="stopped",
        )
        self.db.session.add(service)
        self.db.session.commit()
        usage = ServingUsageRecord(
            model_service_id=service.id,
            served_model_name=service.served_model_name,
            usage_source="test",
        )
        self.db.session.add(usage)
        self.db.session.commit()

        with self.assertRaisesRegex(PublicationError, "asset_in_use"):
            delete_registered_published_asset("published-text", self.published)

        service.status = "deleted"
        self.db.session.commit()
        result = delete_registered_published_asset("published-text", self.published)

        self.assertTrue(result["deleted"])
        self.assertFalse(os.path.exists(self.published_text.path))
        self.assertIsNone(self.db.session.get(self.Model, self.published_text.id))
        self.assertIsNotNone(self.db.session.get(ServingModelService, service.id))
        self.assertIsNotNone(self.db.session.get(ServingUsageRecord, usage.id))


if __name__ == "__main__":
    unittest.main()
