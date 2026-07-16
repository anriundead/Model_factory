"""Gateway binding tests for formal published assets."""
from contextlib import contextmanager
import json
import os
import shutil
import tempfile
import threading
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

    def test_publishable_models_uses_quick_inventory_without_full_hash(self):
        from app.model_gateway.routes import _formal_published_asset

        with mock.patch("app.model_gateway.routes._formal_published_asset", wraps=_formal_published_asset) as validate:
            response = self.client.get("/api/model-gateway/admin/publishable-models", headers=self.admin_headers())

        self.assertEqual(response.status_code, 200)
        self.assertTrue(validate.call_count)
        self.assertTrue(all(call.kwargs.get("full_hash") is False for call in validate.call_args_list))

    def test_publishable_models_safely_summarizes_changed_runtime_version_as_stale(self):
        manifest_path = os.path.join(self.published_text.path, "publication_manifest.json")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["compatibility"]["serving"]["tested_version"] = "0.0.0"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

        response = self.client.get("/api/model-gateway/admin/publishable-models", headers=self.admin_headers())

        row = next(item for item in response.get_json()["models"] if item["model_id"] == self.published_text.id)
        self.assertFalse(row["selectable"])
        self.assertEqual(row["blocked_reason_code"], "version_changed")


class TestPublishedServiceLifecycle(PublicationGatewayTestCase):
    def _payload(self, **overrides):
        payload = {
            "model_id": self.published_text.id,
            "display_name": "Published text",
            "served_model_name": "published-text",
            "gpu_ids": [0],
        }
        payload.update(overrides)
        return payload

    def _create(self, **overrides):
        return self.client.post(
            "/api/model-gateway/admin/model-services",
            headers=self.admin_headers(),
            json=self._payload(**overrides),
        )

    def test_create_resolves_path_from_formal_model(self):
        response = self._create()

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["service"]["model_path"], self.published_text.path)

    def test_create_uses_full_hash_validation(self):
        from app.model_gateway.routes import _formal_published_asset

        with mock.patch("app.model_gateway.routes._formal_published_asset", wraps=_formal_published_asset) as validate:
            response = self._create()

        self.assertEqual(response.status_code, 201)
        validate.assert_called_once()
        self.assertIs(validate.call_args.kwargs.get("full_hash"), True)

    def test_create_rejects_asset_validated_with_a_different_runtime_version(self):
        from app.model_gateway.models import ServingModelService

        manifest_path = os.path.join(self.published_text.path, "publication_manifest.json")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["compatibility"]["serving"]["tested_version"] = "0.0.0"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

        response = self._create()

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error"]["code"], "version_changed")
        self.assertEqual(self.db.session.query(ServingModelService).count(), 0)

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
        from app.model_publication import PublicationError

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
            with self.assertRaises(PublicationError) as raised:
                with mock.patch("app.model_gateway.runtime.validate_model_path"):
                    start_service(service.id, config=Config, timeout_s=0)

        self.assertEqual(raised.exception.code, "unsupported_architecture")
        reserve.assert_not_called()
        popen.assert_not_called()

    def test_formal_service_with_missing_core_row_fails_closed_before_gpu_or_process(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import start_service
        from app.model_publication import PublicationError

        service = ServingModelService(
            model_id="missing-formal-model",
            model_path=self.published_text.path,
            display_name="Missing formal model",
            served_model_name="missing-formal-model",
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

        class FakeProc:
            pid = 12345

            def poll(self):
                return None

        with mock.patch("app.model_gateway.runtime.validate_gpu_availability") as reserve, \
            mock.patch("app.model_gateway.runtime.subprocess.Popen", return_value=FakeProc()) as popen, \
            mock.patch("app.model_gateway.runtime.os.getpgid", return_value=12345), \
            mock.patch("app.model_gateway.runtime._healthcheck", return_value=True), \
            self.assertRaises(PublicationError) as raised:
            start_service(service.id, config=Config, timeout_s=1)

        self.assertEqual(raised.exception.code, "asset_unavailable")
        reserve.assert_not_called()
        popen.assert_not_called()

    def test_formal_service_source_or_path_mismatch_fails_closed(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import _validate_formal_service_asset

        cases = (("base", self.published_text.path, "asset_unavailable"),
                 ("published", self.published_vlm.path, "asset_identity_mismatch"))
        for index, (source, path, expected_code) in enumerate(cases):
            with self.subTest(expected_code=expected_code):
                self.published_text.source = source
                self.db.session.commit()
                service = ServingModelService(
                    model_id=self.published_text.id,
                    model_path=path,
                    display_name="Mismatched formal model",
                    served_model_name="mismatched-formal-%s" % index,
                    gpu_ids=[0],
                )
                self.db.session.add(service)
                self.db.session.commit()

                raised = None
                try:
                    _validate_formal_service_asset(service, type("Config", (), {
                        "PUBLISHED_MODELS_PATH": self.published,
                        "MODEL_POOL_PATH": self.app.config["MODEL_POOL_PATH"],
                        "LOCAL_MODELS_PATH": self.app.config["LOCAL_MODELS_PATH"],
                        "MERGE_DIR": self.app.config["MERGE_DIR"],
                        "LOCAL_MODELS_EXTRA_PATHS": [],
                    }))
                except Exception as exc:
                    raised = exc

                self.assertIsNotNone(raised)
                self.assertEqual(getattr(raised, "code", None), expected_code)
                self.db.session.delete(service)
                self.published_text.source = "published"
                self.db.session.commit()

    def test_path_under_publication_root_without_model_id_is_not_legacy(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import start_service
        from app.model_publication import PublicationError

        service = ServingModelService(
            model_path=self.published_text.path,
            display_name="Unbound formal model",
            served_model_name="unbound-formal-model",
            gpu_ids=[0],
        )
        self.db.session.add(service)
        self.db.session.commit()

        class FakeProc:
            pid = 12345

            def poll(self):
                return None

        with mock.patch("app.model_gateway.runtime.validate_gpu_availability") as reserve, \
            mock.patch("app.model_gateway.runtime.subprocess.Popen", return_value=FakeProc()) as popen, \
            mock.patch("app.model_gateway.runtime.os.getpgid", return_value=12345), \
            mock.patch("app.model_gateway.runtime._healthcheck", return_value=True), \
            self.assertRaises(PublicationError) as raised:
            start_service(service.id, config=type("Config", (), {
                "PUBLISHED_MODELS_PATH": self.published,
                "MODEL_POOL_PATH": self.app.config["MODEL_POOL_PATH"],
                "LOCAL_MODELS_PATH": self.app.config["LOCAL_MODELS_PATH"],
                "MERGE_DIR": self.app.config["MERGE_DIR"],
                "LOCAL_MODELS_EXTRA_PATHS": [],
            }), timeout_s=1)

        self.assertEqual(raised.exception.code, "asset_unavailable")
        reserve.assert_not_called()
        popen.assert_not_called()

    def test_formal_start_detects_corrupt_hash_stale_status_and_version_change(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import start_service
        from app.model_publication import PublicationError

        manifest_path = os.path.join(self.published_text.path, "publication_manifest.json")
        weight_path = os.path.join(self.published_text.path, "model.safetensors")
        with open(manifest_path, encoding="utf-8") as handle:
            original_manifest = json.load(handle)
        with open(weight_path, "rb") as handle:
            original_weight = handle.read()

        cases = (
            ("corrupt-hash", "validation_failed"),
            ("stale-status", "version_changed"),
            ("changed-version", "version_changed"),
        )
        for name, expected_code in cases:
            with self.subTest(name=name):
                manifest = json.loads(json.dumps(original_manifest))
                with open(weight_path, "wb") as handle:
                    handle.write(original_weight)
                if name == "corrupt-hash":
                    with open(weight_path, "wb") as handle:
                        handle.write(b"other")
                elif name == "stale-status":
                    manifest["compatibility"]["serving"].update({
                        "status": "stale",
                        "reason_code": "version_changed",
                    })
                else:
                    manifest["compatibility"]["serving"]["tested_version"] = "0.0.0"
                with open(manifest_path, "w", encoding="utf-8") as handle:
                    json.dump(manifest, handle)
                service = ServingModelService(
                    model_id=self.published_text.id,
                    model_path=self.published_text.path,
                    display_name=name,
                    served_model_name=name,
                    gpu_ids=[0],
                )
                self.db.session.add(service)
                self.db.session.commit()

                with mock.patch("app.model_gateway.runtime.validate_gpu_availability") as reserve, \
                    mock.patch("app.model_gateway.runtime.subprocess.Popen") as popen, \
                    mock.patch.dict("sys.modules", {"vllm": mock.Mock(__version__="0.7.0")}), \
                    self.assertRaises(PublicationError) as raised:
                    start_service(service.id, config=type("Config", (), {
                        "PUBLISHED_MODELS_PATH": self.published,
                        "MODEL_POOL_PATH": self.app.config["MODEL_POOL_PATH"],
                        "LOCAL_MODELS_PATH": self.app.config["LOCAL_MODELS_PATH"],
                        "MERGE_DIR": self.app.config["MERGE_DIR"],
                        "LOCAL_MODELS_EXTRA_PATHS": [],
                    }), timeout_s=0)

                self.assertEqual(raised.exception.code, expected_code)
                reserve.assert_not_called()
                popen.assert_not_called()
                self.db.session.delete(service)
                self.db.session.commit()

        with open(weight_path, "wb") as handle:
            handle.write(original_weight)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(original_manifest, handle)

    def test_deleted_service_cannot_start_or_mutate_runtime_state(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import start_service

        service = ServingModelService(
            model_id=self.published_text.id,
            model_path=self.published_text.path,
            display_name="Deleted model",
            served_model_name="deleted-model-start",
            status="deleted",
            gpu_ids=[0],
        )
        self.db.session.add(service)
        self.db.session.commit()

        with mock.patch("app.model_gateway.runtime._validate_formal_service_asset") as asset_check, \
            mock.patch("app.model_gateway.runtime.validate_gpu_availability") as reserve, \
            mock.patch("app.model_gateway.runtime.subprocess.Popen") as popen, \
            self.assertRaisesRegex(ValueError, "deleted"):
            start_service(service.id, config=object(), timeout_s=0)

        self.db.session.refresh(service)
        self.assertEqual(service.status, "deleted")
        asset_check.assert_not_called()
        reserve.assert_not_called()
        popen.assert_not_called()

    def test_deleted_service_cannot_be_stopped_or_resurrected(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import stop_service

        service = ServingModelService(
            model_path=self.published_text.path,
            display_name="Deleted model",
            served_model_name="deleted-model-stop",
            status="deleted",
            vllm_pid=4321,
            vllm_pgid=4321,
        )
        self.db.session.add(service)
        self.db.session.commit()

        with mock.patch("app.model_gateway.runtime._find_marked_service_pids") as find_pids, \
            mock.patch("app.model_gateway.runtime._terminate_service_processes") as terminate, \
            self.assertRaisesRegex(ValueError, "deleted"):
            stop_service(service.id)

        self.db.session.refresh(service)
        self.assertEqual(service.status, "deleted")
        self.assertEqual(service.vllm_pid, 4321)
        find_pids.assert_not_called()
        terminate.assert_not_called()


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
        model_id = self.published_text.id
        model_path = self.published_text.path
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
        self.assertFalse(os.path.exists(model_path))
        self.assertIsNone(self.db.session.get(self.Model, model_id))
        self.assertIsNotNone(self.db.session.get(ServingModelService, service.id))
        self.assertIsNotNone(self.db.session.get(ServingUsageRecord, usage.id))

    def test_task_read_failure_remains_fail_closed_as_asset_in_use(self):
        from sqlalchemy import event

        from app.model_publication import PublicationError, delete_registered_published_asset

        core_engine = self.db.engine
        model_path = self.published_text.path

        def fail_task_read(_conn, _cursor, statement, _parameters, _context, _many):
            if "FROM tasks" in statement:
                raise RuntimeError("task read failed")

        event.listen(core_engine, "before_cursor_execute", fail_task_read)
        try:
            with self.assertRaises(PublicationError) as raised:
                delete_registered_published_asset("published-text", self.published)
        finally:
            event.remove(core_engine, "before_cursor_execute", fail_task_read)

        self.assertEqual(raised.exception.code, "asset_in_use")
        self.assertTrue(os.path.isdir(model_path))


class TestServiceLifecycleRace(PublicationGatewayTestCase):
    def _service(self, name, status="stopped"):
        from app.model_gateway.models import ServingModelService

        service = ServingModelService(
            model_id=self.published_text.id,
            model_path=self.published_text.path,
            display_name=name,
            served_model_name=name,
            status=status,
            gpu_ids=[0],
            vllm_port=18123,
        )
        self.db.session.add(service)
        self.db.session.commit()
        return service.id

    def test_start_claim_wins_delete_loses_and_prelaunch_failure_marks_failed(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import start_service

        service_id = self._service("start-wins")
        validation_entered = threading.Event()
        release_validation = threading.Event()
        start_result = {}

        def fail_validation(_service, _config):
            validation_entered.set()
            self.assertTrue(release_validation.wait(5))
            raise RuntimeError("validation exploded")

        def start():
            with self.app.app_context():
                try:
                    start_service(service_id, config=object(), timeout_s=0)
                except Exception as exc:
                    start_result["error"] = exc

        with mock.patch("app.model_gateway.runtime._validate_formal_service_asset", side_effect=fail_validation), \
            mock.patch("app.model_gateway.runtime.validate_gpu_availability") as gpu, \
            mock.patch("app.model_gateway.runtime.subprocess.Popen") as popen:
            starter = threading.Thread(target=start)
            starter.start()
            self.assertTrue(validation_entered.wait(5))

            deleted = self.app.test_client().delete(
                "/api/model-gateway/admin/model-services/%s" % service_id,
                headers=self.admin_headers(),
            )
            release_validation.set()
            starter.join(5)

        self.assertFalse(starter.is_alive())
        self.assertEqual(deleted.status_code, 409)
        self.assertEqual(deleted.get_json()["error"]["code"], "service_not_stopped")
        self.assertIsInstance(start_result.get("error"), RuntimeError)
        self.db.session.expire_all()
        self.assertEqual(self.db.session.get(ServingModelService, service_id).status, "failed")
        gpu.assert_not_called()
        popen.assert_not_called()

    def test_delete_claim_wins_start_returns_deleted_without_launch(self):
        from sqlalchemy import event

        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import ServiceStateError, start_service

        service_id = self._service("delete-wins")
        delete_update_entered = threading.Event()
        release_delete = threading.Event()
        start_attempted = threading.Event()
        start_done = threading.Event()
        delete_result = {}
        start_result = {}
        gateway_engine = self.db.engines["model_gateway"]

        class FakeProc:
            pid = 12345

            def poll(self):
                return None

        def block_delete_update(_conn, _cursor, statement, parameters, _context, _many):
            if statement.lstrip().upper().startswith("UPDATE SERVING_MODEL_SERVICES") and "deleted" in parameters:
                delete_update_entered.set()
                self.assertTrue(release_delete.wait(5))

        def delete():
            response = self.app.test_client().delete(
                "/api/model-gateway/admin/model-services/%s" % service_id,
                headers=self.admin_headers(),
            )
            delete_result.update(status=response.status_code, body=response.get_json())

        def start():
            with self.app.app_context():
                start_attempted.set()
                try:
                    start_service(service_id, config=object(), timeout_s=0)
                except ServiceStateError as exc:
                    start_result["code"] = exc.code
                finally:
                    start_done.set()

        event.listen(gateway_engine, "after_cursor_execute", block_delete_update)
        try:
            with mock.patch("app.model_gateway.runtime._validate_formal_service_asset"), \
                mock.patch("app.model_gateway.runtime.validate_model_path"), \
                mock.patch("app.model_gateway.runtime.validate_gpu_availability"), \
                mock.patch("app.model_gateway.runtime.subprocess.Popen", return_value=FakeProc()) as popen, \
                mock.patch("app.model_gateway.runtime.os.getpgid", return_value=12345), \
                mock.patch("app.model_gateway.runtime._terminate_process_group"):
                deleter = threading.Thread(target=delete)
                deleter.start()
                self.assertTrue(delete_update_entered.wait(5))
                starter = threading.Thread(target=start)
                starter.start()
                self.assertTrue(start_attempted.wait(5))
                self.assertFalse(start_done.wait(0.2))
                release_delete.set()
                deleter.join(5)
                starter.join(5)
        finally:
            release_delete.set()
            event.remove(gateway_engine, "after_cursor_execute", block_delete_update)

        self.assertFalse(deleter.is_alive())
        self.assertFalse(starter.is_alive())
        self.assertEqual(delete_result.get("status"), 200)
        self.assertEqual(delete_result.get("body", {}).get("service", {}).get("status"), "deleted")
        self.assertEqual(start_result.get("code"), "service_deleted")
        popen.assert_not_called()
        self.db.session.expire_all()
        self.assertEqual(self.db.session.get(ServingModelService, service_id).status, "deleted")

    def test_stop_rejects_unlaunched_starting_service(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import ServiceStateError, stop_service

        service_id = self._service("starting-stop", status="starting")

        with mock.patch("app.model_gateway.runtime._find_marked_service_pids") as find_pids, \
            self.assertRaises(ServiceStateError) as raised:
            stop_service(service_id)

        self.assertEqual(raised.exception.code, "service_starting")
        find_pids.assert_not_called()
        self.db.session.expire_all()
        self.assertEqual(self.db.session.get(ServingModelService, service_id).status, "starting")

    def test_start_is_idempotent_only_for_running_and_rejects_inflight_states(self):
        from app.model_gateway.runtime import ServiceStateError, start_service

        running_id = self._service("already-running", status="running")
        starting_id = self._service("already-starting", status="starting")
        stopping_id = self._service("already-stopping", status="stopping")

        with mock.patch("app.model_gateway.runtime._validate_formal_service_asset") as validate, \
            mock.patch("app.model_gateway.runtime.validate_gpu_availability") as gpu, \
            mock.patch("app.model_gateway.runtime.subprocess.Popen") as popen:
            self.assertEqual(start_service(running_id, config=object()).status, "running")
            for service_id in (starting_id, stopping_id):
                with self.subTest(service_id=service_id), self.assertRaises(ServiceStateError) as raised:
                    start_service(service_id, config=object())
                self.assertEqual(raised.exception.code, "service_state_conflict")

        validate.assert_not_called()
        gpu.assert_not_called()
        popen.assert_not_called()


class TestServiceCreateDeleteRace(PublicationGatewayTestCase):
    def _create_in_thread(self, result, model_id, **overrides):
        payload = {
            "model_id": model_id,
            "display_name": "Published text",
            "served_model_name": "race-published-text",
            "gpu_ids": [0],
        }
        payload.update(overrides)
        response = self.app.test_client().post(
            "/api/model-gateway/admin/model-services",
            headers=self.admin_headers(),
            json=payload,
        )
        result.update(status=response.status_code, body=response.get_json())

    def test_creation_wins_and_deletion_observes_asset_in_use(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.routes import _formal_published_asset
        from app.model_publication import PublicationError, delete_registered_published_asset, publication_lock

        validation_entered = threading.Event()
        release_creation = threading.Event()
        deletion_attempted = threading.Event()
        deletion_done = threading.Event()
        creation_result = {}
        deletion_result = {}
        model_id = self.published_text.id
        model_path = self.published_text.path

        def blocking_validate(model, full_hash=False):
            validation_entered.set()
            self.assertTrue(release_creation.wait(5))
            return _formal_published_asset(model, full_hash=full_hash)

        @contextmanager
        def observed_lock(root):
            if threading.current_thread().name == "delete-asset":
                deletion_attempted.set()
            with publication_lock(root) as locked_root:
                yield locked_root

        def delete_asset():
            with self.app.app_context():
                try:
                    delete_registered_published_asset("published-text", self.published)
                except PublicationError as exc:
                    deletion_result["code"] = exc.code
                finally:
                    deletion_done.set()

        with mock.patch("app.model_gateway.routes._formal_published_asset", side_effect=blocking_validate), \
            mock.patch("app.model_publication.publication_lock", side_effect=observed_lock):
            creator = threading.Thread(
                name="create-service",
                target=self._create_in_thread,
                args=(creation_result, model_id),
            )
            creator.start()
            self.assertTrue(validation_entered.wait(5))
            deleter = threading.Thread(name="delete-asset", target=delete_asset)
            deleter.start()
            self.assertTrue(deletion_attempted.wait(5))
            self.assertFalse(deletion_done.wait(0.2))
            release_creation.set()
            creator.join(5)
            deleter.join(5)

        self.assertFalse(creator.is_alive())
        self.assertFalse(deleter.is_alive())
        self.assertEqual(creation_result.get("status"), 201)
        self.assertEqual(deletion_result.get("code"), "asset_in_use")
        self.db.session.expire_all()
        self.assertEqual(self.db.session.query(ServingModelService).count(), 1)
        self.assertTrue(os.path.isdir(model_path))

    def test_deletion_wins_and_creation_conflicts_without_orphan_row(self):
        from app.model_gateway.models import ServingModelService
        from app.model_publication import (
            _delete_core_model_by_canonical_path,
            delete_registered_published_asset,
            publication_lock,
        )

        core_delete_entered = threading.Event()
        release_deletion = threading.Event()
        creation_attempted = threading.Event()
        creation_done = threading.Event()
        creation_result = {}
        deletion_result = {}
        model_id = self.published_text.id
        model_path = self.published_text.path

        def blocking_core_delete(path):
            core_delete_entered.set()
            self.assertTrue(release_deletion.wait(5))
            return _delete_core_model_by_canonical_path(path)

        @contextmanager
        def observed_lock(root):
            if threading.current_thread().name == "create-service":
                creation_attempted.set()
            with publication_lock(root) as locked_root:
                yield locked_root

        def delete_asset():
            with self.app.app_context():
                deletion_result.update(delete_registered_published_asset("published-text", self.published))

        def create_service():
            try:
                self._create_in_thread(creation_result, model_id)
            finally:
                creation_done.set()

        with mock.patch("app.model_publication._delete_core_model_by_canonical_path", side_effect=blocking_core_delete), \
            mock.patch("app.model_publication.publication_lock", side_effect=observed_lock):
            deleter = threading.Thread(name="delete-asset", target=delete_asset)
            deleter.start()
            self.assertTrue(core_delete_entered.wait(5))
            creator = threading.Thread(
                name="create-service",
                target=create_service,
            )
            creator.start()
            self.assertTrue(creation_attempted.wait(5))
            self.assertFalse(creation_done.wait(0.2))
            release_deletion.set()
            deleter.join(5)
            creator.join(5)

        self.assertFalse(deleter.is_alive())
        self.assertFalse(creator.is_alive())
        self.assertTrue(deletion_result.get("deleted"))
        self.assertEqual(creation_result.get("status"), 409)
        self.assertEqual(creation_result.get("body", {}).get("error", {}).get("code"), "asset_unavailable")
        self.db.session.expire_all()
        self.assertEqual(self.db.session.query(ServingModelService).count(), 0)
        self.assertIsNone(self.db.session.get(self.Model, model_id))
        self.assertFalse(os.path.exists(model_path))


class TestPublicationBindIsolation(PublicationGatewayTestCase):
    def _create_service(self, served_model_name):
        return self.app.test_client().post(
            "/api/model-gateway/admin/model-services",
            headers=self.admin_headers(),
            json={
                "model_id": self.published_text.id,
                "display_name": "Published text",
                "served_model_name": served_model_name,
                "gpu_ids": [0],
            },
        )

    def test_creation_never_commits_the_core_read_transaction(self):
        from sqlalchemy import event

        from app.model_gateway.models import ServingModelService

        core_engine = self.db.engine
        attempted = []

        def reject_core_commit(_conn):
            attempted.append(True)
            raise RuntimeError("core read transaction was committed")

        event.listen(core_engine, "commit", reject_core_commit)
        caught = None
        response = None
        try:
            response = self._create_service("isolated-create")
        except RuntimeError as exc:
            caught = exc
        finally:
            event.remove(core_engine, "commit", reject_core_commit)
            self.db.session.rollback()

        self.assertIsNone(caught)
        self.assertEqual(attempted, [])
        self.assertEqual(response.status_code, 201)
        self.db.session.expire_all()
        self.assertEqual(
            self.db.session.query(ServingModelService).filter_by(served_model_name="isolated-create").count(),
            1,
        )

    def test_failed_gateway_commit_leaves_no_partial_service_row(self):
        from sqlalchemy import event

        from app.model_gateway.models import ServingModelService

        gateway_engine = self.db.engines["model_gateway"]

        def fail_gateway_commit(_conn):
            raise RuntimeError("gateway commit failed")

        event.listen(gateway_engine, "commit", fail_gateway_commit)
        with self.assertRaisesRegex(RuntimeError, "gateway commit failed"):
            try:
                self._create_service("failed-gateway-commit")
            finally:
                event.remove(gateway_engine, "commit", fail_gateway_commit)
                self.db.session.rollback()

        self.db.session.expire_all()
        self.assertEqual(
            self.db.session.query(ServingModelService).filter_by(served_model_name="failed-gateway-commit").count(),
            0,
        )

    def test_core_read_transaction_failure_prevents_gateway_insert(self):
        from sqlalchemy import event

        from app.model_gateway.models import ServingModelService

        core_engine = self.db.engine

        def fail_core_rollback(_conn):
            raise RuntimeError("core read rollback failed")

        event.listen(core_engine, "rollback", fail_core_rollback)
        with self.assertRaisesRegex(RuntimeError, "core read rollback failed"):
            try:
                self._create_service("core-read-failed")
            finally:
                event.remove(core_engine, "rollback", fail_core_rollback)
                self.db.session.remove()

        self.assertEqual(
            self.db.session.query(ServingModelService).filter_by(served_model_name="core-read-failed").count(),
            0,
        )

    def test_start_never_commits_the_core_validation_transaction(self):
        from sqlalchemy import event

        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import start_service

        service_id = self._create_service("isolated-start").get_json()["service"]["id"]
        self.db.session.remove()
        core_engine = self.db.engine
        attempted = []

        class FakeProc:
            pid = 12345
            returncode = None

            def poll(self):
                return None

        def reject_core_commit(_conn):
            attempted.append(True)
            raise RuntimeError("core validation transaction was committed")

        config = type("Config", (), {
            "PUBLISHED_MODELS_PATH": self.published,
            "MODEL_POOL_PATH": self.app.config["MODEL_POOL_PATH"],
            "LOCAL_MODELS_PATH": self.app.config["LOCAL_MODELS_PATH"],
            "MERGE_DIR": self.app.config["MERGE_DIR"],
            "LOCAL_MODELS_EXTRA_PATHS": [],
            "PROJECT_ROOT": self.tmp,
            "MERGEKIT_MODEL_GATEWAY_LOG_DIR": os.path.join(self.tmp, "logs"),
        })
        event.listen(core_engine, "commit", reject_core_commit)
        caught = None
        result = None
        try:
            with mock.patch("app.model_publication.package_version", return_value="0.7.0"), \
                mock.patch("app.model_gateway.runtime.validate_gpu_availability"), \
                mock.patch("app.model_gateway.runtime.subprocess.Popen", return_value=FakeProc()), \
                mock.patch("app.model_gateway.runtime.os.getpgid", return_value=12345), \
                mock.patch("app.model_gateway.runtime._healthcheck", return_value=True):
                result = start_service(service_id, config=config, timeout_s=1)
        except RuntimeError as exc:
            caught = exc
        finally:
            event.remove(core_engine, "commit", reject_core_commit)
            self.db.session.rollback()

        self.assertIsNone(caught)
        self.assertEqual(attempted, [])
        self.assertEqual(result.status, "running")
        self.db.session.expire_all()
        self.assertEqual(self.db.session.get(ServingModelService, service_id).status, "running")

    def test_deletion_never_commits_the_gateway_read_transaction(self):
        from sqlalchemy import event

        from app.model_publication import delete_registered_published_asset

        gateway_engine = self.db.engines["model_gateway"]
        attempted = []
        model_path = self.published_text.path

        def reject_gateway_commit(_conn):
            attempted.append(True)
            raise RuntimeError("gateway read transaction was committed")

        event.listen(gateway_engine, "commit", reject_gateway_commit)
        caught = None
        result = None
        try:
            result = delete_registered_published_asset("published-text", self.published)
        except RuntimeError as exc:
            caught = exc
        finally:
            event.remove(gateway_engine, "commit", reject_gateway_commit)
            self.db.session.rollback()

        self.assertIsNone(caught)
        self.assertEqual(attempted, [])
        self.assertTrue(result["deleted"])
        self.assertFalse(os.path.exists(model_path))

    def test_failed_core_delete_commit_restores_asset_without_gateway_commit(self):
        from sqlalchemy import event

        from app.model_publication import delete_registered_published_asset

        core_engine = self.db.engine
        gateway_engine = self.db.engines["model_gateway"]
        gateway_commits = []
        model_id = self.published_text.id
        model_path = self.published_text.path

        def fail_core_commit(_conn):
            raise RuntimeError("core delete commit failed")

        def observe_gateway_commit(_conn):
            gateway_commits.append(True)

        event.listen(core_engine, "commit", fail_core_commit)
        event.listen(gateway_engine, "commit", observe_gateway_commit)
        with self.assertRaisesRegex(RuntimeError, "core delete commit failed"):
            try:
                delete_registered_published_asset("published-text", self.published)
            finally:
                event.remove(core_engine, "commit", fail_core_commit)
                event.remove(gateway_engine, "commit", observe_gateway_commit)
                self.db.session.remove()

        self.assertEqual(gateway_commits, [])
        self.assertTrue(os.path.isdir(model_path))
        self.assertFalse(os.path.exists(os.path.join(self.published, ".trash", "published-text")))
        self.assertIsNotNone(self.db.session.get(self.Model, model_id))


if __name__ == "__main__":
    unittest.main()
