import json
import os
import queue
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

from flask import Flask


class PublicationRouteTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.recipes = os.path.join(self.tmpdir.name, "recipes")
        self.published = os.path.join(self.tmpdir.name, "published")
        self.model_path = os.path.join(self.tmpdir.name, "core")
        os.makedirs(self.recipes)
        os.makedirs(self.model_path)
        with open(os.path.join(self.recipes, "valid.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        self.state = SimpleNamespace(
            recipes_dir=self.recipes,
            merge_dir=os.path.join(self.tmpdir.name, "merges"),
            tasks={},
            task_queue=queue.PriorityQueue(),
            scheduler_lock=threading.Lock(),
            running_task_info={"id": None, "priority": None, "process": None},
            priority_map={"common": 10},
            logger=mock.Mock(),
            model_pool_path=self.model_path,
            project_root=self.tmpdir.name,
            config=SimpleNamespace(PUBLISHED_MODELS_PATH=self.published),
        )
        app = Flask(__name__)
        app.config.update(
            TESTING=True,
            MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN="admin-token",
            PUBLISHED_MODELS_PATH=self.published,
            SQLALCHEMY_DATABASE_URI="sqlite:///%s" % os.path.join(self.tmpdir.name, "routes.db"),
            SQLALCHEMY_BINDS={"model_gateway": "sqlite:///%s" % os.path.join(self.tmpdir.name, "gateway.db")},
        )
        from app.extensions import db
        from app.models import Model, Task
        from app.routes import register_routes
        from app.services import Services

        db.init_app(app)
        with app.app_context():
            db.create_all()
            db.session.add(Model(id="model-1", name="core", path=self.model_path, source="base"))
            db.session.commit()
        self.services = Services(self.state)
        self.services.app = app
        self.services.logger = self.state.logger
        register_routes(app, self.state, self.services, SimpleNamespace())
        self.app = app
        self.db = db
        self.Model = Model
        self.Task = Task

        import merge_manager
        self.old_recipes_dir = merge_manager.RECIPES_DIR
        merge_manager.RECIPES_DIR = self.recipes

    def tearDown(self):
        import merge_manager
        merge_manager.RECIPES_DIR = self.old_recipes_dir
        with self.app.app_context():
            self.db.session.remove()
        self.tmpdir.cleanup()

    @property
    def headers(self):
        return {"Authorization": "Bearer admin-token"}

    def _post(self, path, payload, **headers):
        return self.app.test_client().post(path, json=payload, headers=headers)

    def _create(self, key="key-1", payload=None):
        headers = {**self.headers, "Idempotency-Key": key}
        return self._post(
            "/api/model-publications",
            payload or {"source_type": "recipe", "recipe_path": "valid.json", "display_name": "published"},
            **headers,
        )

    def _add_publication_task(self, task_id, status, config):
        with self.app.app_context():
            task = self.Task(id=task_id, task_type="model_publication", status=status, config=config)
            self.db.session.add(task)
            self.db.session.commit()

    def _published_model(self, publication_id):
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
            {"serving": {"backend": "vllm", "tested_version": "0.7.0", "status": "ready"}},
        )
        committed = commit_staging(staging, self.published, manifest, lambda _path, _manifest: None)
        with self.app.app_context():
            model = self.Model(
                path=os.path.join(self.published, committed["publication_id"]),
                name=committed["display_name"],
                source="published",
                architecture=committed["model"]["model_type"],
                is_vlm=False,
            )
            self.db.session.add(model)
            self.db.session.commit()
            result = SimpleNamespace(id=model.id, path=model.path, name=model.name)
        return result

    def _gateway_service(self, model, status="stopped"):
        from app.model_gateway.models import ServingModelService

        with self.app.app_context():
            service = ServingModelService(
                model_id=model.id,
                model_path=model.path,
                display_name=model.name,
                served_model_name="served-%s" % model.id,
                status=status,
            )
            self.db.session.add(service)
            self.db.session.commit()
            return SimpleNamespace(id=service.id)

    def _make_legacy_model_dir(self, name):
        path = os.path.join(self.model_path, name)
        os.makedirs(path)
        for filename in ("config.json", "tokenizer.json", "model.safetensors"):
            with open(os.path.join(path, filename), "wb") as handle:
                handle.write(b"legacy")
        return path

    def test_admin_and_idempotency_contract(self):
        payload = {"source_type": "recipe", "recipe_path": "valid.json", "display_name": "published"}
        response = self._post("/api/model-publications", payload)
        self.assertIn(response.status_code, (401, 503))
        self.assertEqual(self._post("/api/model-publications", payload, **self.headers).status_code, 400)

        first = self._create(payload=payload)
        second = self._create(payload=payload)
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.get_json()["task"]["id"], second.get_json()["task"]["id"])
        conflict = self._create(payload={**payload, "display_name": "different"})
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.get_json()["error"]["code"], "idempotency_conflict")

    def test_concurrent_same_key_is_one_db_task_and_one_queue_item(self):
        from app.repositories import publication_task_by_idempotency_key as lookup

        def slow_lookup(key):
            result = lookup(key)
            if result is None:
                time.sleep(0.1)
            return result

        def submit():
            return self._create(key="concurrent-key")

        with mock.patch("app.repositories.publication_task_by_idempotency_key", side_effect=slow_lookup):
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(pool.map(lambda _index: submit(), range(2)))

        self.assertEqual(sorted(response.status_code for response in responses), [200, 202])
        with self.app.app_context():
            self.assertEqual(self.Task.query.filter_by(task_type="model_publication").count(), 1)
        self.assertEqual(self.state.task_queue.qsize(), 1)

    def test_concurrent_same_key_different_payload_conflicts(self):
        from app.repositories import publication_task_by_idempotency_key as lookup

        def slow_lookup(key):
            result = lookup(key)
            if result is None:
                time.sleep(0.1)
            return result

        payloads = [
            {"source_type": "recipe", "recipe_path": "valid.json", "display_name": "one"},
            {"source_type": "recipe", "recipe_path": "valid.json", "display_name": "two"},
        ]
        with mock.patch("app.repositories.publication_task_by_idempotency_key", side_effect=slow_lookup):
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(pool.map(lambda payload: self._create("conflict-key", payload), payloads))

        self.assertEqual(sorted(response.status_code for response in responses), [202, 409])
        self.assertEqual(self.state.task_queue.qsize(), 1)

    def test_nested_and_absolute_recipes_are_rejected_explicitly(self):
        os.makedirs(os.path.join(self.recipes, "nested"))
        with open(os.path.join(self.recipes, "nested", "valid.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        absolute = self._create("absolute", {
            "source_type": "recipe",
            "recipe_path": os.path.join(self.recipes, "valid.json"),
            "display_name": "published",
        })
        nested = self._create("nested", {
            "source_type": "recipe",
            "recipe_path": "nested/valid.json",
            "display_name": "published",
        })
        self.assertEqual(absolute.status_code, 400)
        self.assertEqual(nested.status_code, 400)
        self.assertEqual(nested.get_json()["error"]["code"], "invalid_recipe")

    def test_waiting_validation_cancel_persists_and_removes_staging_immediately(self):
        task_id = "waiting-task"
        publication_id = "waiting-publication"
        staging = os.path.join(self.published, ".staging", publication_id)
        os.makedirs(staging)
        config = {"publication_id": publication_id, "publication_root": self.published, "staging_path": staging}
        self._add_publication_task(task_id, "validating", config)
        self.state.tasks[task_id] = {"status": "validating", "control": {"aborted": False}}

        response = self._post("/api/model-publications/%s/cancel" % task_id, {}, **self.headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["task"]["status"], "canceled")
        self.assertFalse(os.path.exists(staging))

    def test_generic_stop_route_rejects_publication_without_canceling_it(self):
        task_id = "generic-stop-publication"
        publication_id = "generic-stop-publication"
        staging = os.path.join(self.published, ".staging", publication_id)
        os.makedirs(staging)
        config = {"publication_id": publication_id, "publication_root": self.published, "staging_path": staging}
        self._add_publication_task(task_id, "validating", config)
        process = mock.Mock(pid=1234)
        control = {"aborted": False, "process": process}
        self.state.tasks[task_id] = {
            "status": "running",
            "type": "model_publication",
            "original_data": {"type": "model_publication"},
            "control": control,
        }
        self.state.running_task_info["id"] = task_id
        self.state.running_task_info["process"] = process

        response = self.app.test_client().post("/api/stop/%s" % task_id)

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error"]["code"], "publication_cancel_required")
        self.assertFalse(control["aborted"])
        process.terminate.assert_not_called()
        self.assertTrue(os.path.isdir(staging))
        with self.app.app_context():
            self.assertEqual(self.db.session.get(self.Task, task_id).status, "validating")

    def test_generic_stop_all_fails_atomically_when_publication_is_active(self):
        publication_id = "publication-active"
        staging = os.path.join(self.published, ".staging", publication_id)
        os.makedirs(staging)
        self._add_publication_task(
            "publication-task",
            "validating",
            {"publication_id": publication_id, "publication_root": self.published, "staging_path": staging},
        )
        publication_process = mock.Mock(pid=111)
        merge_process = mock.Mock(pid=222)
        self.state.tasks["publication-task"] = {
            "status": "running",
            "type": "model_publication",
            "original_data": {"type": "model_publication"},
            "control": {"aborted": False, "process": publication_process},
        }
        self.state.tasks["merge-task"] = {
            "status": "queued",
            "type": "merge",
            "original_data": {"type": "merge"},
            "control": {"aborted": False, "process": merge_process},
        }
        self.state.task_queue.put((10, 1.0, "publication-task", {"type": "model_publication"}))
        self.state.task_queue.put((10, 2.0, "merge-task", {"type": "merge"}))
        self.state.running_task_info["id"] = "publication-task"
        self.state.running_task_info["process"] = publication_process

        response = self.app.test_client().post("/api/stop_all")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error"]["code"], "publication_cancel_required")
        self.assertEqual(self.state.task_queue.qsize(), 2)
        self.assertFalse(self.state.tasks["publication-task"]["control"]["aborted"])
        self.assertFalse(self.state.tasks["merge-task"]["control"]["aborted"])
        publication_process.terminate.assert_not_called()
        merge_process.terminate.assert_not_called()
        self.assertEqual(self.state.tasks["publication-task"]["status"], "running")
        self.assertEqual(self.state.tasks["merge-task"]["status"], "queued")

    def test_generic_stop_rechecks_publication_target_after_waiting_for_scheduler_lock(self):
        task_id = "race-target"
        ordinary_process = mock.Mock(pid=222)
        publication_process = mock.Mock(pid=111)
        self.state.tasks[task_id] = {
            "status": "queued",
            "type": "merge",
            "original_data": {"type": "merge"},
            "control": {"aborted": False, "process": ordinary_process},
        }
        self.services.kill_process_tree_by_pid = mock.Mock()
        result = {}
        lock = self.state.scheduler_lock
        guard_checked = threading.Event()
        original_guard = self.services._publication_stop_denied

        def observe_guard(*args, **kwargs):
            denied = original_guard(*args, **kwargs)
            guard_checked.set()
            return denied

        self.services._publication_stop_denied = observe_guard

        lock.acquire()
        try:
            thread = threading.Thread(
                target=lambda: result.update(self.services.stop_task_with_cleanup(task_id)),
            )
            thread.start()
            guard_checked.wait(0.2)
            self.state.tasks[task_id] = {
                "status": "running",
                "type": "model_publication",
                "original_data": {"type": "model_publication"},
                "control": {"aborted": False, "process": publication_process},
            }
        finally:
            lock.release()
        thread.join(5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result["error_code"], "publication_cancel_required")
        self.assertEqual(self.state.tasks[task_id]["status"], "running")
        self.assertFalse(self.state.tasks[task_id]["control"]["aborted"])
        self.services.kill_process_tree_by_pid.assert_not_called()

    def test_generic_stop_all_blocks_on_in_memory_publication_when_db_guard_fails(self):
        merge_process = mock.Mock(pid=222)
        publication_process = mock.Mock(pid=111)
        self.state.tasks["merge-task"] = {
            "status": "queued",
            "type": "merge",
            "original_data": {"type": "merge"},
            "control": {"aborted": False, "process": merge_process},
        }
        self.state.task_queue.put((10, 1.0, "merge-task", {"type": "merge"}))
        self.state.running_task_info.update({"id": "merge-task", "process": merge_process})
        self.services.kill_process_tree_by_pid = mock.Mock()
        result = {}
        lock = self.state.scheduler_lock
        guard_checked = threading.Event()
        original_guard = self.services._publication_stop_denied

        def observe_guard(*args, **kwargs):
            denied = original_guard(*args, **kwargs)
            guard_checked.set()
            return denied

        self.services._publication_stop_denied = observe_guard
        db_guard = mock.patch(
            "app.repositories.active_publication_tasks",
            side_effect=RuntimeError("db unavailable"),
        )
        active_publication_tasks = db_guard.start()

        lock.acquire()
        try:
            thread = threading.Thread(target=lambda: result.update(self.services.stop_all_active_tasks()))
            thread.start()
            guard_checked.wait(0.2)
            self.state.tasks["publication-task"] = {
                "status": "running",
                "type": "model_publication",
                "original_data": {"type": "model_publication"},
                "control": {"aborted": False, "process": publication_process},
            }
        finally:
            lock.release()
        thread.join(5)
        db_guard.stop()

        self.assertFalse(thread.is_alive())
        self.assertEqual(result["error_code"], "publication_cancel_required")
        self.assertEqual(self.state.task_queue.qsize(), 1)
        self.assertEqual(self.state.tasks["merge-task"]["status"], "queued")
        self.assertFalse(self.state.tasks["merge-task"]["control"]["aborted"])
        self.assertEqual(self.state.tasks["publication-task"]["status"], "running")
        self.assertFalse(self.state.tasks["publication-task"]["control"]["aborted"])
        active_publication_tasks.assert_not_called()
        self.services.kill_process_tree_by_pid.assert_not_called()

    def test_generic_stop_all_fails_closed_when_publication_guard_lookup_raises(self):
        merge_process = mock.Mock(pid=222)
        self.state.tasks["merge-task"] = {
            "status": "running",
            "type": "merge",
            "original_data": {"type": "merge"},
            "control": {"aborted": False, "process": merge_process},
        }
        self.state.task_queue.put((10, 1.0, "merge-task", {"type": "merge"}))
        self.state.running_task_info.update({"id": "merge-task", "process": merge_process})
        self.services.kill_process_tree_by_pid = mock.Mock()

        with mock.patch(
            "app.repositories.active_publication_tasks",
            side_effect=RuntimeError("db unavailable"),
        ) as active_publication_tasks:
            response = self.app.test_client().post("/api/stop_all")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error"]["code"], "publication_cancel_required")
        self.assertEqual(self.state.task_queue.qsize(), 1)
        self.assertEqual(self.state.tasks["merge-task"]["status"], "running")
        self.assertFalse(self.state.tasks["merge-task"]["control"]["aborted"])
        self.services.kill_process_tree_by_pid.assert_not_called()
        active_publication_tasks.assert_called_once_with()

    def test_generic_stop_fails_closed_when_publication_guard_lookup_raises(self):
        task_id = "merge-db-lookup-error"
        process = mock.Mock(pid=222)
        self.state.tasks[task_id] = {
            "status": "running",
            "type": "merge",
            "original_data": {"type": "merge"},
            "control": {"aborted": False, "process": process},
        }
        self.services.kill_process_tree_by_pid = mock.Mock()

        with mock.patch.object(self.db.session, "get", side_effect=RuntimeError("db unavailable")):
            response = self.app.test_client().post("/api/stop/%s" % task_id)

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error"]["code"], "publication_cancel_required")
        self.assertEqual(self.state.tasks[task_id]["status"], "running")
        self.assertFalse(self.state.tasks[task_id]["control"]["aborted"])
        self.services.kill_process_tree_by_pid.assert_not_called()

    def test_validate_is_single_flight_and_marks_the_durable_enqueue(self):
        task_id = "single-flight"
        self._add_publication_task(task_id, "validating", {"publication_id": "single-flight", "display_name": "published"})

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(
                lambda _index: self._post(
                    "/api/model-publications/%s/validate" % task_id,
                    {"gpu_ids": [0]},
                    **self.headers,
                ),
                range(2),
            ))

        self.assertEqual(sorted(response.status_code for response in responses), [202, 409])
        self.assertEqual(self.state.task_queue.qsize(), 1)
        self.assertEqual(
            next(response for response in responses if response.status_code == 409).get_json()["error"]["code"],
            "validation_in_progress",
        )
        self.assertTrue(self.state.tasks[task_id]["validation_enqueued"])
        with self.app.app_context():
            self.assertTrue(self.db.session.get(self.Task, task_id).config["validation_enqueued"])

    def test_recovery_clears_stale_validation_enqueue_and_allows_validate_again(self):
        from app.services import Services

        task_id = "stale-validation-enqueue"
        self._add_publication_task(task_id, "validating", {
            "publication_id": "stale-validation-enqueue",
            "display_name": "published",
            "validation_enqueued": False,
        })

        first = self._post(
            "/api/model-publications/%s/validate" % task_id,
            {"gpu_ids": [0]},
            **self.headers,
        )
        self.assertEqual(first.status_code, 202)
        with self.app.app_context():
            self.assertTrue(self.db.session.get(self.Task, task_id).config["validation_enqueued"])

        self.state.task_queue = queue.PriorityQueue()
        self.state.tasks = {}
        self.state.running_task_info = {"id": None, "priority": None, "process": None}
        services = Services(self.state)
        services.app = self.app
        services.logger = mock.Mock()

        services.recover_publication_tasks_on_startup()

        with self.app.app_context():
            recovered = self.db.session.get(self.Task, task_id)
            self.assertEqual(recovered.status, "validating")
            self.assertFalse(recovered.config["validation_enqueued"])

        second = self._post(
            "/api/model-publications/%s/validate" % task_id,
            {"gpu_ids": [0]},
            **self.headers,
        )
        self.assertEqual(second.status_code, 202)
        self.assertEqual(self.state.task_queue.qsize(), 1)

    def test_history_collection_redacts_publication_db_rows(self):
        secret = os.path.join(self.tmpdir.name, "secret-publication.json")
        self._add_publication_task("history-db-publication", "validating", {
            "publication_id": "history-db-publication",
            "display_name": "Published",
            "recipe_path": secret,
            "publication_root": self.published,
            "staging_path": secret,
            "request_fingerprint": "fingerprint",
        })

        response = self.app.test_client().get("/api/history")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()["history"]
        text = json.dumps(payload, sort_keys=True)
        self.assertNotIn(secret, text)
        item = next(entry for entry in payload if entry["id"] == "history-db-publication")
        self.assertEqual(item["type"], "model_publication")
        self.assertNotIn("config", item)
        self.assertEqual(item["publication_id"], "history-db-publication")
        self.assertEqual(item["display_name"], "Published")

    def test_history_collection_redacts_publication_disk_metadata_fallback(self):
        task_id = "history-disk-publication"
        secret = os.path.join(self.tmpdir.name, "secret-publication.json")
        task_dir = os.path.join(self.state.merge_dir, task_id)
        os.makedirs(task_dir)
        with open(os.path.join(task_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump({
                "id": task_id,
                "type": "model_publication",
                "publication_id": "history-disk-publication",
                "display_name": "Published",
                "recipe_path": secret,
                "publication_root": self.published,
                "staging_path": secret,
                "error": "failed at %s" % secret,
                "status": "error",
            }, handle)

        response = self.app.test_client().get("/api/history")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()["history"]
        text = json.dumps(payload, sort_keys=True)
        self.assertNotIn(secret, text)
        item = next(entry for entry in payload if entry["id"] == task_id)
        self.assertEqual(item["type"], "model_publication")
        self.assertNotIn("config", item)
        self.assertNotIn("error", item)
        self.assertEqual(item["publication_id"], "history-disk-publication")

    def test_publication_disk_fallbacks_redact_history_and_status_paths(self):
        task_id = "restart-publication"
        secret = os.path.join(self.tmpdir.name, "publication-recipe.json")
        self._add_publication_task(task_id, "validating", {
            "publication_id": "restart-publication",
            "display_name": "Published",
            "error_code": "validation_failed",
        })
        metadata_dir = os.path.join(self.state.merge_dir, task_id)
        os.makedirs(metadata_dir)
        with open(os.path.join(metadata_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump({"type": "model_publication", "recipe_path": secret, "error": "failed at %s" % secret}, handle)
        self.services.status_from_disk = lambda _task_id: {
            "status": "error",
            "message": "failed at %s" % secret,
            "result": {"staging_path": secret},
        }

        history = self.app.test_client().get("/api/history/%s" % task_id)
        status = self.app.test_client().get("/api/status/%s" % task_id)

        self.assertEqual(history.status_code, 200)
        self.assertEqual(status.status_code, 200)
        self.assertNotIn(secret, json.dumps(history.get_json(), sort_keys=True))
        self.assertNotIn(secret, json.dumps(status.get_json(), sort_keys=True))
        self.assertEqual(status.get_json()["status"], "validating")

    def test_publication_disk_redaction_survives_db_lookup_failure(self):
        task_id = "restart-publication-db-failure"
        secret = os.path.join(self.tmpdir.name, "publication-recipe.json")
        metadata_dir = os.path.join(self.state.merge_dir, task_id)
        os.makedirs(metadata_dir)
        with open(os.path.join(metadata_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump({
                "type": "model_publication",
                "publication_id": "restart-publication",
                "display_name": "Published",
                "error_code": "validation_failed",
                "recipe_path": secret,
            }, handle)
        self.services.status_from_disk = lambda _task_id: {
            "status": "error",
            "message": "failed at %s" % secret,
            "result": {"staging_path": secret},
        }

        with mock.patch.object(self.db.session, "get", side_effect=RuntimeError("db unavailable")):
            history = self.app.test_client().get("/api/history/%s" % task_id)
            status = self.app.test_client().get("/api/status/%s" % task_id)

        self.assertEqual(history.status_code, 200)
        self.assertEqual(status.status_code, 200)
        self.assertNotIn(secret, json.dumps(history.get_json(), sort_keys=True))
        self.assertNotIn(secret, json.dumps(status.get_json(), sort_keys=True))
        self.assertEqual(status.get_json()["status"], "error")

    def test_non_publication_disk_history_and_status_remain_unchanged(self):
        task_id = "ordinary-restart"
        secret = os.path.join(self.tmpdir.name, "ordinary-output")
        metadata_dir = os.path.join(self.state.merge_dir, task_id)
        os.makedirs(metadata_dir)
        with open(os.path.join(metadata_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump({"type": "merge", "output_path": secret}, handle)
        self.services.status_from_disk = lambda _task_id: {"status": "completed", "result": {"output_path": secret}}

        history = self.app.test_client().get("/api/history/%s" % task_id)
        status = self.app.test_client().get("/api/status/%s" % task_id)

        self.assertEqual(history.get_json()["data"]["output_path"], secret)
        self.assertEqual(status.get_json()["result"]["output_path"], secret)

    def test_resume_clears_stale_abort_control(self):
        task_id = "ordinary-interrupted"
        data = {"type": "merge"}
        control = {"aborted": True, "process": "stale"}
        self.state.tasks[task_id] = {
            "status": "interrupted",
            "priority": "common",
            "created_at": time.time(),
            "original_data": data,
            "control": control,
        }

        response = self._post("/api/resume/%s" % task_id, {})

        self.assertEqual(response.status_code, 200)
        self.assertFalse(control["aborted"])
        self.assertIsNone(control["process"])

    def test_running_validation_cancel_sets_abort_and_terminates_child(self):
        task_id = "running-task"
        staging = os.path.join(self.published, ".staging", "running-publication")
        os.makedirs(staging)
        config = {"publication_id": "running-publication", "publication_root": self.published, "staging_path": staging}
        self._add_publication_task(task_id, "validating", config)
        process = mock.Mock()
        control = {"aborted": False, "process": process}
        self.state.tasks[task_id] = {"status": "running", "control": control}
        self.state.running_task_info["id"] = task_id

        response = self._post("/api/model-publications/%s/cancel" % task_id, {}, **self.headers)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(control["aborted"])
        process.terminate.assert_called_once_with()

    def test_cancel_accepted_before_commit_transition_prevents_commit(self):
        from app.model_publication_tasks import run_publication_validation
        from app.model_publication import _inventory

        task_id = "commit-task"
        publication_id = "commit-publication"
        staging = os.path.join(self.published, ".staging", publication_id)
        os.makedirs(staging)
        files, total_bytes = _inventory(staging)
        config = {
            "publication_id": publication_id,
            "publication_root": self.published,
            "staging_path": staging,
            "staging_inventory": {"files": files, "total_bytes": total_bytes},
            "display_name": "published",
        }
        self._add_publication_task(task_id, "validating", config)
        control = {"aborted": False, "process": None, "lock": self.state.scheduler_lock}
        self.state.tasks[task_id] = {"status": "running", "control": control}
        self.state.running_task_info["id"] = task_id
        manifest_started = threading.Event()
        release_manifest = threading.Event()
        result = {}
        inspection = SimpleNamespace(architectures=("Qwen2ForCausalLM",))

        def build_manifest(*_args):
            manifest_started.set()
            release_manifest.wait(5)
            return {
                "publication_id": publication_id,
                "files": {"entries": files, "total_bytes": total_bytes},
            }

        def validate():
            with self.app.app_context():
                result.update(run_publication_validation(
                    task_id, [0], lambda *_args: None, control,
                    functional_validate_fn=lambda *_args: {"status": "passed"},
                ))

        patches = [
            mock.patch("app.model_publication_tasks.publication_gpu_preflight", return_value=[{"index": 0}]),
            mock.patch("app.model_publication_tasks.inspect_model", return_value=inspection),
            mock.patch("app.model_publication_tasks.inspect_serving_compatibility", return_value={"status": "ready"}),
            mock.patch("app.model_publication_tasks.build_manifest", side_effect=build_manifest),
            mock.patch("app.model_publication_tasks.commit_staging", return_value={"publication_id": publication_id}),
        ]
        active_mocks = [patcher.start() for patcher in patches]
        commit_mock = active_mocks[-1]
        thread = threading.Thread(target=validate)
        try:
            thread.start()
            self.assertTrue(manifest_started.wait(5))
            response = self._post("/api/model-publications/%s/cancel" % task_id, {}, **self.headers)
            self.assertEqual(response.status_code, 200)
        finally:
            release_manifest.set()
            thread.join(5)
            for patcher in reversed(patches):
                patcher.stop()
        self.assertEqual(result["status"], "canceled")
        commit_mock.assert_not_called()
        self.assertFalse(os.path.exists(staging))
        with self.app.app_context():
            self.assertEqual(self.db.session.get(self.Task, task_id).status, "canceled")

    def test_cancel_after_commit_transition_returns_commit_in_progress(self):
        task_id = "committing-task"
        config = {
            "publication_id": "committing-publication",
            "publication_root": self.published,
            "commit_in_progress": True,
        }
        self._add_publication_task(task_id, "registration_pending", config)

        response = self._post("/api/model-publications/%s/cancel" % task_id, {}, **self.headers)

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error"]["code"], "commit_in_progress")

    def test_public_status_redacts_publication_host_paths_and_snapshots_only(self):
        secret = os.path.join(self.tmpdir.name, "secret-model")
        self.state.tasks["publication-status"] = {
            "type": "model_publication",
            "status": "validating",
            "progress": 80,
            "message": "Awaiting validation",
            "original_data": {"type": "model_publication", "recipe_snapshot": {"path": secret}},
            "result": {"status": "validating", "staging_path": secret, "inspection": {"path": secret}},
        }
        self.state.tasks["ordinary-status"] = {
            "type": "merge",
            "status": "completed",
            "result": {"output_path": secret},
        }

        publication = self.app.test_client().get("/api/status/publication-status")
        ordinary = self.app.test_client().get("/api/status/ordinary-status")

        publication_text = json.dumps(publication.get_json(), sort_keys=True)
        self.assertNotIn(secret, publication_text)
        self.assertNotIn("staging_path", publication_text)
        self.assertNotIn("inspection", publication_text)
        self.assertEqual(ordinary.get_json()["result"]["output_path"], secret)

    def test_delete_guard_matches_symlink_equivalent_service_path(self):
        from app.model_gateway.models import ServingModelService
        from app.model_publication import PublicationError

        publication_id = "published-model"
        real_path = os.path.join(self.published, publication_id)
        alias = os.path.join(self.tmpdir.name, "published-alias")
        os.makedirs(real_path)
        os.symlink(real_path, alias)
        with self.app.app_context():
            self.db.session.add(ServingModelService(
                model_id=None,
                model_path=alias + "/",
                display_name="service",
                served_model_name="service-name",
                status="stopped",
            ))
            self.db.session.commit()

        def delete(candidate_id, _root, reference_check, _delete_model):
            if reference_check(candidate_id, os.path.realpath(real_path)):
                raise PublicationError("publication_referenced", "referenced")
            return {"publication_id": candidate_id, "deleted": True}

        with mock.patch("app.model_publication.delete_published_asset", side_effect=delete):
            response = self.app.test_client().delete(
                "/api/model-publications/%s" % publication_id,
                headers=self.headers,
            )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error"]["code"], "asset_in_use")

    def test_legacy_path_delete_requires_gateway_admin_for_published_asset_including_alias(self):
        model = self._published_model("legacy-path-auth")
        alias = os.path.join(self.tmpdir.name, "published-auth-alias")
        os.symlink(model.path, alias)

        missing = self.app.test_client().post("/api/models/delete", json={"path": alias})
        wrong = self.app.test_client().post(
            "/api/models/delete",
            json={"path": alias},
            headers={"Authorization": "Bearer wrong"},
        )
        authorized = self.app.test_client().post(
            "/api/models/delete",
            json={"path": alias},
            headers=self.headers,
        )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        self.assertFalse(os.path.exists(model.path))

    def test_legacy_id_delete_requires_gateway_admin_for_published_asset(self):
        model = self._published_model("legacy-id-auth")

        missing = self.app.test_client().delete("/api/model_repo/%s" % model.id)
        wrong = self.app.test_client().delete(
            "/api/model_repo/%s" % model.id,
            headers={"Authorization": "Bearer wrong"},
        )
        authorized = self.app.test_client().delete("/api/model_repo/%s" % model.id, headers=self.headers)

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        self.assertFalse(os.path.exists(model.path))

    def test_legacy_raw_id_cannot_bypass_published_auth_or_reference_guard(self):
        model = self._published_model("legacy-raw-id-guard")
        self._gateway_service(model, status="stopped")
        self.services.model_repo_save_raw({
            "alternate-raw-id": {"path": model.path + os.sep, "name": "alias"},
        })

        unauthorized = self.app.test_client().delete("/api/model_repo/alternate-raw-id")
        self.assertEqual(unauthorized.status_code, 401)
        self.assertTrue(os.path.isdir(model.path))
        self.assertIn("alternate-raw-id", self.services.model_repo_load_raw())

        authorized = self.app.test_client().delete(
            "/api/model_repo/alternate-raw-id",
            headers=self.headers,
        )
        self.assertEqual(authorized.status_code, 409)
        self.assertEqual(authorized.get_json()["error"]["code"], "asset_in_use")
        self.assertTrue(os.path.isdir(model.path))
        self.assertIn("alternate-raw-id", self.services.model_repo_load_raw())

    def test_legacy_core_id_alias_cannot_bypass_unconfigured_admin_or_reference_guard(self):
        model = self._published_model("legacy-core-id-guard")
        self._gateway_service(model, status="stopped")
        alias = os.path.join(self.tmpdir.name, "legacy-core-id-alias")
        os.symlink(model.path, alias)
        with self.app.app_context():
            self.db.session.add(self.Model(
                id="alternate-core-id",
                name="alias",
                path=alias + os.sep,
                source="base",
            ))
            self.db.session.commit()

        token = self.app.config["MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN"]
        self.app.config["MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN"] = ""
        unavailable = self.app.test_client().delete(
            "/api/model_repo/alternate-core-id",
            headers=self.headers,
        )
        self.assertEqual(unavailable.status_code, 503)
        self.assertTrue(os.path.isdir(model.path))
        self.assertTrue(os.path.lexists(alias))

        self.app.config["MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN"] = token
        authorized = self.app.test_client().delete(
            "/api/model_repo/alternate-core-id",
            headers=self.headers,
        )
        self.assertEqual(authorized.status_code, 409)
        self.assertEqual(authorized.get_json()["error"]["code"], "asset_in_use")
        self.assertTrue(os.path.isdir(model.path))
        self.assertTrue(os.path.lexists(alias))

    def test_legacy_delete_routes_preserve_unauthenticated_non_published_behavior(self):
        path = self._make_legacy_model_dir("legacy-path")

        by_path = self.app.test_client().post("/api/models/delete", json={"path": path})
        by_id = self.app.test_client().delete("/api/model_repo/model-1")

        self.assertEqual(by_path.status_code, 200)
        self.assertEqual(by_id.status_code, 200)

    def test_all_model_delete_routes_return_asset_in_use_for_active_gateway_service(self):
        model = self._published_model("cross-route-in-use")
        self._gateway_service(model)

        by_path = self.app.test_client().post(
            "/api/models/delete",
            json={"path": model.path},
            headers=self.headers,
        )
        by_id = self.app.test_client().delete("/api/model_repo/%s" % model.id, headers=self.headers)
        formal = self.app.test_client().delete(
            "/api/model-publications/cross-route-in-use",
            headers=self.headers,
        )

        for response in (by_path, by_id, formal):
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.get_json()["error"]["code"], "asset_in_use")
        self.assertTrue(os.path.isdir(model.path))

    def test_formal_delete_route_restores_asset_when_core_delete_raises(self):
        model = self._published_model("rollback-core-delete")

        with mock.patch("app.model_publication._delete_core_model_by_canonical_path", side_effect=RuntimeError("db failed")):
            response = self.app.test_client().delete(
                "/api/model-publications/rollback-core-delete",
                headers=self.headers,
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"]["code"], "asset_delete_failed")
        self.assertTrue(os.path.isdir(model.path))
        self.assertFalse(os.path.exists(os.path.join(self.published, ".trash", "rollback-core-delete")))

    def test_formal_delete_route_authorized_success(self):
        model = self._published_model("formal-delete-success")

        response = self.app.test_client().delete(
            "/api/model-publications/formal-delete-success",
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(os.path.exists(model.path))

    def test_manifest_rejects_symlink_escape_and_traversal(self):
        outside = os.path.join(self.tmpdir.name, "outside")
        os.makedirs(outside)
        with open(os.path.join(outside, "publication_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"host_path": outside}, handle)
        os.makedirs(self.published)
        os.symlink(outside, os.path.join(self.published, "escaped"))

        escaped = self.app.test_client().get("/api/model-publications/escaped/manifest", headers=self.headers)
        traversal = self.app.test_client().get("/api/model-publications/%2e%2e/manifest", headers=self.headers)

        self.assertNotEqual(escaped.status_code, 200)
        self.assertNotEqual(traversal.status_code, 200)


if __name__ == "__main__":
    unittest.main()
