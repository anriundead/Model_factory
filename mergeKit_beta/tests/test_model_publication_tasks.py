import json
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from flask import Flask, has_app_context


class _OneShotQueue:
    def __init__(self, item):
        self.item = item
        self.used = False

    def get(self):
        if self.used:
            raise KeyboardInterrupt
        self.used = True
        return self.item

    def task_done(self):
        pass


class PublicationTaskTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmpdir.name, "published")
        self.source = os.path.join(self.tmpdir.name, "source")
        self.current_source = os.path.join(self.tmpdir.name, "current-source")
        self._write_model(self.source, b"stale")
        self._write_model(self.current_source, b"current")
        self.request = {
            "publication_id": "publication-a",
            "publication_root": self.root,
            "source_type": "existing_model",
            "model_id": "model-1",
            "source_path": self.source,
            "display_name": "Published test model",
        }
        self.progress = lambda _percent, _message: None

        app = Flask(__name__)
        app.config.update(
            TESTING=True,
            SQLALCHEMY_DATABASE_URI="sqlite:///%s" % os.path.join(self.tmpdir.name, "tasks.db"),
            SQLALCHEMY_BINDS={"model_gateway": "sqlite:///%s" % os.path.join(self.tmpdir.name, "gateway.db")},
        )
        from app.extensions import db
        from app.models import Model

        db.init_app(app)
        with app.app_context():
            db.create_all()
            db.session.add(Model(id="model-1", name="core", path=self.current_source, source="base"))
            db.session.commit()
        self.app = app
        self.db = db

    def tearDown(self):
        with self.app.app_context():
            self.db.session.remove()
        self.tmpdir.cleanup()

    def _write_model(self, path, weights=b"weights"):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as handle:
            json.dump({
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "vocab_size": 32,
            }, handle)
        with open(os.path.join(path, "model.safetensors"), "wb") as handle:
            handle.write(weights)
        with open(os.path.join(path, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
            json.dump({"weight_map": {"model.weight": "model.safetensors"}}, handle)

    def _add_task(self, task_id="task-a", status="queued", config=None):
        from app.models import Task

        task = Task(
            id=task_id,
            task_type="model_publication",
            status=status,
            config=dict(config or self.request),
        )
        self.db.session.add(task)
        self.db.session.commit()
        return task

    def _staging_task(self, task_id="task-a"):
        staging = os.path.join(self.root, ".staging", self.request["publication_id"])
        self._write_model(staging)
        config = {**self.request, "staging_path": staging, "structural_validation": {"status": "passed"}}
        self._add_task(task_id, "validating", config)
        return staging

    def structural_validate(self, staging):
        self.assertTrue(os.path.isfile(os.path.join(staging, "model.safetensors")))
        return {"status": "passed"}

    def test_worker_runs_only_publication_branch_with_app_context_and_persists_status(self):
        from app.models import Task
        from app.services import Services

        task_id = "worker-task"
        with self.app.app_context():
            self._add_task(task_id, "queued")

        data = {**self.request, "task_id": task_id, "type": "model_publication"}
        state = SimpleNamespace(
            task_queue=_OneShotQueue((10, 1.0, task_id, data)),
            scheduler_lock=threading.Lock(),
            tasks={task_id: {"status": "queued", "original_data": data}},
            running_task_info={"id": None, "priority": None, "process": None},
            merge_dir=self.tmpdir.name,
        )
        service = Services.__new__(Services)
        service.state = state
        service.app = self.app
        service.logger = mock.Mock()
        service._post_task_gpu_cleanup = mock.Mock()
        observed_context = []

        def publication(*_args, **_kwargs):
            observed_context.append(has_app_context())
            from app.repositories import task_set_status
            task_set_status(task_id, "validating")
            return {"status": "validating"}

        with mock.patch("app.model_publication_tasks.run_model_publication_task", side_effect=publication):
            with self.assertRaises(KeyboardInterrupt):
                service.worker()

        with self.app.app_context():
            self.assertEqual(observed_context, [True])
            self.assertEqual(self.db.session.get(Task, task_id).status, "validating")

    def test_worker_preserves_cancellation_control_already_set_while_queued(self):
        from app.services import Services

        task_id = "queued-race"
        data = {**self.request, "task_id": task_id, "type": "model_publication"}
        control = {"aborted": True, "process": None}
        state = SimpleNamespace(
            task_queue=_OneShotQueue((10, 1.0, task_id, data)),
            scheduler_lock=threading.Lock(),
            tasks={task_id: {"status": "queued", "original_data": data, "control": control}},
            running_task_info={"id": None, "priority": None, "process": None},
            merge_dir=self.tmpdir.name,
        )
        service = Services.__new__(Services)
        service.state = state
        service.app = self.app
        service.logger = mock.Mock()
        service._post_task_gpu_cleanup = mock.Mock()

        with mock.patch("app.model_publication_tasks.run_model_publication_task", return_value={"status": "canceled"}) as run:
            with self.assertRaises(KeyboardInterrupt):
                service.worker()

        self.assertIs(run.call_args.args[3], control)
        self.assertTrue(run.call_args.args[3]["aborted"])

    def test_materialization_re_resolves_existing_model_from_core_orm(self):
        from app.model_publication_tasks import run_model_publication_task

        with self.app.app_context():
            self._add_task()
            result = run_model_publication_task(
                "task-a", self.request, self.progress, {"aborted": False},
                structural_validate_fn=self.structural_validate,
            )

        staged = os.path.join(self.root, ".staging", "publication-a", "model.safetensors")
        self.assertEqual(result["status"], "validating")
        with open(staged, "rb") as handle:
            self.assertEqual(handle.read(), b"current")

    def test_cancel_before_commit_removes_staging_and_persists(self):
        from app.models import Task
        from app.model_publication_tasks import run_model_publication_task

        with self.app.app_context():
            self._add_task("task-b")
            result = run_model_publication_task("task-b", self.request, self.progress, {"aborted": True})
            task = self.db.session.get(Task, "task-b")

        self.assertEqual(result["error_code"], "canceled")
        self.assertEqual(task.status, "canceled")
        self.assertFalse(os.path.exists(os.path.join(self.root, ".staging", "publication-a")))

    def test_validate_requires_explicit_non_gpu2_ids(self):
        from app.model_publication import PublicationError
        from app.model_publication_tasks import run_publication_validation

        with self.assertRaisesRegex(PublicationError, "gpu_selection_required"):
            run_publication_validation("task-a", [], self.progress, {}, functional_validate_fn=lambda *_args: None)
        with self.assertRaisesRegex(PublicationError, "gpu_selection_required"):
            run_publication_validation("task-a", [1, 1], self.progress, {}, functional_validate_fn=lambda *_args: None)
        with self.assertRaisesRegex(PublicationError, "protected_gpu"):
            run_publication_validation("task-a", [2], self.progress, {}, functional_validate_fn=lambda *_args: None)

    def test_preflight_failure_keeps_validating_staging_resumable(self):
        from app.models import Task
        from app.model_publication import PublicationError
        from app.model_publication_tasks import run_publication_validation

        with self.app.app_context():
            staging = self._staging_task()
            with mock.patch("app.model_publication_tasks.publication_gpu_preflight", side_effect=PublicationError("gpu_busy", "busy")):
                result = run_publication_validation("task-a", [0], self.progress, {"aborted": False})
            task = self.db.session.get(Task, "task-a")

        self.assertEqual(result["status"], "validating")
        self.assertEqual(task.status, "validating")
        self.assertEqual(task.config["error_code"], "gpu_busy")
        self.assertTrue(os.path.isdir(staging))

    def test_registration_failure_remains_registration_pending_for_reconciliation(self):
        from app.models import Task
        from app.model_publication_tasks import run_publication_validation

        inspection = SimpleNamespace(architectures=("Qwen2ForCausalLM",))
        with self.app.app_context():
            staging = self._staging_task()
            with mock.patch("app.model_publication_tasks.publication_gpu_preflight", return_value=[{"index": 0}]):
                with mock.patch("app.model_publication_tasks.inspect_model", return_value=inspection):
                    with mock.patch("app.model_publication_tasks.inspect_serving_compatibility", return_value={"status": "ready"}):
                        with mock.patch("app.model_publication_tasks.build_manifest", return_value={"publication_id": "publication-a"}):
                            with mock.patch("app.model_publication_tasks.commit_staging", side_effect=OSError("rename failed")):
                                result = run_publication_validation(
                                    "task-a", [0], self.progress, {"aborted": False, "lock": threading.Lock()},
                                    functional_validate_fn=lambda *_args: {"status": "passed"},
                                )
            task = self.db.session.get(Task, "task-a")

        self.assertEqual(result["status"], "registration_pending")
        self.assertEqual(task.status, "registration_pending")
        self.assertTrue(os.path.isdir(staging))

    def test_vlm_cmmmu_result_is_recorded_in_manifest_evaluation(self):
        from app.model_publication_tasks import run_publication_validation

        inspection = SimpleNamespace(architectures=("Qwen2_5_VLForConditionalGeneration",))
        functional = {
            "status": "passed",
            "image": {"status": "passed"},
            "evaluation": {"cmmmu": {"samples": 1, "acc": 100.0}},
        }
        with self.app.app_context():
            self._staging_task()
            with mock.patch("app.model_publication_tasks.publication_gpu_preflight", return_value=[{"index": 0}]):
                with mock.patch("app.model_publication_tasks.inspect_model", return_value=inspection):
                    with mock.patch("app.model_publication_tasks.inspect_serving_compatibility", return_value={"status": "ready"}):
                        with mock.patch("app.model_publication_tasks.build_manifest", return_value={"publication_id": "publication-a"}) as build:
                            with mock.patch("app.model_publication_tasks.commit_staging", return_value={"publication_id": "publication-a"}):
                                run_publication_validation(
                                    "task-a", [0], self.progress, {"aborted": False, "lock": threading.Lock()},
                                    functional_validate_fn=lambda *_args: functional,
                                )

        self.assertEqual(build.call_args.args[3]["evaluation"], functional["evaluation"])

    def test_preflight_fails_closed_for_process_query_memory_and_uuid(self):
        from core.gpu_topology import GpuInfo
        from app.model_publication import PublicationError
        from app.model_publication_tasks import publication_gpu_preflight

        gpu_line = "0, GPU-good, 100, 24576\n"
        topology = [GpuInfo(index=0, mem_free_mib=24476, mem_total_mib=24576)]

        def completed(stdout="", returncode=0, stderr=""):
            return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

        with mock.patch("core.gpu_topology.query_gpus", return_value=topology):
            with mock.patch("app.model_publication_tasks.subprocess.run", side_effect=[completed(gpu_line), completed(returncode=1, stderr="query failed")]):
                with self.assertRaisesRegex(PublicationError, "gpu_preflight_failed"):
                    publication_gpu_preflight([0], required_bytes=1)

            with mock.patch("app.model_publication_tasks.subprocess.run", side_effect=[completed(gpu_line), completed()]):
                with self.assertRaisesRegex(PublicationError, "insufficient_gpu_memory"):
                    publication_gpu_preflight([0], required_bytes=30 * 1024**3)

            bad_uuid = "0, not-a-uuid, 100, 24576\n"
            with mock.patch("app.model_publication_tasks.subprocess.run", side_effect=[completed(bad_uuid), completed()]):
                with self.assertRaisesRegex(PublicationError, "gpu_preflight_failed"):
                    publication_gpu_preflight([0], required_bytes=1)

    def test_functional_validation_scopes_cuda_only_to_cancellable_child(self):
        from app.model_publication_tasks import validate_model_functionally

        process = mock.Mock()
        process.poll.return_value = 0
        process.returncode = 0
        process.communicate.return_value = ('{"status":"passed"}', "")
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        with mock.patch("app.model_publication_tasks.subprocess.Popen", return_value=process) as popen:
            result = validate_model_functionally(self.source, [1, 3], {"aborted": False})

        self.assertEqual(result["status"], "passed")
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1,3")
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), original)

    def test_vlm_recipe_materializes_with_fresh_managed_base(self):
        from app.model_publication_tasks import _materialize_recipe

        recipe = os.path.join(self.tmpdir.name, "recipe.json")
        with open(recipe, "w", encoding="utf-8") as handle:
            json.dump({"artifact_type": "vlm", "model_paths": [self.source, self.current_source], "best_genotype": [1, 1]}, handle)
        params = {
            "recipe_id": "recipe",
            "recipe_path": recipe,
            "vlm_base_model_id": "model-1",
            "vlm_base_path": self.source,
        }
        with open(recipe, encoding="utf-8") as handle:
            recipe_data = json.load(handle)
        with self.app.app_context():
            with mock.patch("app.model_publication_tasks._resolve_recipe", return_value=(recipe, recipe_data)):
                with mock.patch("merge_manager.run_recipe_apply_task", return_value={"status": "success"}):
                    with mock.patch("app.model_inspection.resolve_vlm_base", return_value=SimpleNamespace(path=self.current_source)):
                        with mock.patch("evolution.vendor.vlm_merge.model_composition.materialize_full_vlm") as materialize:
                            _materialize_recipe("task-vlm", params, os.path.join(self.root, ".staging", "vlm"), self.progress, {})
        self.assertEqual(materialize.call_args.args[1], os.path.realpath(self.current_source))


if __name__ == "__main__":
    unittest.main()
