import json
import os
import subprocess
import sys
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
        self.protected_gpu_env = mock.patch.dict(
            os.environ,
            {"MERGEKIT_PROTECTED_GPU_UUIDS": "GPU-7eff453d-60f0-37ed-92c1-0aec341c497d"},
            clear=False,
        )
        self.protected_gpu_env.start()
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
        self.protected_gpu_env.stop()

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
        from app.model_publication import _inventory

        staging = os.path.join(self.root, ".staging", self.request["publication_id"])
        self._write_model(staging)
        files, total_bytes = _inventory(staging)
        config = {
            **self.request,
            "staging_path": staging,
            "structural_validation": {"status": "passed"},
            "staging_inventory": {"files": files, "total_bytes": total_bytes},
        }
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

    def test_worker_gives_non_publication_tasks_a_fresh_control_after_resume(self):
        from app.services import Services

        task_id = "ordinary-resume"
        data = {"task_id": task_id, "type": "merge", "model_paths": []}
        state = SimpleNamespace(
            task_queue=_OneShotQueue((10, 1.0, task_id, data)),
            scheduler_lock=threading.Lock(),
            tasks={task_id: {
                "status": "queued", "original_data": data,
                "control": {"aborted": True, "process": "stale"},
            }},
            running_task_info={"id": None, "priority": None, "process": None},
            merge_dir=self.tmpdir.name,
        )
        service = Services.__new__(Services)
        service.state = state
        service.app = self.app
        service.logger = mock.Mock()
        service._post_task_gpu_cleanup = mock.Mock()

        with mock.patch("importlib.reload", side_effect=lambda module: module):
            with mock.patch("merge_manager.run_merge_task", return_value={"status": "success"}) as run:
                with self.assertRaises(KeyboardInterrupt):
                    service.worker()

        control = run.call_args.args[3]
        self.assertFalse(control["aborted"])
        self.assertIsNone(control["process"])

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

    def test_validation_failure_clears_enqueued_marker_for_retry(self):
        from app.models import Task
        from app.model_publication import PublicationError
        from app.model_publication_tasks import run_publication_validation

        with self.app.app_context():
            self._staging_task()
            task = self.db.session.get(Task, "task-a")
            task.config = {**task.config, "validation_enqueued": True}
            self.db.session.commit()
            with mock.patch("app.model_publication_tasks.publication_gpu_preflight", side_effect=PublicationError("gpu_busy", "busy")):
                result = run_publication_validation("task-a", [0], self.progress, {"aborted": False})
            task = self.db.session.get(Task, "task-a")

        self.assertEqual(result["status"], "validating")
        self.assertFalse(task.config["validation_enqueued"])

    def test_validation_rejects_staging_tampered_after_structural_baseline(self):
        from app.models import Task
        from app.model_publication_tasks import run_model_publication_task, run_publication_validation

        with self.app.app_context():
            self._add_task()
            materialized = run_model_publication_task(
                "task-a", self.request, self.progress, {"aborted": False},
                structural_validate_fn=self.structural_validate,
            )
            task = self.db.session.get(Task, "task-a")
            baseline = task.config.get("staging_inventory")
            staging = task.config["staging_path"]
            with open(os.path.join(staging, "model.safetensors"), "ab") as handle:
                handle.write(b"tampered")
            functional = mock.Mock(return_value={"status": "passed"})
            result = run_publication_validation("task-a", [0], self.progress, {"aborted": False}, functional_validate_fn=functional)
            task = self.db.session.get(Task, "task-a")

        self.assertEqual(materialized["status"], "validating")
        self.assertTrue(baseline["files"][0]["sha256"])
        self.assertEqual(result["error_code"], "staging_changed")
        functional.assert_not_called()
        self.assertEqual(task.status, "validating")

    def test_validation_rechecks_manifest_inventory_after_functional_validation(self):
        from app.models import Task
        from app.model_publication_tasks import run_publication_validation

        inspection = SimpleNamespace(
            architectures=("Qwen2ForCausalLM",),
            is_vlm=False,
            model_type="qwen2",
            processor_class=None,
        )
        with self.app.app_context():
            staging = self._staging_task()

            selected_devices = []

            def functional_validate(path, devices, *_args):
                selected_devices.extend(devices)
                with open(os.path.join(path, "model.safetensors"), "ab") as handle:
                    handle.write(b"mutated-after-baseline")
                return {"status": "passed"}

            with mock.patch("app.model_publication_tasks.publication_gpu_preflight", return_value=[{
                "index": 0,
                "uuid": "GPU-23348268-6430-c539-b7e5-762583f50e91",
                "pci_bus_id": "00000000:01:00.0",
            }]):
                with mock.patch("app.model_publication_tasks.inspect_model", return_value=inspection):
                    with mock.patch("app.model_publication_tasks.inspect_serving_compatibility", return_value={"status": "ready"}):
                        with mock.patch("app.model_publication_tasks.commit_staging") as commit:
                            result = run_publication_validation(
                                "task-a", [0], self.progress, {"aborted": False, "lock": threading.Lock()},
                                functional_validate_fn=functional_validate,
                            )
            task = self.db.session.get(Task, "task-a")

        self.assertEqual(result["status"], "validating")
        self.assertEqual(result["error_code"], "staging_changed")
        self.assertEqual(task.status, "validating")
        self.assertEqual(task.config["error_code"], "staging_changed")
        self.assertTrue(os.path.isdir(staging))
        self.assertEqual(selected_devices, ["GPU-23348268-6430-c539-b7e5-762583f50e91"])
        commit.assert_not_called()

    def test_materialization_cancels_after_structural_validation_before_status_write(self):
        from app.models import Task
        from app.model_publication_tasks import run_model_publication_task

        control = {"aborted": False}

        def structural_validate(staging):
            self.structural_validate(staging)
            control["aborted"] = True
            return {"status": "passed"}

        with self.app.app_context():
            self._add_task()
            result = run_model_publication_task("task-a", self.request, self.progress, control, structural_validate_fn=structural_validate)
            task = self.db.session.get(Task, "task-a")

        self.assertEqual(result["status"], "canceled")
        self.assertEqual(task.status, "canceled")
        self.assertFalse(os.path.exists(os.path.join(self.root, ".staging", "publication-a")))

    def test_materialization_cancels_after_inventory_before_status_write(self):
        from app.models import Task
        from app.model_publication import _inventory as inventory
        from app.model_publication_tasks import run_model_publication_task

        control = {"aborted": False}

        def inventory_then_cancel(*args, **kwargs):
            result = inventory(*args, **kwargs)
            control["aborted"] = True
            return result

        with self.app.app_context():
            self._add_task()
            with mock.patch("app.model_publication_tasks._inventory", side_effect=inventory_then_cancel):
                result = run_model_publication_task(
                    "task-a", self.request, self.progress, control, structural_validate_fn=self.structural_validate,
                )
            task = self.db.session.get(Task, "task-a")

        self.assertEqual(result["status"], "canceled")
        self.assertEqual(task.status, "canceled")
        self.assertFalse(os.path.exists(os.path.join(self.root, ".staging", "publication-a")))

    def test_materialization_error_leaves_terminal_failure_to_worker(self):
        from app.models import Task
        from app.model_publication_tasks import run_model_publication_task

        with self.app.app_context():
            self._add_task()
            result = run_model_publication_task(
                "task-a", self.request, self.progress, {"aborted": False},
                copy_fn=mock.Mock(side_effect=OSError("copy failed")),
            )
            task = self.db.session.get(Task, "task-a")

        self.assertEqual(result["status"], "error")
        self.assertEqual(task.status, "materializing")
        self.assertFalse(os.path.exists(os.path.join(self.root, ".staging", "publication-a")))

    def test_worker_exception_persists_publication_failure_unless_durable_state(self):
        from app.models import Task
        from app.services import Services

        task_id = "worker-persist-failure"
        with self.app.app_context():
            self._add_task(task_id, "materializing")
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

        with mock.patch("app.model_publication_tasks.run_model_publication_task", side_effect=OSError("status write failed")):
            with self.assertRaises(KeyboardInterrupt):
                service.worker()

        with self.app.app_context():
            self.assertEqual(self.db.session.get(Task, task_id).status, "failed")

    def test_registration_failure_remains_registration_pending_for_reconciliation(self):
        from app.models import Task
        from app.model_publication_tasks import run_publication_validation

        inspection = SimpleNamespace(architectures=("Qwen2ForCausalLM",))
        with self.app.app_context():
            staging = self._staging_task()
            baseline = self.db.session.get(Task, "task-a").config["staging_inventory"]
            manifest = {
                "publication_id": "publication-a",
                "files": {"entries": baseline["files"], "total_bytes": baseline["total_bytes"]},
            }
            with mock.patch("app.model_publication_tasks.publication_gpu_preflight", return_value=[{
                "index": 0,
                "uuid": "GPU-23348268-6430-c539-b7e5-762583f50e91",
                "pci_bus_id": "00000000:01:00.0",
            }]):
                with mock.patch("app.model_publication_tasks.inspect_model", return_value=inspection):
                    with mock.patch("app.model_publication_tasks.inspect_serving_compatibility", return_value={"status": "ready"}):
                        with mock.patch("app.model_publication_tasks.build_manifest", return_value=manifest):
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
        from app.models import Task
        from app.model_publication_tasks import run_publication_validation

        inspection = SimpleNamespace(architectures=("Qwen2_5_VLForConditionalGeneration",))
        functional = {
            "status": "passed",
            "image": {"status": "passed"},
            "evaluation": {"cmmmu": {"samples": 1, "acc": 100.0}},
        }
        with self.app.app_context():
            self._staging_task()
            task = self.db.session.get(Task, "task-a")
            task.error = "old validation failure"
            task.config = {**task.config, "validation_enqueued": True}
            self.db.session.commit()
            baseline = task.config["staging_inventory"]
            manifest = {
                "publication_id": "publication-a",
                "files": {"entries": baseline["files"], "total_bytes": baseline["total_bytes"]},
            }
            with mock.patch("app.model_publication_tasks.publication_gpu_preflight", return_value=[{
                "index": 0,
                "uuid": "GPU-23348268-6430-c539-b7e5-762583f50e91",
                "pci_bus_id": "00000000:01:00.0",
            }]):
                with mock.patch("app.model_publication_tasks.inspect_model", return_value=inspection):
                    with mock.patch("app.model_publication_tasks.inspect_serving_compatibility", return_value={"status": "ready"}):
                        with mock.patch("app.model_publication_tasks.build_manifest", return_value=manifest) as build:
                            with mock.patch("app.model_publication_tasks.commit_staging", return_value={"publication_id": "publication-a"}):
                                run_publication_validation(
                                    "task-a", [0], self.progress, {"aborted": False, "lock": threading.Lock()},
                                    functional_validate_fn=lambda *_args: functional,
                                )
            task = self.db.session.get(Task, "task-a")

        self.assertEqual(build.call_args.args[3]["evaluation"], functional["evaluation"])
        self.assertEqual(task.status, "completed")
        self.assertEqual(task.error, "")
        self.assertFalse(task.config["validation_enqueued"])

    def test_preflight_fails_closed_for_process_query_memory_and_uuid(self):
        from core.gpu_topology import GpuInfo
        from app.model_publication import PublicationError
        from app.model_publication_tasks import publication_gpu_preflight

        gpu_line = "0, GPU-23348268-6430-c539-b7e5-762583f50e91, 00000000:01:00.0, 100, 24576\n"
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

            bad_uuid = "0, not-a-uuid, 00000000:01:00.0, 100, 24576\n"
            with mock.patch("app.model_publication_tasks.subprocess.run", side_effect=[completed(bad_uuid), completed()]):
                with self.assertRaisesRegex(PublicationError, "gpu_preflight_failed"):
                    publication_gpu_preflight([0], required_bytes=1)

            truncated_uuid = "0, GPU-23348268-6430-c539-b7e5, 00000000:01:00.0, 100, 24576\n"
            with mock.patch("app.model_publication_tasks.subprocess.run", side_effect=[completed(truncated_uuid), completed()]):
                with self.assertRaisesRegex(PublicationError, "gpu_preflight_failed"):
                    publication_gpu_preflight([0], required_bytes=1)

    def test_preflight_rejects_non_decimal_gpu_inventory_numbers(self):
        from core.gpu_topology import GpuInfo
        from app.model_publication import PublicationError
        from app.model_publication_tasks import publication_gpu_preflight

        topology = [GpuInfo(index=0, mem_free_mib=24476, mem_total_mib=24576)]

        def completed(stdout="", returncode=0, stderr=""):
            return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

        with mock.patch("core.gpu_topology.query_gpus", return_value=topology):
            for field_index in range(3):
                for invalid in ("1.0", "1e3", "+1", "-1", ""):
                    with self.subTest(field_index=field_index, invalid=invalid):
                        line = "0, GPU-23348268-6430-c539-b7e5-762583f50e91, 00000000:01:00.0, 100, 24576\n"
                        if field_index == 0:
                            line = "%s, GPU-23348268-6430-c539-b7e5-762583f50e91, 00000000:01:00.0, 100, 24576\n" % invalid
                        elif field_index == 1:
                            line = "0, GPU-23348268-6430-c539-b7e5-762583f50e91, 00000000:01:00.0, %s, 24576\n" % invalid
                        else:
                            line = "0, GPU-23348268-6430-c539-b7e5-762583f50e91, 00000000:01:00.0, 100, %s\n" % invalid
                        with mock.patch("app.model_publication_tasks.subprocess.run", side_effect=[completed(line), completed()]):
                            with self.assertRaisesRegex(PublicationError, "gpu_preflight_failed"):
                                publication_gpu_preflight([0], required_bytes=1)

    def test_preflight_rejects_non_decimal_compute_pids(self):
        from core.gpu_topology import GpuInfo
        from app.model_publication import PublicationError
        from app.model_publication_tasks import publication_gpu_preflight

        topology = [GpuInfo(index=0, mem_free_mib=24476, mem_total_mib=24576)]
        inventory = "0, GPU-23348268-6430-c539-b7e5-762583f50e91, 00000000:01:00.0, 100, 24576\n"

        def completed(stdout="", returncode=0, stderr=""):
            return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

        with mock.patch("core.gpu_topology.query_gpus", return_value=topology):
            for invalid_pid in ("+123", "1.5", "1e3", "-1"):
                with self.subTest(invalid_pid=invalid_pid):
                    process_output = (
                        "GPU-23348268-6430-c539-b7e5-762583f50e91, %s\n" % invalid_pid
                    )
                    with mock.patch(
                        "app.model_publication_tasks.subprocess.run",
                        side_effect=[completed(inventory), completed(process_output)],
                    ):
                        with self.assertRaisesRegex(PublicationError, "gpu_preflight_failed"):
                            publication_gpu_preflight([0], required_bytes=1)

    def test_preflight_rejects_protected_uuid_after_index_remap(self):
        from core.gpu_topology import GpuInfo
        from app.model_publication import PublicationError
        from app.model_publication_tasks import publication_gpu_preflight

        protected_uuid = "GPU-7eff453d-60f0-37ed-92c1-0aec341c497d"
        topology = [GpuInfo(index=0, mem_free_mib=24476, mem_total_mib=24576)]
        inventory = "0, %s, 00000000:81:00.0, 100, 24576\n" % protected_uuid

        def completed(stdout="", returncode=0, stderr=""):
            return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

        with mock.patch.dict(os.environ, {"MERGEKIT_PROTECTED_GPU_UUIDS": protected_uuid}, clear=False):
            with mock.patch("core.gpu_topology.query_gpus", return_value=topology):
                with mock.patch(
                    "app.model_publication_tasks.subprocess.run",
                    side_effect=[completed(inventory), completed()],
                ):
                    with self.assertRaisesRegex(PublicationError, "protected_gpu"):
                        publication_gpu_preflight([0], required_bytes=1)

    def test_preflight_uses_new_inventory_snapshot_not_topology_free_memory(self):
        from core.gpu_topology import GpuInfo
        from app.model_publication import PublicationError
        from app.model_publication_tasks import publication_gpu_preflight

        topology = [GpuInfo(index=0, mem_free_mib=24000, mem_total_mib=24576)]
        gpu_line = "0, GPU-23348268-6430-c539-b7e5-762583f50e91, 00000000:01:00.0, 24000, 24576\n"

        def completed(stdout="", returncode=0, stderr=""):
            return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

        with mock.patch("core.gpu_topology.query_gpus", return_value=topology):
            with mock.patch("app.model_publication_tasks.subprocess.run", side_effect=[completed(gpu_line), completed()]):
                with self.assertRaisesRegex(PublicationError, "insufficient_gpu_memory"):
                    publication_gpu_preflight([0], required_bytes=1024**3)

    def test_gpu_ids_are_exact_non_boolean_integers(self):
        from app.model_publication import PublicationError
        from app.model_publication_tasks import _normalize_gpu_ids

        for invalid in (True, 1.9, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(PublicationError, "gpu_selection_required"):
                    _normalize_gpu_ids([invalid])

    def test_functional_validation_scopes_environment_to_cancellable_child(self):
        from app.model_publication_tasks import validate_model_functionally

        process = mock.Mock()
        process.poll.return_value = 0
        process.returncode = 0
        process.communicate.return_value = ('{"status":"passed"}', "")
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        with mock.patch("app.model_publication_tasks.subprocess.Popen", return_value=process) as popen:
            result = validate_model_functionally(
                self.source,
                [
                    "GPU-3f409b14-e414-b97e-346b-5de726e75aaa",
                    "GPU-ddf96bec-7977-9a3c-8508-602946b44a56",
                ],
                {"aborted": False},
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"],
            "GPU-3f409b14-e414-b97e-346b-5de726e75aaa,GPU-ddf96bec-7977-9a3c-8508-602946b44a56",
        )
        self.assertEqual(popen.call_args.kwargs["env"]["MERGEKIT_CLI_SCRIPT"], "1")
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), original)
        self.assertEqual(
            popen.call_args.args[0][:2],
            [sys.executable, "-m"],
        )
        self.assertEqual(popen.call_args.args[0][2], "app.model_publication_tasks")

    def test_functional_validation_module_import_smoke_reaches_cuda_requirement(self):
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        env["MERGEKIT_CLI_SCRIPT"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""

        result = subprocess.run(
            [sys.executable, "-m", "app.model_publication_tasks", "--functional-validation", self.source],
            cwd=package_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("functional validation requires visible CUDA devices", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_vlm_cmmmu_reuses_local_evolution_helpers_without_lmms_eval(self):
        from app.model_publication_tasks import _functional_validation_worker

        inspection = SimpleNamespace(is_vlm=True)
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
            bfloat16=object(),
            Tensor=(),
            is_floating_point=lambda _value: False,
            no_grad=lambda: mock.MagicMock(__enter__=lambda *_args: None, __exit__=lambda *_args: None),
            device=lambda _value: _value,
        )
        model = mock.Mock(device="cuda", config=SimpleNamespace(image_token_id=None))
        model.generate.return_value = ["tokens"]
        processor = mock.Mock()
        processor.apply_chat_template.return_value = "prompt"
        processor.return_value = {}
        processor.batch_decode.side_effect = [["ok"], ["A"]]
        fake_transformers = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: processor), AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: model))
        fake_dataset = [{"question": "q", "option1": "a", "option2": "b", "answer": "A", "image_1": object()}]
        fake_datasets = SimpleNamespace(load_dataset=mock.Mock(return_value=fake_dataset))
        fake_fitness = SimpleNamespace(
            _cmmmu_first_image=lambda row: row["image_1"],
            _coerce_hf_image_to_pil=lambda image: image,
            _normalize_cmmmu_gold=lambda answer: answer,
            build_cmmmu_prompt=lambda _row, _config: "question",
            load_prompt_cfg=lambda *_args: {},
            parse_choice=lambda output: output,
        )

        with mock.patch("app.model_publication_tasks.inspect_model", return_value=inspection):
            with mock.patch.dict(sys.modules, {
                "torch": fake_torch,
                "transformers": fake_transformers,
                "datasets": fake_datasets,
                "evolution.vendor.vlm_merge.vlm_fitness": fake_fitness,
            }):
                with mock.patch("PIL.Image.new", return_value=object()):
                    with mock.patch("merge_manager.run_lmms_eval_stream", side_effect=AssertionError("lmms_eval must not be required")) as legacy_evaluate:
                        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
                            result = _functional_validation_worker(self.source)

        legacy_evaluate.assert_not_called()
        fake_datasets.load_dataset.assert_called_once_with(
            "m-a-p/CMMMU",
            "health_and_medicine",
            split="val",
            trust_remote_code=True,
            cache_dir=mock.ANY,
        )
        self.assertEqual(result["evaluation"]["cmmmu"]["samples"], 1)
        self.assertEqual(result["evaluation"]["cmmmu"]["acc"], 100.0)

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
                    inspection = SimpleNamespace(
                        path=self.current_source,
                        model_type="qwen2_5_vl",
                        architectures=("Qwen2_5_VLForConditionalGeneration",),
                        processor_class="Qwen2_5_VLProcessor",
                        image_token_ids={"image_token_id": 7},
                        visual_weight_count=3,
                        language_weight_count=4,
                        language_signature=(8, 1, 32),
                        config_sha256="a" * 64,
                    )
                    with mock.patch("app.model_inspection.resolve_vlm_base", return_value=inspection):
                        with mock.patch("evolution.vendor.vlm_merge.model_composition.materialize_full_vlm") as materialize:
                            provenance = _materialize_recipe("task-vlm", params, os.path.join(self.root, ".staging", "vlm"), self.progress, {})
        self.assertEqual(materialize.call_args.args[1], os.path.realpath(self.current_source))
        self.assertEqual(provenance["recipe_snapshot"], recipe_data)
        self.assertEqual(len(provenance["recipe_sha256"]), 64)
        self.assertEqual(provenance["parents"], [self.source, self.current_source])
        self.assertEqual(
            [item["source_path"] for item in provenance["parent_fingerprints"]],
            [os.path.realpath(self.source), os.path.realpath(self.current_source)],
        )
        self.assertTrue(all(len(item["weights_sha256"]) == 64 for item in provenance["parent_fingerprints"]))
        self.assertEqual(provenance["vlm_base"]["source_path"], os.path.realpath(self.current_source))
        self.assertEqual(provenance["vlm_base"]["visual_weight_count"], 3)
        self.assertEqual(len(provenance["vlm_base"]["weights_sha256"]), 64)

    def test_recipe_materialization_rejects_parent_weight_replacement(self):
        from app.model_publication import PublicationError
        from app.model_publication_tasks import _materialize_recipe

        recipe = os.path.join(self.tmpdir.name, "recipe.json")
        recipe_data = {
            "artifact_type": "text",
            "model_paths": [self.source, self.current_source],
            "best_genotype": [0.5, 0.5],
        }
        with open(recipe, "w", encoding="utf-8") as handle:
            json.dump(recipe_data, handle)

        def fingerprint(path, marker):
            return {
                "source_path": os.path.realpath(path),
                "weights_sha256": marker * 64,
                "weight_bytes": 7,
                "weight_files": [{
                    "path": "model.safetensors",
                    "size_bytes": 7,
                    "sha256": marker * 64,
                }],
            }

        fingerprints = [
            fingerprint(self.source, "a"),
            fingerprint(self.current_source, "b"),
            fingerprint(self.source, "a"),
            fingerprint(self.current_source, "c"),
        ]
        with self.app.app_context():
            with mock.patch("app.model_publication_tasks._resolve_recipe", return_value=(recipe, recipe_data)):
                with mock.patch("merge_manager.run_recipe_apply_task", return_value={"status": "success"}):
                    with mock.patch("app.model_publication_tasks.model_weight_fingerprint", side_effect=fingerprints):
                        with self.assertRaisesRegex(PublicationError, "source_fingerprint_mismatch"):
                            _materialize_recipe(
                                "task-text",
                                {"recipe_id": "recipe", "recipe_path": recipe},
                                os.path.join(self.root, ".staging", "text"),
                                self.progress,
                                {},
                            )

    def test_recipe_materialization_rejects_parent_changed_since_search(self):
        from app.model_publication import PublicationError
        from app.model_publication_tasks import _materialize_recipe

        recipe = os.path.join(self.tmpdir.name, "recipe.json")
        recorded = {
            "source_path": os.path.realpath(self.source),
            "weights_sha256": "a" * 64,
            "weight_bytes": 7,
            "weight_files": [{
                "path": "model.safetensors",
                "size_bytes": 7,
                "sha256": "a" * 64,
            }],
        }
        recipe_data = {
            "artifact_type": "text",
            "model_paths": [self.source],
            "parent_fingerprints": [recorded],
            "best_genotype": [1.0],
        }
        with open(recipe, "w", encoding="utf-8") as handle:
            json.dump(recipe_data, handle)
        changed = {**recorded, "weights_sha256": "b" * 64}

        with self.app.app_context():
            with mock.patch("app.model_publication_tasks._resolve_recipe", return_value=(recipe, recipe_data)):
                with mock.patch("app.model_publication_tasks.model_weight_fingerprint", return_value=changed):
                    with mock.patch("merge_manager.run_recipe_apply_task") as apply_recipe:
                        with self.assertRaisesRegex(PublicationError, "source_fingerprint_mismatch"):
                            _materialize_recipe(
                                "task-text",
                                {"recipe_id": "recipe", "recipe_path": recipe},
                                os.path.join(self.root, ".staging", "text"),
                                self.progress,
                                {},
                            )

        apply_recipe.assert_not_called()

    def test_recipe_provenance_is_persisted_before_explicit_validation(self):
        from app.models import Task
        from app.model_publication_tasks import run_model_publication_task

        request = {
            **self.request,
            "source_type": "recipe",
            "recipe_id": "recipe-a",
            "recipe_path": os.path.join(self.tmpdir.name, "recipe-a.json"),
        }
        provenance = {
            "recipe_sha256": "b" * 64,
            "recipe_snapshot": {"model_paths": [self.source], "vlm_base": {"source_path": self.current_source}},
            "parents": [self.source],
            "parent_fingerprints": [{"source_path": self.source, "weights_sha256": "c" * 64}],
            "vlm_base": {"source_path": self.current_source},
        }

        def materialize(_task_id, _params, staging, _progress, _control):
            self._write_model(staging)
            return provenance

        with self.app.app_context():
            self._add_task(config=request)
            with mock.patch("app.model_publication_tasks._recipe_model_paths", return_value=[self.source]):
                with mock.patch("app.model_publication_tasks._materialize_recipe", side_effect=materialize):
                    result = run_model_publication_task(
                        "task-a", request, self.progress, {"aborted": False},
                        structural_validate_fn=self.structural_validate,
                    )
            task = self.db.session.get(Task, "task-a")

        self.assertEqual(result["status"], "validating")
        for key, value in provenance.items():
            self.assertEqual(task.config[key], value)

    def test_recipe_apply_metadata_override_marks_publication_fallback(self):
        import merge_manager

        task_id = "recipe-publication-meta"
        recipe_id = "recipe-publication-meta"
        recipe_path = os.path.join(self.tmpdir.name, "%s.json" % recipe_id)
        with open(recipe_path, "w", encoding="utf-8") as handle:
            json.dump({
                "model_paths": [self.source, self.current_source],
                "best_genotype": [0.5, 0.5],
                "custom_name": "Recipe publication",
            }, handle)
        task_root = os.path.join(self.tmpdir.name, "merge-root")
        output_root = os.path.join(task_root, task_id, "output")
        publication_recipe_path = os.path.join(self.tmpdir.name, "secret-publication.json")
        old_merge_dir = merge_manager.MERGE_DIR
        old_recipes_dir = merge_manager.RECIPES_DIR
        merge_manager.MERGE_DIR = task_root
        merge_manager.RECIPES_DIR = self.tmpdir.name
        try:
            with mock.patch("merge_manager.subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="", stderr="failed")):
                merge_manager.run_recipe_apply_task(
                    task_id,
                    {"recipe_id": recipe_id},
                    self.progress,
                    output_dir_override=output_root,
                    metadata_type_override="model_publication",
                    metadata_extra={"publication_id": "publication-meta", "recipe_path": publication_recipe_path},
                )
            with open(os.path.join(task_root, task_id, "metadata.json"), encoding="utf-8") as handle:
                metadata = json.load(handle)
        finally:
            merge_manager.MERGE_DIR = old_merge_dir
            merge_manager.RECIPES_DIR = old_recipes_dir

        self.assertEqual(metadata["type"], "model_publication")
        self.assertEqual(metadata["publication_id"], "publication-meta")
        self.assertEqual(metadata["recipe_path"], publication_recipe_path)


if __name__ == "__main__":
    unittest.main()
