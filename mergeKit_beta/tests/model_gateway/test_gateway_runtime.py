"""Serving runtime unit tests."""
import os
import shutil
import signal
import socket
import tempfile
import unittest
from unittest.mock import mock_open, patch

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class RuntimeTestCase(unittest.TestCase):
    def setUp(self):
        from flask import Flask
        from app.extensions import db
        from app import models as base_models  # noqa: F401
        from app.model_gateway import models as serving_models  # noqa: F401

        self.db = db
        self.app = Flask(__name__)
        self.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        self.app.config["SQLALCHEMY_BINDS"] = {"model_gateway": "sqlite:///:memory:"}
        self.app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
        db.init_app(self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        self.db.session.remove()
        self.db.drop_all()
        self.ctx.pop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_model_dir(self):
        path = os.path.join(self.tmp, "qwen-merged")
        os.makedirs(path)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        with open(os.path.join(path, "model.safetensors"), "w", encoding="utf-8") as f:
            f.write("placeholder")
        return path


class TestRuntimeValidation(RuntimeTestCase):
    def test_safe_default_max_model_len_is_bounded_by_service_type(self):
        from app.model_gateway.runtime import safe_default_max_model_len

        self.assertEqual(safe_default_max_model_len("text"), 65536)
        self.assertEqual(safe_default_max_model_len("vlm"), 16384)
        self.assertEqual(safe_default_max_model_len("unknown"), 16384)

    def test_pid_alive_treats_a_zombie_process_as_exited(self):
        from app.model_gateway.runtime import _pid_alive

        with patch("app.model_gateway.runtime.os.kill"), \
            patch("builtins.open", mock_open(read_data="4321 (vllm) Z 69 4321 4321 0")):
            self.assertFalse(_pid_alive(4321))

    def test_only_runtime_process_runs_startup_recovery(self):
        from app.model_gateway.runtime import should_recover_services_on_start

        with patch.dict(os.environ, {"MERGEKIT_MODEL_GATEWAY_RUNTIME_PROCESS": ""}):
            self.assertFalse(should_recover_services_on_start())
        with patch.dict(os.environ, {"MERGEKIT_MODEL_GATEWAY_RUNTIME_PROCESS": "1"}):
            self.assertTrue(should_recover_services_on_start())

    def test_validate_model_path_accepts_hf_model_under_allowed_root(self):
        from app.model_gateway.runtime import validate_model_path

        model_path = self.make_model_dir()

        validate_model_path(model_path, [self.tmp])

    def test_validate_model_path_rejects_path_outside_allowed_roots(self):
        from app.model_gateway.runtime import validate_model_path

        model_path = self.make_model_dir()

        with self.assertRaises(ValueError):
            validate_model_path(model_path, [os.path.join(self.tmp, "other")])

    def test_allowed_model_roots_includes_published_models(self):
        from app.model_gateway.runtime import allowed_model_roots

        class Config:
            MODEL_POOL_PATH = "/models"
            LOCAL_MODELS_PATH = "/local"
            MERGE_DIR = "/merges"
            PUBLISHED_MODELS_PATH = "/published"
            LOCAL_MODELS_EXTRA_PATHS = []

        self.assertIn("/published", allowed_model_roots(Config))

    def test_find_free_port_skips_reserved_and_listening_ports(self):
        from app.model_gateway.runtime import find_free_port

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy_port = sock.getsockname()[1]
        try:
            selected = find_free_port(busy_port, busy_port + 2, {busy_port + 1})
        finally:
            sock.close()

        self.assertEqual(selected, busy_port + 2)

    def test_validate_gpu_availability_rejects_selected_gpu_with_existing_heavy_use(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import validate_gpu_availability
        from core.gpu_topology import GpuInfo

        service = ServingModelService(
            model_path="/models/qwen",
            display_name="Qwen demo",
            served_model_name="qwen-demo",
            gpu_ids=[2],
        )

        with self.assertRaisesRegex(ValueError, "already has"):
            validate_gpu_availability(
                service,
                query_fn=lambda: [GpuInfo(index=2, mem_free_mib=7000, mem_total_mib=24576)],
                max_used_mib=1024,
            )

    def test_start_timeout_escalates_to_sigkill_before_marking_failed(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import start_service

        model_path = self.make_model_dir()
        service = ServingModelService(
            model_path=model_path,
            display_name="Qwen demo",
            served_model_name="qwen-demo",
            gpu_ids=[],
            vllm_port=18001,
            internal_api_key="internal-secret",
        )
        self.db.session.add(service)
        self.db.session.commit()

        class Config:
            PROJECT_ROOT = self.tmp
            MODEL_POOL_PATH = self.tmp
            LOCAL_MODELS_PATH = self.tmp
            MERGE_DIR = self.tmp
            LOCAL_MODELS_EXTRA_PATHS = []
            MERGEKIT_MODEL_GATEWAY_LOG_DIR = self.tmp
            MERGEKIT_MODEL_GATEWAY_PYTHON = "python"

        class FakeProc:
            pid = 12345

            def poll(self):
                return None

        sent = []

        with patch("app.model_gateway.runtime.subprocess.Popen", return_value=FakeProc()), \
            patch("app.model_gateway.runtime.os.getpgid", return_value=12345), \
            patch("app.model_gateway.runtime._healthcheck", return_value=False), \
            patch("app.model_gateway.runtime._kill_process_group", side_effect=lambda pgid, sig: sent.append(sig)), \
            patch("app.model_gateway.runtime._pid_alive", side_effect=[True, True, False]), \
            patch("app.model_gateway.runtime.time.sleep"), \
            patch("app.model_gateway.runtime.time.time", side_effect=[0, 2, 2, 3, 13]):
            result = start_service(service.id, config=Config, timeout_s=1)

        self.assertEqual(result.status, "failed")
        self.assertIn(signal.SIGTERM, sent)
        self.assertIn(signal.SIGKILL, sent)


class TestRuntimeCommand(RuntimeTestCase):
    def test_vllm_kv_cache_failure_is_explained_without_persisting_log_content(self):
        from app.model_gateway.runtime import describe_vllm_exit

        log_path = os.path.join(self.tmp, "vllm.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(
                "ValueError: The model's max seq len (128000) is larger than "
                "the maximum number of tokens that can be stored in KV cache (102368)."
            )

        message = describe_vllm_exit(log_path, 1)

        self.assertEqual(
            message,
            "vLLM exited early: configured context length exceeds available KV cache; "
            "lower max_model_len or increase GPU memory utilization",
        )

    def test_build_vllm_command_uses_loopback_and_whitelisted_args(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import build_vllm_command

        service = ServingModelService(
            model_path="/models/qwen",
            display_name="Qwen demo",
            served_model_name="qwen-demo",
            vllm_port=18001,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            dtype="auto",
            max_model_len=4096,
            max_num_seqs=8,
            max_num_batched_tokens=None,
            trust_remote_code=True,
            internal_api_key="internal-secret",
        )

        class Config:
            MERGEKIT_MODEL_GATEWAY_VLLM_BIN = "/opt/conda/envs/mergenetic/bin/vllm"
            MERGEKIT_MODEL_GATEWAY_VLLM_USE_COMPAT_WRAPPER = False

        cmd = build_vllm_command(service, Config)

        self.assertEqual(cmd[:3], ["/opt/conda/envs/mergenetic/bin/vllm", "serve", "/models/qwen"])
        self.assertIn("--host", cmd)
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
        self.assertEqual(cmd[cmd.index("--api-key") + 1], "internal-secret")
        self.assertIn("--disable-log-requests", cmd)
        self.assertIn("--enable-request-id-headers", cmd)
        self.assertIn("--trust-remote-code", cmd)
        self.assertNotIn("--max-num-batched-tokens", cmd)

    def test_build_vllm_command_uses_compat_wrapper_by_default(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import build_vllm_command

        service = ServingModelService(
            model_path="/models/qwen",
            display_name="Qwen demo",
            served_model_name="qwen-demo",
            vllm_port=18001,
            internal_api_key="internal-secret",
        )

        class Config:
            MERGEKIT_MODEL_GATEWAY_PYTHON = "/opt/conda/envs/mergenetic/bin/python"
            MERGEKIT_MODEL_GATEWAY_VLLM_BIN = "/opt/conda/envs/mergenetic/bin/vllm"

        cmd = build_vllm_command(service, Config)

        self.assertEqual(cmd[:5], [
            "/opt/conda/envs/mergenetic/bin/python",
            "-m",
            "app.model_gateway.vllm_entrypoint",
            "serve",
            "/models/qwen",
        ])

    def test_vllm_entrypoint_patches_transformers_5_tokenizer_property(self):
        from transformers import PreTrainedTokenizerBase
        from app.model_gateway.vllm_entrypoint import _patch_transformers_tokenizers

        _patch_transformers_tokenizers()

        self.assertTrue(hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"))

    def test_vllm_entrypoint_private_abort_route_authenticates_and_aborts(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from vllm.entrypoints.openai import api_server
        from app.model_gateway.vllm_entrypoint import _install_internal_abort_endpoint

        class Engine:
            def __init__(self):
                self.request_ids = []

            async def abort(self, request_id):
                self.request_ids.append(request_id)

        engine = Engine()
        app = FastAPI()
        app.state.engine_client = engine

        with patch.dict(os.environ, {"MERGEKIT_MODEL_GATEWAY_INTERNAL_API_KEY": "internal-secret"}), \
            patch.object(api_server, "build_app", return_value=app), \
            patch.object(api_server, "_model_gateway_abort_installed", False, create=True):
            _install_internal_abort_endpoint()
            client = TestClient(api_server.build_app(None))
            rejected = client.post("/internal/model-gateway/abort/chatcmpl-request")
            accepted = client.post(
                "/internal/model-gateway/abort/chatcmpl-request",
                headers={"Authorization": "Bearer internal-secret"},
            )

        self.assertEqual(rejected.status_code, 401)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(engine.request_ids, ["chatcmpl-request"])

    def test_vllm_subprocess_pythonpath_does_not_shadow_standard_library(self):
        from app.model_gateway.runtime import _with_model_gateway_pythonpath

        class Config:
            PROJECT_ROOT = "/app/project"

        env = _with_model_gateway_pythonpath({"PYTHONPATH": "/old"}, Config)

        self.assertEqual(env["PYTHONPATH"], "/app/project/app/model_gateway/runtime_hooks:/old")

    def test_vllm_subprocess_is_marked_as_cli_to_skip_runtime_recovery(self):
        from app.model_gateway.runtime import _with_model_gateway_pythonpath

        class Config:
            PROJECT_ROOT = "/app/project"

        env = _with_model_gateway_pythonpath({}, Config)

        self.assertEqual(env["MERGEKIT_CLI_SCRIPT"], "1")


class TestRestartRecovery(RuntimeTestCase):
    def test_stop_service_recovers_marked_process_when_persisted_pid_is_missing(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import stop_service

        service = ServingModelService(
            model_path="/models/qwen",
            display_name="Qwen demo",
            served_model_name="qwen-stop-fallback",
            status="running",
            vllm_pid=None,
            vllm_pgid=None,
        )
        self.db.session.add(service)
        self.db.session.commit()

        with patch("app.model_gateway.runtime._find_marked_service_pids", side_effect=[[4321, 4322], []]), \
            patch("app.model_gateway.runtime._terminate_service_processes") as terminate, \
            patch("app.model_gateway.runtime._pid_alive", return_value=False):
            result = stop_service(service.id)

        terminate.assert_called_once_with([4321, 4322], timeout_s=30)
        self.assertEqual(result.status, "stopped")

    def test_restart_recovery_marks_runtime_states_stopped(self):
        from app.model_gateway.models import ServingModelService
        from app.model_gateway.runtime import mark_services_stopped_after_restart

        running = ServingModelService(
            model_path="/models/qwen-a",
            display_name="A",
            served_model_name="qwen-a",
            status="running",
        )
        starting = ServingModelService(
            model_path="/models/qwen-b",
            display_name="B",
            served_model_name="qwen-b",
            status="starting",
        )
        failed = ServingModelService(
            model_path="/models/qwen-c",
            display_name="C",
            served_model_name="qwen-c",
            status="failed",
        )
        self.db.session.add_all([running, starting, failed])
        self.db.session.commit()

        changed = mark_services_stopped_after_restart(self.db.session)

        self.assertEqual(changed, 2)
        self.assertEqual(running.status, "stopped")
        self.assertEqual(starting.status, "stopped")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(running.last_exit_reason, "system_restarted_manual_recovery_required")

    def test_restart_recovery_terminates_inflight_requests_without_replaying_them(self):
        from app.model_gateway.models import ServingRequest
        from app.model_gateway.runtime import recover_inflight_requests_after_restart

        running = ServingRequest(served_model_name="qwen-demo", status="running")
        streaming = ServingRequest(served_model_name="qwen-demo", status="streaming")
        cancel_requested = ServingRequest(served_model_name="qwen-demo", status="cancel_requested")
        queued = ServingRequest(served_model_name="qwen-demo", status="queued")
        self.db.session.add_all([running, streaming, cancel_requested, queued])
        self.db.session.commit()

        changed = recover_inflight_requests_after_restart(self.db.session)

        self.assertEqual(changed, 3)
        self.assertEqual(running.status, "failed")
        self.assertEqual(running.error_code, "request_interrupted_by_restart")
        self.assertIsNotNone(running.finished_at)
        self.assertEqual(streaming.status, "failed")
        self.assertEqual(streaming.error_code, "stream_interrupted_by_restart")
        self.assertEqual(cancel_requested.status, "canceled")
        self.assertEqual(cancel_requested.error_code, "canceled_by_restart")
        self.assertIsNotNone(cancel_requested.finished_at)
        self.assertEqual(queued.status, "queued")


if __name__ == "__main__":
    unittest.main()
