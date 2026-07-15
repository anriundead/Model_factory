"""
计划 §4.3：在无完整 mergenetic 环境时，用 mock 捕获 run_lm_eval_stream / run_lmms_eval_stream
拼出的命令与 eval_full_output.log 路径，验证 limit/多卡语义。

真实多卡 + Hub 评测仍需在目标机器上执行。
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _install_fake_lm_eval():
    if "lm_eval.tasks" in sys.modules:
        return
    tasks_mod = types.ModuleType("lm_eval.tasks")

    class TaskManager:
        all_groups: list = []

        def load(self, name):
            return {"task": name}

        def load_task_or_group(self, name):
            return {"task": name}

    tasks_mod.TaskManager = TaskManager
    lm_eval_pkg = types.ModuleType("lm_eval")
    lm_eval_pkg.tasks = tasks_mod
    sys.modules["lm_eval"] = lm_eval_pkg
    sys.modules["lm_eval.tasks"] = tasks_mod


class MockLmEvalProc:
    """满足 run_lm_eval_stream 读 stdout / wait / returncode。"""

    returncode = 0

    def __init__(self, output_path: str, task_key: str):
        self.stdout = io.StringIO("")
        self._output_path = output_path
        self._task_key = task_key

    def poll(self):
        return 0

    def wait(self):
        rp = os.path.join(self._output_path, "results.json")
        with open(rp, "w", encoding="utf-8") as f:
            json.dump({"results": {self._task_key: {"acc,none": 0.5, "n_samples": 1}}}, f)
        return 0


class MockLmmsProc:
    returncode = 0

    def __init__(self):
        self.stdout = io.StringIO("")

    def poll(self):
        return 0

    def wait(self):
        return 0


class TestLimitPlanLmEvalCmdSmoke(unittest.TestCase):
    def setUp(self):
        _install_fake_lm_eval()
        import config

        self._eval_cache = tempfile.mkdtemp(prefix="eval_cache_")
        self._p_cache = patch.object(config.Config, "EVAL_HF_DATASETS_CACHE", new=self._eval_cache)
        self._p_cache.start()

        import merge_manager as mm

        self.mm = mm
        self._td = tempfile.TemporaryDirectory()
        self.model_path = self._td.name
        self.out = tempfile.mkdtemp(prefix="eval_out_")

    def tearDown(self):
        self._p_cache.stop()
        import shutil

        shutil.rmtree(self._eval_cache, ignore_errors=True)
        self._td.cleanup()
        shutil.rmtree(self.out, ignore_errors=True)

    def _run_llm(self, limit, num_gpus: int):
        captured = []

        def fake_popen(cmd, **kwargs):
            captured.append(list(cmd))
            return MockLmEvalProc(self.out, "hellaswag")

        with patch.object(self.mm.subprocess, "Popen", side_effect=fake_popen):
            with patch.object(self.mm.subprocess, "check_call", lambda *a, **k: None):
                with patch.object(
                    self.mm.subprocess,
                    "run",
                    return_value=MagicMock(stdout="GPU0\nGPU1\nGPU2\nGPU3\n", returncode=0),
                ):
                    with patch.object(self.mm, "_get_available_gpus", return_value=[0, 1, 2, 3]):
                        with patch.object(self.mm, "_estimate_model_vram_mib", return_value=1000):
                            with patch.object(self.mm, "_get_conda_activate_cmd", side_effect=lambda c: c):
                                with patch.object(self.mm, "_prepare_custom_eval_task", return_value=(None, None)):
                                    with patch.object(self.mm, "_popen_group_kwargs", return_value={}):
                                        cb = lambda p, m: None
                                        tc = {}
                                        self.mm.run_lm_eval_stream(
                                            self.model_path,
                                            self.out,
                                            "hellaswag",
                                            cb,
                                            0,
                                            100,
                                            task_control=tc,
                                            limit=limit,
                                            num_gpus=num_gpus,
                                        )
        log_path = os.path.join(self.out, "eval_full_output.log")
        return captured, log_path

    def test_llm_1_0_string_multigpu_accelerate_no_limit_flag(self):
        captured, log_path = self._run_llm("1.0", num_gpus=4)
        self.assertEqual(len(captured), 1)
        cmd = captured[0]
        self.assertIn("accelerate", cmd[0])
        self.assertNotIn("--limit", cmd)
        self.assertTrue(os.path.isfile(log_path), "应有 eval_full_output.log")

    def test_llm_int_1_multigpu_no_limit_flag(self):
        captured, log_path = self._run_llm(1, num_gpus=4)
        cmd = captured[0]
        self.assertIn("accelerate", cmd[0])
        self.assertNotIn("--limit", cmd)
        self.assertTrue(os.path.isfile(log_path))

    def test_llm_absolute_3_fallback_single_gpu(self):
        captured, log_path = self._run_llm("3", num_gpus=4)
        cmd = captured[0]
        self.assertEqual(os.path.basename(cmd[0]), "lm_eval")
        self.assertIn("--limit", cmd)
        self.assertIn("3", cmd)
        self.assertTrue(os.path.isfile(log_path))

    def test_llm_absolute_6_multigpu_has_limit_6(self):
        captured, log_path = self._run_llm("6", num_gpus=4)
        cmd = captured[0]
        self.assertIn("accelerate", cmd[0])
        idx = cmd.index("--limit")
        self.assertEqual(cmd[idx + 1], "6")
        self.assertTrue(os.path.isfile(log_path))


class TestLimitPlanLmmsCliSmoke(unittest.TestCase):
    def setUp(self):
        _install_fake_lm_eval()
        import merge_manager as mm

        self.mm = mm
        self._td = tempfile.TemporaryDirectory()
        self.model_path = self._td.name
        self.out = tempfile.mkdtemp(prefix="lmms_out_")

        ds_mod = types.ModuleType("datasets")

        def ok_load(*a, **kw):
            class _D:
                def __len__(self):
                    return 100

            return _D()

        ds_mod.load_dataset = ok_load
        sys.modules["datasets"] = ds_mod

    def tearDown(self):
        sys.modules.pop("datasets", None)
        self._td.cleanup()
        try:
            import shutil

            shutil.rmtree(self.out, ignore_errors=True)
        except Exception:
            pass

    def test_lmms_load_dataset_success_integer_limit_in_cmd(self):
        captured = []

        def fake_popen(cmd, **kwargs):
            captured.append(list(cmd))
            with open(os.path.join(self.out, "zoo_results.json"), "w", encoding="utf-8") as f:
                json.dump({"results": {"zoo": {"acc": 0.5, "n_samples": 1}}}, f)
            return MockLmmsProc()

        with patch.object(self.mm.subprocess, "Popen", side_effect=fake_popen):
            with patch.object(self.mm, "_popen_group_kwargs", return_value={}):
                cb = lambda p, m: None
                tc = {}
                self.mm.run_lmms_eval_stream(
                    self.model_path,
                    self.out,
                    cb,
                    0,
                    100,
                    task_control=tc,
                    limit="0.5",
                    hf_dataset="dummy/zoo",
                    hf_subset="sub",
                    hf_split="validation",
                    num_gpus=1,
                )
        self.assertEqual(len(captured), 1)
        cmd = captured[0]
        self.assertIn("--limit", cmd)
        i = cmd.index("--limit")
        self.assertEqual(cmd[i + 1], "50")

    def test_lmms_load_dataset_failure_keeps_original_limit_str(self):
        import types as _t

        ds_bad = _t.ModuleType("datasets")

        def boom(*a, **kw):
            raise RuntimeError("no hub")

        ds_bad.load_dataset = boom
        sys.modules["datasets"] = ds_bad

        captured = []

        def fake_popen(cmd, **kwargs):
            captured.append(list(cmd))
            with open(os.path.join(self.out, "zoo_results.json"), "w", encoding="utf-8") as f:
                json.dump({"results": {"zoo": {"acc": 0.5, "n_samples": 1}}}, f)
            return MockLmmsProc()

        with patch.object(self.mm.subprocess, "Popen", side_effect=fake_popen):
            with patch.object(self.mm, "_popen_group_kwargs", return_value={}):
                cb = lambda p, m: None
                tc = {}
                self.mm.run_lmms_eval_stream(
                    self.model_path,
                    self.out,
                    cb,
                    0,
                    100,
                    task_control=tc,
                    limit="0.25",
                    hf_dataset="dummy/zoo",
                    hf_subset="sub",
                    hf_split="validation",
                    num_gpus=1,
                )
        cmd = captured[0]
        i = cmd.index("--limit")
        self.assertEqual(cmd[i + 1], "0.25")


if __name__ == "__main__":
    unittest.main()
