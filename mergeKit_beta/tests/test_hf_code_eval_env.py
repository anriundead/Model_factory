"""merge_manager: lm_eval 子进程 HF_ALLOW_CODE_EVAL 注入逻辑。"""
import os
import unittest

from merge_manager import (
    _ensure_hf_allow_code_eval_for_lm_eval,
    _lm_eval_tasks_need_hf_allow_code_eval,
)


class TestLmEvalCodeEvalDetection(unittest.TestCase):
    def test_need_code_eval_humaneval(self):
        self.assertTrue(
            _lm_eval_tasks_need_hf_allow_code_eval("humaneval", ["humaneval"])
        )

    def test_need_code_eval_mbpp(self):
        self.assertTrue(_lm_eval_tasks_need_hf_allow_code_eval("mbpp", ["mbpp"]))

    def test_need_mixed_tasks(self):
        self.assertTrue(
            _lm_eval_tasks_need_hf_allow_code_eval("arc_easy,mbpp", ["arc_easy", "mbpp"])
        )

    def test_no_need_mmlu(self):
        self.assertFalse(
            _lm_eval_tasks_need_hf_allow_code_eval("mmlu_college_biology", ["mmlu_college_biology"])
        )


class TestEnsureHfAllowCodeEval(unittest.TestCase):
    def test_always_sets_for_non_code_task_when_not_forbid(self):
        env = {"PATH": os.environ.get("PATH", "")}
        _ensure_hf_allow_code_eval_for_lm_eval(env, "hellaswag", ["hellaswag"])
        self.assertEqual(env.get("HF_ALLOW_CODE_EVAL"), "1")

    def test_forbid_with_code_task_raises(self):
        env = os.environ.copy()
        env.pop("HF_ALLOW_CODE_EVAL", None)
        old = os.environ.get("MERGEKIT_FORBID_CODE_EVAL")
        old_hf = os.environ.get("HF_ALLOW_CODE_EVAL")
        try:
            os.environ["MERGEKIT_FORBID_CODE_EVAL"] = "1"
            os.environ.pop("HF_ALLOW_CODE_EVAL", None)
            with self.assertRaises(ValueError):
                _ensure_hf_allow_code_eval_for_lm_eval(env, "mbpp", ["mbpp"])
        finally:
            if old is None:
                os.environ.pop("MERGEKIT_FORBID_CODE_EVAL", None)
            else:
                os.environ["MERGEKIT_FORBID_CODE_EVAL"] = old
            if old_hf is None:
                os.environ.pop("HF_ALLOW_CODE_EVAL", None)
            else:
                os.environ["HF_ALLOW_CODE_EVAL"] = old_hf

    def test_forbid_without_code_task_no_inject(self):
        env = {"PATH": os.environ.get("PATH", "")}
        old = os.environ.get("MERGEKIT_FORBID_CODE_EVAL")
        old_hf = os.environ.get("HF_ALLOW_CODE_EVAL")
        try:
            os.environ["MERGEKIT_FORBID_CODE_EVAL"] = "1"
            os.environ.pop("HF_ALLOW_CODE_EVAL", None)
            _ensure_hf_allow_code_eval_for_lm_eval(env, "hellaswag", ["hellaswag"])
            self.assertNotIn("HF_ALLOW_CODE_EVAL", env)
        finally:
            if old is None:
                os.environ.pop("MERGEKIT_FORBID_CODE_EVAL", None)
            else:
                os.environ["MERGEKIT_FORBID_CODE_EVAL"] = old
            if old_hf is None:
                os.environ.pop("HF_ALLOW_CODE_EVAL", None)
            else:
                os.environ["HF_ALLOW_CODE_EVAL"] = old_hf

    def test_explicit_hf_zero_skips_inject(self):
        env = {"PATH": os.environ.get("PATH", "")}
        old = os.environ.get("MERGEKIT_FORBID_CODE_EVAL")
        old_hf = os.environ.get("HF_ALLOW_CODE_EVAL")
        try:
            os.environ.pop("MERGEKIT_FORBID_CODE_EVAL", None)
            os.environ["HF_ALLOW_CODE_EVAL"] = "0"
            _ensure_hf_allow_code_eval_for_lm_eval(env, "hellaswag", ["hellaswag"])
            self.assertNotIn("HF_ALLOW_CODE_EVAL", env)
        finally:
            if old is None:
                os.environ.pop("MERGEKIT_FORBID_CODE_EVAL", None)
            else:
                os.environ["MERGEKIT_FORBID_CODE_EVAL"] = old
            if old_hf is None:
                os.environ.pop("HF_ALLOW_CODE_EVAL", None)
            else:
                os.environ["HF_ALLOW_CODE_EVAL"] = old_hf


if __name__ == "__main__":
    unittest.main()
