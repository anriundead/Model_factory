"""
纯函数测试：_resolve_eval_dataset_cap（不加载 GPU / 模型）。
limit 在 (0,1] 为数据集比例（1.0=全量），>1 为绝对条数。
"""
import os
import sys
import unittest

# 项目根 mergeKit_beta
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from merge_manager import (  # noqa: E402
    _resolve_eval_dataset_cap,
    _should_fallback_single_gpu_for_limit,
)


class TestResolveEvalDatasetCap(unittest.TestCase):
    def test_n1000_frontend_and_scripts(self):
        self.assertEqual(_resolve_eval_dataset_cap(1000, "0.1"), 100)
        self.assertEqual(_resolve_eval_dataset_cap(1000, 0.1), 100)
        self.assertEqual(_resolve_eval_dataset_cap(1000, "0.5"), 500)
        self.assertEqual(_resolve_eval_dataset_cap(1000, "1.0"), 1000)
        self.assertEqual(_resolve_eval_dataset_cap(1000, 1.0), 1000)
        self.assertEqual(_resolve_eval_dataset_cap(1000, 1), 1000)
        self.assertEqual(_resolve_eval_dataset_cap(1000, "6"), 6)
        self.assertEqual(_resolve_eval_dataset_cap(1000, 6), 6)
        self.assertEqual(_resolve_eval_dataset_cap(1000, "100"), 100)
        self.assertEqual(_resolve_eval_dataset_cap(1000, 1500), 1000)

    def test_n3_small_dataset_at_least_one_when_fractional(self):
        self.assertEqual(_resolve_eval_dataset_cap(3, "0.1"), 1)
        self.assertEqual(_resolve_eval_dataset_cap(3, "0.5"), 1)
        self.assertEqual(_resolve_eval_dataset_cap(3, "1.0"), 3)
        self.assertEqual(_resolve_eval_dataset_cap(3, "6"), 3)

    def test_n_zero(self):
        self.assertEqual(_resolve_eval_dataset_cap(0, "0.5"), 0)

    def test_unparseable_defaults_half(self):
        self.assertEqual(_resolve_eval_dataset_cap(100, "bogus"), 50)

    def test_non_positive_parsed_defaults_half(self):
        self.assertEqual(_resolve_eval_dataset_cap(10, 0), 5)
        self.assertEqual(_resolve_eval_dataset_cap(10, -1), 5)


class TestShouldFallbackSingleGpuForLimit(unittest.TestCase):
    def test_ratio_full_no_fallback(self):
        self.assertFalse(_should_fallback_single_gpu_for_limit("1.0", 4))
        self.assertFalse(_should_fallback_single_gpu_for_limit(1.0, 4))

    def test_absolute_below_gpu_count(self):
        self.assertTrue(_should_fallback_single_gpu_for_limit("3", 4))
        self.assertTrue(_should_fallback_single_gpu_for_limit(3, 4))

    def test_absolute_not_below_gpu_count(self):
        self.assertFalse(_should_fallback_single_gpu_for_limit("6", 4))
        self.assertFalse(_should_fallback_single_gpu_for_limit(6, 4))

    def test_boundary_one_is_not_absolute_multi(self):
        # 1.0 是比例全量，不是「1 条绝对」
        self.assertFalse(_should_fallback_single_gpu_for_limit(1, 4))

    def test_single_gpu_never_fallback(self):
        self.assertFalse(_should_fallback_single_gpu_for_limit("3", 1))

    def test_unparseable_no_fallback(self):
        self.assertFalse(_should_fallback_single_gpu_for_limit("bogus", 4))


if __name__ == "__main__":
    unittest.main()
