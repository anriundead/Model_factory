import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evolution.vendor.vlm_merge.run_vlm_search import _resolve_best_genotype


class VlmSearchPostprocessTest(unittest.TestCase):
    def test_uses_problem_best_after_max_eval_early_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            best = _resolve_best_genotype(
                Path(tmpdir),
                SimpleNamespace(best_x=[0.2, 0.8]),
                result_x=None,
                result_f=None,
            )

        self.assertEqual(best, [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()
