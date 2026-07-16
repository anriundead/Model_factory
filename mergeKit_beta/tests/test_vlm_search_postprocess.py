import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from evolution.runner import _do_success_path
from evolution.vendor.vlm_merge.run_vlm_search import (
    _materialize_best_model,
    _resolve_best_genotype,
)


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

    def test_required_final_model_rejects_missing_best_genotype(self):
        merger = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "best genotype"):
            _materialize_best_model(
                merger,
                best_x=None,
                final_output="/tmp/required-final-model",
            )

        merger.create_individual_configuration.assert_not_called()

    def test_required_final_model_propagates_move_failure(self):
        merger = mock.Mock()
        merger.create_individual_configuration.return_value = {"models": []}
        merger.merge_model_from_configuration.return_value = "/tmp/generated-model"

        with mock.patch("evolution.vendor.vlm_merge.run_vlm_search.shutil.move", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                _materialize_best_model(
                    merger,
                    best_x=[0.2, 0.8],
                    final_output="/tmp/required-final-model",
                )

    def test_runner_does_not_mark_missing_final_model_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = os.path.join(tmpdir, "metadata.json")
            progress_path = os.path.join(tmpdir, "progress.json")
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump({"id": "task-a", "status": "running", "model_paths": []}, handle)
            with open(progress_path, "w", encoding="utf-8") as handle:
                json.dump({"status": "running"}, handle)

            with self.assertRaisesRegex(RuntimeError, "final_vlm_output"):
                _do_success_path(
                    merge_dir=tmpdir,
                    final_vlm_output=os.path.join(tmpdir, "final_vlm"),
                    meta_path=meta_path,
                    progress_path=progress_path,
                    task_id="task-a",
                    hf_split_final="",
                    hf_split="val",
                    logger=logging.getLogger("test"),
                )

            with open(meta_path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["status"], "running")


if __name__ == "__main__":
    unittest.main()
