import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from app.model_inspection import ModelInspection
from evolution import runner


class VlmRecipeContractTest(unittest.TestCase):
    def setUp(self):
        self.inspection = ModelInspection(
            path="/models/complete-vlm",
            model_type="qwen2_5_vl",
            architectures=("Qwen2_5_VLForConditionalGeneration",),
            is_vlm=True,
            is_complete_vlm=True,
            processor_class="Qwen2_5_VLProcessor",
            image_token_ids={"image_token_id": 151655},
            visual_weight_count=4,
            language_weight_count=8,
            language_signature=(3584, 28, 152064),
            config_sha256="abc123",
        )

    def test_vlm_recipe_fields_are_additive_and_json_safe(self):
        meta = {"best_genotype": [0.2, 0.8], "custom_field": "kept"}

        enriched = runner.build_recipe_vlm_fields(meta, self.inspection)

        self.assertEqual(enriched["recipe_schema_version"], 2)
        self.assertEqual(enriched["artifact_type"], "vlm")
        self.assertEqual(enriched["capabilities"], ["text_generation", "vision_language"])
        self.assertEqual(enriched["vlm_path"], self.inspection.path)
        self.assertEqual(enriched["vlm_base"]["config_sha256"], self.inspection.config_sha256)
        self.assertIn("best_genotype", enriched)
        self.assertEqual(enriched["custom_field"], "kept")
        json.dumps(enriched)

    def test_vlm_recipe_preserves_historical_vlm_path(self):
        enriched = runner.build_recipe_vlm_fields(
            {"vlm_path": "/models/legacy-vlm"}, self.inspection
        )

        self.assertEqual(enriched["vlm_path"], "/models/legacy-vlm")
        self.assertEqual(enriched["vlm_base"]["source_path"], self.inspection.path)

    def test_text_recipe_has_text_contract_without_vlm_base(self):
        enriched = runner.build_recipe_vlm_fields({"best_genotype": [1.0]}, None)

        self.assertEqual(enriched["artifact_type"], "text")
        self.assertEqual(enriched["capabilities"], ["text_generation"])
        self.assertNotIn("vlm_base", enriched)

    def test_cmmmu_without_complete_vlm_fails_before_lock_or_subprocess(self):
        with tempfile.TemporaryDirectory() as merge_root:
            task_id = "missing-vlm"
            task_dir = os.path.join(merge_root, task_id)
            os.makedirs(task_dir)
            with open(os.path.join(task_dir, "metadata.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "id": task_id,
                        "model_paths": ["/models/text-a", "/models/text-b"],
                        "eval_mode": "text",
                        "hf_dataset": "m-a-p/CMMMU",
                    },
                    handle,
                )

            with mock.patch.object(runner, "MERGE_DIR", merge_root), mock.patch.object(
                runner, "_lock_file"
            ) as lock_file, mock.patch.object(runner.subprocess, "Popen") as popen, mock.patch.object(
                sys, "argv", ["runner.py", "--task-id", task_id]
            ):
                with self.assertRaisesRegex(ValueError, "vlm_base_missing"):
                    runner.main()

            lock_file.assert_not_called()
            popen.assert_not_called()

    def test_cmmmu_text_metadata_normalizes_before_lock_or_subprocess(self):
        meta = {
            "model_paths": ["/models/text-a", "/models/text-b"],
            "eval_mode": "text",
            "hf_dataset": "m-a-p/CMMMU",
            "vlm_path": "/models/legacy-vlm",
        }
        with mock.patch("app.model_inspection.resolve_vlm_base", return_value=self.inspection), mock.patch(
            "app.model_inspection.assert_language_compatible"
        ) as compatible, mock.patch.object(runner, "_lock_file") as lock_file, mock.patch.object(
            runner.subprocess, "Popen"
        ) as popen:
            vlm_mode, eval_mode, resolved_vlm_path, selected = runner.resolve_vlm_preflight(meta)

        self.assertTrue(vlm_mode)
        self.assertEqual(eval_mode, "vlm")
        self.assertEqual(resolved_vlm_path, self.inspection.path)
        self.assertIs(selected, self.inspection)
        compatible.assert_called_once_with(meta["model_paths"], self.inspection)
        lock_file.assert_not_called()
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
