import os
import tempfile
import unittest
from unittest import mock


class PublicationTaskTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmpdir.name, "published")
        self.source = os.path.join(self.tmpdir.name, "source")
        os.makedirs(self.source)
        with open(os.path.join(self.source, "weights.bin"), "wb") as handle:
            handle.write(b"weights")
        self.request = {
            "publication_id": "publication-a",
            "publication_root": self.root,
            "source_type": "existing_model",
            # Routes resolve model_id to this private worker-only field.
            "source_path": self.source,
            "display_name": "Published test model",
        }
        self.progress = lambda _percent, _message: None

    def tearDown(self):
        self.tmpdir.cleanup()

    def copy_model(self, source, destination):
        os.makedirs(destination)
        with open(os.path.join(source, "weights.bin"), "rb") as src:
            with open(os.path.join(destination, "weights.bin"), "wb") as dst:
                dst.write(src.read())

    def structural_validate(self, staging):
        self.assertTrue(os.path.isfile(os.path.join(staging, "weights.bin")))
        return {"status": "passed"}

    def test_without_gpu_stops_at_validating_without_registering(self):
        from app.model_publication_tasks import run_model_publication_task

        result = run_model_publication_task(
            "task-a", self.request, self.progress, {"aborted": False},
            copy_fn=self.copy_model, structural_validate_fn=self.structural_validate,
        )

        self.assertEqual(result["status"], "validating")
        self.assertFalse(os.path.exists(os.path.join(self.root, "publication-a")))
        self.assertTrue(os.path.isfile(os.path.join(self.root, ".staging", "publication-a", "weights.bin")))

    def test_cancel_before_commit_removes_staging(self):
        from app.model_publication_tasks import run_model_publication_task

        result = run_model_publication_task("task-b", self.request, self.progress, {"aborted": True})

        self.assertEqual(result["error_code"], "canceled")
        self.assertFalse(os.path.exists(os.path.join(self.root, ".staging", "publication-a")))

    def test_validate_requires_explicit_non_gpu2_ids(self):
        from app.model_publication import PublicationError
        from app.model_publication_tasks import run_publication_validation

        with self.assertRaisesRegex(PublicationError, "gpu_selection_required"):
            run_publication_validation("task-a", [], self.progress, {}, functional_validate_fn=lambda *_args: None)
        with self.assertRaisesRegex(PublicationError, "protected_gpu"):
            run_publication_validation("task-a", [2], self.progress, {}, functional_validate_fn=lambda *_args: None)

    def test_vlm_recipe_materializes_full_visual_model(self):
        from app.model_publication_tasks import _materialize_recipe

        recipe = os.path.join(self.tmpdir.name, "recipe.json")
        vlm_base = os.path.join(self.tmpdir.name, "vlm-base")
        os.makedirs(vlm_base)
        with open(recipe, "w", encoding="utf-8") as handle:
            handle.write('{"artifact_type":"vlm","vlm_path":"%s"}' % vlm_base)
        with mock.patch("merge_manager.run_recipe_apply_task", return_value={"status": "success"}) as apply_recipe:
            with mock.patch("evolution.vendor.vlm_merge.model_composition.materialize_full_vlm") as materialize:
                _materialize_recipe(
                    "task-vlm",
                    {"recipe_id": "recipe", "recipe_path": recipe},
                    os.path.join(self.root, ".staging", "publication-vlm"),
                    self.progress,
                    {},
                )
        self.assertTrue(apply_recipe.called)
        materialize.assert_called_once()
