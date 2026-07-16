import json
import os
import sys
import tempfile
import time
import types
import unittest
from types import SimpleNamespace
from unittest import mock


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from app.model_inspection import inspect_model  # noqa: E402
from app.model_publication import (  # noqa: E402
    PublicationError,
    build_manifest,
    commit_staging,
    delete_published_asset,
    inspect_serving_compatibility,
    reconcile_publications,
    validate_published_asset,
)


class PublicationFilesystemTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmpdir.name, "published")
        self.publication_id = "publication-001"
        self.staging = os.path.join(self.root, ".staging", self.publication_id)
        self._write_model(self.staging)
        self.request = {
            "publication_id": self.publication_id,
            "display_name": "Published test model",
            "task_id": "task-001",
            "recipe_path": "recipes/task-001.json",
            "recipe_sha256": "recipe-sha",
            "recipe_snapshot": {"models": ["parent-a"]},
            "parents": ["parent-a"],
            "dtype": "bfloat16",
        }
        self.inspection = inspect_model(self.staging)
        self.registered = []

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_model(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model_type": "qwen2",
                    "architectures": ["Qwen2ForCausalLM"],
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "vocab_size": 32,
                },
                handle,
            )
        with open(os.path.join(path, "model.safetensors"), "wb") as handle:
            handle.write(b"weights")
        with open(os.path.join(path, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
            json.dump({"weight_map": {"model.weight": "model.safetensors"}}, handle)
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as handle:
            json.dump({"version": "1.0"}, handle)

    def _manifest(self):
        return build_manifest(
            self.staging,
            self.request,
            self.inspection,
            validation={"structural": {"status": "passed"}},
            compatibility={"serving": {"status": "ready", "tested_version": "0.7.0"}},
        )

    def _register(self, path, manifest):
        self.registered.append((os.path.realpath(path), manifest["publication_id"]))

    def test_commit_is_atomic_and_manifest_excludes_itself_from_hashes(self):
        committed = commit_staging(self.staging, self.root, self._manifest(), register_fn=self._register)

        final = os.path.join(self.root, committed["publication_id"])
        self.assertFalse(os.path.exists(self.staging))
        self.assertTrue(os.path.isfile(os.path.join(final, "publication_manifest.json")))
        self.assertEqual(committed["publication_state"], "published")
        paths = [item["path"] for item in committed["files"]["entries"]]
        self.assertNotIn("publication_manifest.json", paths)
        self.assertEqual(paths, sorted(paths))
        self.assertEqual(len(self.registered), 1)

    def test_registration_failure_leaves_valid_pending_asset_for_recovery(self):
        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            commit_staging(
                self.staging,
                self.root,
                self._manifest(),
                register_fn=mock.Mock(side_effect=RuntimeError("database unavailable")),
            )

        final = os.path.join(self.root, self.publication_id)
        manifest = validate_published_asset(final, full_hash=True)
        self.assertEqual(manifest["publication_state"], "registration_pending")

    def test_reconcile_registers_pending_once(self):
        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            commit_staging(
                self.staging,
                self.root,
                self._manifest(),
                register_fn=mock.Mock(side_effect=RuntimeError("database unavailable")),
            )

        with mock.patch("app.model_publication.inspect_serving_compatibility", return_value={"status": "ready"}):
            first = reconcile_publications(self.root, register_fn=self._register)
            second = reconcile_publications(self.root, register_fn=self._register)

        self.assertEqual(first["registered"], 1)
        self.assertEqual(second["registered"], 0)
        self.assertEqual(len(self.registered), 1)

    def test_invalid_hash_is_quarantined(self):
        committed = commit_staging(self.staging, self.root, self._manifest(), register_fn=self._register)
        final = os.path.join(self.root, committed["publication_id"])
        with open(os.path.join(final, "model.safetensors"), "ab") as handle:
            handle.write(b"modified")

        with self.assertRaisesRegex(PublicationError, "validation_failed"):
            validate_published_asset(final, full_hash=True)

        reconcile_publications(self.root, register_fn=self._register)
        self.assertFalse(os.path.exists(final))
        self.assertTrue(os.path.isdir(os.path.join(self.root, ".quarantine")))

    def test_manifest_rejects_non_object_file_entries(self):
        committed = commit_staging(self.staging, self.root, self._manifest(), register_fn=self._register)
        final = os.path.join(self.root, committed["publication_id"])
        manifest_path = os.path.join(final, "publication_manifest.json")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["files"]["entries"].append("not-an-entry")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

        with self.assertRaisesRegex(PublicationError, "validation_failed"):
            validate_published_asset(final, full_hash=True)

    def test_symlinks_are_rejected_from_full_inventory(self):
        os.symlink("model.safetensors", os.path.join(self.staging, "linked-weights"))

        with self.assertRaisesRegex(PublicationError, "validation_failed"):
            self._manifest()

    def test_stale_staging_cleanup_writes_diagnostic_before_deletion(self):
        stale = os.path.join(self.root, ".staging", "stale-publication")
        self._write_model(stale)
        old = time.time() - 2 * 24 * 60 * 60
        os.utime(stale, (old, old))

        result = reconcile_publications(self.root, register_fn=self._register)

        diagnostic = os.path.join(self.root, ".diagnostics", "stale-publication.json")
        self.assertEqual(result["staging_cleaned"], 1)
        self.assertFalse(os.path.exists(stale))
        self.assertTrue(os.path.isfile(diagnostic))
        self.assertLess(os.path.getsize(diagnostic), 4096)

    def test_delete_moves_through_trash_after_reference_check(self):
        committed = commit_staging(self.staging, self.root, self._manifest(), register_fn=self._register)
        final = os.path.join(self.root, committed["publication_id"])
        reference_check = mock.Mock(return_value=False)
        delete_model = mock.Mock(return_value=True)

        result = delete_published_asset(committed["publication_id"], self.root, reference_check, delete_model)

        self.assertEqual(result, {"publication_id": committed["publication_id"], "deleted": True})
        self.assertFalse(os.path.exists(final))
        self.assertFalse(os.path.exists(os.path.join(self.root, ".trash", committed["publication_id"])))
        reference_check.assert_called_once_with(committed["publication_id"], os.path.realpath(final))
        delete_model.assert_called_once_with(os.path.realpath(final))


class ServingCompatibilityTest(unittest.TestCase):
    def _vllm_modules(self, inspect):
        vllm = types.ModuleType("vllm")
        vllm.__version__ = "0.7.0"
        executor = types.ModuleType("vllm.model_executor")
        models = types.ModuleType("vllm.model_executor.models")
        registry = types.ModuleType("vllm.model_executor.models.registry")
        registry.ModelRegistry = type("ModelRegistry", (), {"inspect_model_cls": staticmethod(inspect)})
        return {
            "vllm": vllm,
            "vllm.model_executor": executor,
            "vllm.model_executor.models": models,
            "vllm.model_executor.models.registry": registry,
        }

    def test_compatibility_uses_vllm_registry_and_marks_version_changes_stale(self):
        def inspect(architectures):
            if architectures == ["Qwen2_5_VLForConditionalGeneration"]:
                raise ValueError("unsupported")
            self.assertEqual(architectures, ["Qwen2ForCausalLM"])
            return object()

        with mock.patch.dict(sys.modules, self._vllm_modules(inspect)):
            ready = inspect_serving_compatibility(["Qwen2ForCausalLM"])
            blocked = inspect_serving_compatibility(["Qwen2_5_VLForConditionalGeneration"])
            stale = inspect_serving_compatibility(["Qwen2ForCausalLM"], recorded_version="0.6.0")

        self.assertEqual(ready["tested_version"], "0.7.0")
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(blocked["reason_code"], "unsupported_architecture")
        self.assertEqual(stale["status"], "stale")


class PublishedSyncGuardTest(unittest.TestCase):
    def test_published_registration_uses_core_orm_fields_only(self):
        from app.repositories import model_register_published

        manifest = {
            "display_name": "Published model",
            "artifact_type": "vlm",
            "provenance": {"task_id": "task-001"},
            "model": {"model_type": "qwen2_5_vl"},
            "files": {"total_bytes": 123},
        }
        with mock.patch("app.repositories.model_register") as register:
            model_register_published("/published/model", manifest)

        register.assert_called_once_with(
            path="/published/model",
            name="Published model",
            source="published",
            task_id="task-001",
            architecture="qwen2_5_vl",
            is_vlm=True,
            size_bytes=123,
        )

    def test_sync_queries_and_preserves_published_rows(self):
        from flask import Flask

        from app.extensions import db
        from app.models import Model
        from app.services import Services

        app = Flask(__name__)
        app.config.update(
            SQLALCHEMY_DATABASE_URI="sqlite://",
            SQLALCHEMY_BINDS={"model_gateway": "sqlite://"},
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
        )
        db.init_app(app)
        with app.app_context():
            db.create_all()
            published = Model(path="/missing/published", name="Published", source="published")
            db.session.add(published)
            db.session.commit()
            model_id = published.id

            services = Services.__new__(Services)
            services.app = app
            services.state = SimpleNamespace(merge_dir="")
            services.logger = None
            services._collect_base_models_on_disk = lambda: []
            services._collect_merged_models_on_disk = lambda: []

            with mock.patch("app.repositories.model_list_by_sources", wraps=__import__("app.repositories", fromlist=["model_list_by_sources"]).model_list_by_sources) as listed:
                services.sync_models_db_from_disk()

            self.assertIn("published", listed.call_args.args[0])
            restored = db.session.get(Model, model_id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.id, model_id)
            db.session.remove()


if __name__ == "__main__":
    unittest.main()
