import json
import os
import shutil
import stat
import sys
import tempfile
import threading
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
            "recipe_sha256": "a" * 64,
            "recipe_snapshot": {"models": ["parent-a"]},
            "parents": ["parent-a"],
            "parent_fingerprints": [{
                "source_path": "parent-a",
                "weights_sha256": "b" * 64,
                "weight_bytes": 7,
                "weight_files": [{
                    "path": "model.safetensors",
                    "size_bytes": 7,
                    "sha256": "c" * 64,
                }],
                "index_bytes": 0,
                "index_files": [],
            }],
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

    def _new_staging(self, publication_id):
        staging = os.path.join(self.root, ".staging", publication_id)
        self._write_model(staging)
        request = dict(self.request, publication_id=publication_id, task_id="task-%s" % publication_id)
        return staging, build_manifest(
            staging,
            request,
            inspect_model(staging),
            validation={"structural": {"status": "passed"}},
            compatibility={"serving": {"backend": "vllm", "status": "ready", "tested_version": "0.7.0"}},
        )

    def test_commit_is_atomic_and_manifest_excludes_itself_from_hashes(self):
        manifest = self._manifest()
        self.assertEqual(manifest["schema_version"], 2)
        committed = commit_staging(self.staging, self.root, manifest, register_fn=self._register)

        final = os.path.join(self.root, committed["publication_id"])
        self.assertFalse(os.path.exists(self.staging))
        self.assertTrue(os.path.isfile(os.path.join(final, "publication_manifest.json")))
        self.assertEqual(
            stat.S_IMODE(os.stat(os.path.join(final, "publication_manifest.json")).st_mode),
            0o644,
        )
        self.assertEqual(committed["publication_state"], "published")
        paths = [item["path"] for item in committed["files"]["entries"]]
        self.assertNotIn("publication_manifest.json", paths)
        self.assertEqual(paths, sorted(paths))
        self.assertEqual(len(self.registered), 1)

    def test_schema1_recipe_manifest_remains_readable_without_weight_fingerprints(self):
        manifest = self._manifest()
        manifest["schema_version"] = 1
        manifest["provenance"].pop("parent_fingerprints", None)

        committed = commit_staging(self.staging, self.root, manifest, register_fn=self._register)

        self.assertEqual(committed["schema_version"], 1)

    def test_schema2_existing_model_manifest_requires_source_fingerprint(self):
        request = {
            **self.request,
            "recipe_path": None,
            "recipe_sha256": None,
            "recipe_snapshot": {},
            "parents": [],
            "parent_fingerprints": [],
            "source_model": {
                "model_id": "model-1",
                "source_path": "/models/source",
                "weights_sha256": "d" * 64,
                "weight_bytes": 7,
                "weight_files": [{
                    "path": "model.safetensors",
                    "size_bytes": 7,
                    "sha256": "e" * 64,
                }],
                "index_bytes": 0,
                "index_files": [],
            },
        }
        manifest = build_manifest(
            self.staging,
            request,
            self.inspection,
            validation={"structural": {"status": "passed"}},
            compatibility={"serving": {"status": "ready", "tested_version": "0.7.0"}},
        )
        self.assertEqual(manifest["provenance"]["source_model"]["model_id"], "model-1")
        manifest["provenance"].pop("source_model")

        with self.assertRaisesRegex(PublicationError, "manifest contract is invalid"):
            commit_staging(self.staging, self.root, manifest, register_fn=self._register)

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

    def test_explicit_test_fault_exits_after_rename_before_registration(self):
        with mock.patch.dict(os.environ, {
            "MERGEKIT_ENABLE_TEST_FAULTS": "1",
            "MERGEKIT_PUBLICATION_TEST_CRASH_AFTER_RENAME": "1",
        }, clear=False):
            with mock.patch("app.model_publication.os._exit", side_effect=RuntimeError("simulated crash")) as exit_process:
                with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                    commit_staging(self.staging, self.root, self._manifest(), register_fn=self._register)

        final = os.path.join(self.root, self.publication_id)
        exit_process.assert_called_once_with(86)
        self.assertFalse(os.path.exists(self.staging))
        self.assertTrue(os.path.isdir(final))
        self.assertEqual(self.registered, [])
        self.assertEqual(validate_published_asset(final, full_hash=True)["publication_state"], "registration_pending")

    def test_recipe_manifest_rejects_missing_snapshot_and_hash(self):
        manifest = self._manifest()
        manifest["provenance"]["recipe_snapshot"] = {}
        manifest["provenance"]["recipe_sha256"] = None

        with self.assertRaisesRegex(PublicationError, "manifest contract is invalid"):
            commit_staging(self.staging, self.root, manifest, register_fn=self._register)

    def test_recipe_manifest_rejects_missing_parent_weight_fingerprints(self):
        manifest = self._manifest()
        manifest["provenance"].pop("parent_fingerprints")

        with self.assertRaisesRegex(PublicationError, "manifest contract is invalid"):
            commit_staging(self.staging, self.root, manifest, register_fn=self._register)

    def test_vlm_recipe_manifest_requires_complete_vlm_base_provenance(self):
        manifest = self._manifest()
        manifest["artifact_type"] = "vlm"
        manifest["capabilities"] = ["text_generation", "vision_language"]
        manifest["model"]["model_type"] = "qwen2_5_vl"
        manifest["model"]["architectures"] = ["Qwen2_5_VLForConditionalGeneration"]
        manifest["provenance"]["vlm_base"] = {}

        with self.assertRaisesRegex(PublicationError, "manifest contract is invalid"):
            commit_staging(self.staging, self.root, manifest, register_fn=self._register)

    def test_vlm_recipe_manifest_rejects_missing_base_weight_fingerprint(self):
        manifest = self._manifest()
        manifest["artifact_type"] = "vlm"
        manifest["capabilities"] = ["text_generation", "vision_language"]
        manifest["model"]["model_type"] = "qwen2_5_vl"
        manifest["model"]["architectures"] = ["Qwen2_5_VLForConditionalGeneration"]
        manifest["provenance"]["vlm_base"] = {
            "source_path": "/models/vlm",
            "config_sha256": "d" * 64,
            "visual_weight_count": 3,
        }

        with self.assertRaisesRegex(PublicationError, "manifest contract is invalid"):
            commit_staging(self.staging, self.root, manifest, register_fn=self._register)

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
        self.assertEqual(first["published"], 1)
        self.assertEqual(second["registered"], 1)
        self.assertEqual(second["published"], 0)
        self.assertEqual(len(self.registered), 2)

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

    def test_manifest_rejects_wrong_individual_file_size(self):
        committed = commit_staging(self.staging, self.root, self._manifest(), self._register)
        final = os.path.join(self.root, committed["publication_id"])
        manifest_path = os.path.join(final, "publication_manifest.json")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["files"]["entries"][0]["size_bytes"] += 1
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

        with self.assertRaisesRegex(PublicationError, "validation_failed"):
            validate_published_asset(final, full_hash=False)

    def test_manifest_rejects_non_hex_sha256_without_hashing_files(self):
        committed = commit_staging(self.staging, self.root, self._manifest(), self._register)
        final = os.path.join(self.root, committed["publication_id"])
        manifest_path = os.path.join(final, "publication_manifest.json")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["files"]["entries"][0]["sha256"] = "z" * 64
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

        with self.assertRaisesRegex(PublicationError, "validation_failed"):
            validate_published_asset(final, full_hash=False)

    def test_nested_manifest_temp_file_is_part_of_inventory(self):
        committed = commit_staging(self.staging, self.root, self._manifest(), self._register)
        final = os.path.join(self.root, committed["publication_id"])
        nested = os.path.join(final, "nested")
        os.makedirs(nested)
        with open(os.path.join(nested, ".manifest-interrupted.json"), "w", encoding="utf-8") as handle:
            handle.write("nested")

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

        result = reconcile_publications(self.root, register_fn=self._register, active_check_fn=lambda task_id: False)

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

    def test_commit_rejects_arbitrary_or_changed_staging(self):
        arbitrary = os.path.join(self.tmpdir.name, self.publication_id)
        self._write_model(arbitrary)
        arbitrary_manifest = build_manifest(
            arbitrary, self.request, inspect_model(arbitrary), {"structural": {}}, {"serving": {"status": "ready"}}
        )
        with self.assertRaisesRegex(PublicationError, "staging_invalid"):
            commit_staging(arbitrary, self.root, arbitrary_manifest, self._register)

        manifest = self._manifest()
        with open(os.path.join(self.staging, "model.safetensors"), "ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(PublicationError, "validation_failed"):
            commit_staging(self.staging, self.root, manifest, self._register)

    def test_commit_rejects_cross_device_staging(self):
        real_stat = os.stat

        def stat_with_other_root_device(path, *args, **kwargs):
            value = real_stat(path, *args, **kwargs)
            if os.path.abspath(path) == os.path.abspath(self.root):
                return SimpleNamespace(st_dev=value.st_dev + 1)
            return value

        with mock.patch("app.model_publication.os.stat", side_effect=stat_with_other_root_device):
            with self.assertRaisesRegex(PublicationError, "cross_device_staging"):
                commit_staging(self.staging, self.root, self._manifest(), self._register)

    def test_commit_fsyncs_staging_parent_after_rename(self):
        events = []
        final = os.path.join(self.root, self.publication_id)
        real_fsync = __import__("app.model_publication", fromlist=["_fsync_directory"])._fsync_directory
        real_replace = os.replace

        def record_fsync(path):
            events.append(("fsync", os.path.realpath(path)))
            return real_fsync(path)

        def record_replace(source, destination):
            events.append(("replace", os.path.realpath(source), os.path.realpath(destination)))
            return real_replace(source, destination)

        with mock.patch("app.model_publication._fsync_directory", side_effect=record_fsync), mock.patch(
            "app.model_publication.os.replace", side_effect=record_replace
        ):
            commit_staging(self.staging, self.root, self._manifest(), self._register)

        rename_index = next(
            index for index, event in enumerate(events)
            if event[0] == "replace" and event[1] == os.path.realpath(self.staging) and event[2] == os.path.realpath(final)
        )
        fsyncs_after_rename = [event[1] for event in events[rename_index + 1:] if event[0] == "fsync"]
        self.assertIn(os.path.realpath(os.path.join(self.root, ".staging")), fsyncs_after_rename)
        self.assertIn(os.path.realpath(self.root), fsyncs_after_rename)

    def test_validate_rejects_wrong_directory_and_manifest_contract(self):
        committed = commit_staging(self.staging, self.root, self._manifest(), self._register)
        final = os.path.join(self.root, committed["publication_id"])
        renamed = os.path.join(self.root, "wrong-directory")
        os.rename(final, renamed)
        with self.assertRaisesRegex(PublicationError, "validation_failed"):
            validate_published_asset(renamed)

        os.rename(renamed, final)
        manifest_path = os.path.join(final, "publication_manifest.json")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["files"]["hash_algorithm"] = "md5"
        manifest["capabilities"] = ["vision_language"]
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        with self.assertRaisesRegex(PublicationError, "validation_failed"):
            validate_published_asset(final)

    def test_structural_validation_does_not_hash_files(self):
        committed = commit_staging(self.staging, self.root, self._manifest(), self._register)
        final = os.path.join(self.root, committed["publication_id"])
        with mock.patch("app.model_publication._hash_file", side_effect=AssertionError("unexpected hash")):
            validate_published_asset(final, full_hash=False)

    def test_manifest_temp_files_do_not_invalidate_asset(self):
        committed = commit_staging(self.staging, self.root, self._manifest(), self._register)
        final = os.path.join(self.root, committed["publication_id"])
        with open(os.path.join(final, ".manifest-interrupted.json"), "w", encoding="utf-8") as handle:
            handle.write("partial")

        validate_published_asset(final, full_hash=True)
        with mock.patch("app.model_publication.inspect_serving_compatibility", return_value={"backend": "vllm", "status": "ready", "tested_version": "0.7.0"}):
            result = reconcile_publications(self.root, self._register)
        self.assertEqual(result["quarantined"], 0)

    def test_symlinked_root_and_control_directories_are_rejected(self):
        real_root = os.path.join(self.tmpdir.name, "real-root")
        os.makedirs(real_root)
        shutil.rmtree(self.root)
        os.symlink(real_root, self.root)
        with self.assertRaisesRegex(PublicationError, "unsafe_path"):
            reconcile_publications(self.root, self._register)

        os.unlink(self.root)
        os.makedirs(self.root)
        os.symlink(self.tmpdir.name, os.path.join(self.root, ".staging"))
        self._write_model(self.staging)
        with self.assertRaisesRegex(PublicationError, "unsafe_path"):
            commit_staging(self.staging, self.root, self._manifest(), self._register)

    def test_reconcile_rejects_unused_symlinked_control_directory(self):
        os.makedirs(self.root, exist_ok=True)
        os.symlink(self.tmpdir.name, os.path.join(self.root, ".diagnostics"))

        with self.assertRaisesRegex(PublicationError, "unsafe_path"):
            reconcile_publications(self.root, self._register)

    def test_active_or_unknown_old_staging_is_preserved(self):
        stale = os.path.join(self.root, ".staging", "stale-publication")
        self._write_model(stale)
        old = time.time() - 2 * 24 * 60 * 60
        os.utime(stale, (old, old))

        active = reconcile_publications(self.root, self._register, active_check_fn=lambda task_id: True)
        unknown = reconcile_publications(self.root, self._register, active_check_fn=lambda task_id: None)

        self.assertEqual(active["staging_cleaned"], 0)
        self.assertEqual(unknown["staging_cleaned"], 0)
        self.assertTrue(os.path.isdir(stale))

    def test_empty_publication_root_has_no_staging_error(self):
        empty_root = os.path.join(self.tmpdir.name, "empty-published")
        result = reconcile_publications(empty_root, self._register, active_check_fn=lambda task_id: False)

        self.assertEqual(result["staging_cleaned"], 0)
        self.assertEqual(result["errors"], [])

    def test_recovery_isolates_bad_assets_and_restores_published_rows(self):
        first_staging, first_manifest = self._new_staging("publication-a")
        second_staging, second_manifest = self._new_staging("publication-b")
        first = commit_staging(first_staging, self.root, first_manifest, lambda path, manifest: None)
        second = commit_staging(second_staging, self.root, second_manifest, lambda path, manifest: None)
        with open(os.path.join(self.root, first["publication_id"], "model.safetensors"), "ab") as handle:
            handle.write(b"bad")
        calls = []

        with mock.patch("app.model_publication.inspect_serving_compatibility", return_value={"backend": "vllm", "status": "ready", "tested_version": "0.7.0"}):
            result = reconcile_publications(self.root, lambda path, manifest: calls.append(manifest["publication_id"]))

        self.assertEqual(result["quarantined"], 1)
        self.assertEqual(calls, [second["publication_id"]])

    def test_recovery_filesystem_error_does_not_block_later_assets(self):
        first_staging, first_manifest = self._new_staging("publication-a")
        second_staging, second_manifest = self._new_staging("publication-b")
        first = commit_staging(first_staging, self.root, first_manifest, lambda path, manifest: None)
        second = commit_staging(second_staging, self.root, second_manifest, lambda path, manifest: None)
        with open(os.path.join(self.root, first["publication_id"], "model.safetensors"), "ab") as handle:
            handle.write(b"bad")
        calls = []

        with mock.patch("app.model_publication._quarantine_locked", side_effect=OSError("disk busy")), mock.patch(
            "app.model_publication.inspect_serving_compatibility",
            return_value={"backend": "vllm", "status": "ready", "tested_version": "0.7.0"},
        ):
            result = reconcile_publications(self.root, lambda path, manifest: calls.append(manifest["publication_id"]))

        self.assertEqual(calls, [second["publication_id"]])
        self.assertTrue(any(error["asset"] == first["publication_id"] for error in result["errors"]))

    def test_recovery_keeps_transient_failures_and_continues(self):
        assets = []
        for publication_id in ("publication-a", "publication-b", "publication-c"):
            staging, manifest = self._new_staging(publication_id)
            assets.append(commit_staging(staging, self.root, manifest, lambda path, manifest: None))
        calls = []

        def register(path, manifest):
            calls.append(manifest["publication_id"])
            if manifest["publication_id"] == "publication-b":
                raise RuntimeError("database transient")

        with mock.patch(
            "app.model_publication.inspect_serving_compatibility",
            side_effect=[RuntimeError("vllm transient"), {"backend": "vllm", "status": "ready", "tested_version": "0.7.0"}, {"backend": "vllm", "status": "ready", "tested_version": "0.7.0"}],
        ):
            result = reconcile_publications(self.root, register)

        self.assertEqual(calls, ["publication-b", "publication-c"])
        self.assertEqual({error["asset"] for error in result["errors"]}, {"publication-a", "publication-b"})
        self.assertTrue(all(os.path.isdir(os.path.join(self.root, asset["publication_id"])) for asset in assets))

    def test_delete_waits_for_reconcile_without_recreating_asset(self):
        committed = commit_staging(self.staging, self.root, self._manifest(), self._register)
        entered = threading.Event()
        release = threading.Event()
        deleted = []

        def register(path, manifest):
            entered.set()
            release.wait(1)

        def reconcile():
            reconcile_publications(self.root, register)

        def delete():
            deleted.append(delete_published_asset(committed["publication_id"], self.root, lambda publication_id, path: False, lambda path: True))

        with mock.patch("app.model_publication.inspect_serving_compatibility", return_value={"backend": "vllm", "status": "ready", "tested_version": "0.7.0"}):
            reconcile_thread = threading.Thread(target=reconcile)
            reconcile_thread.start()
            self.assertTrue(entered.wait(1))
            delete_thread = threading.Thread(target=delete)
            delete_thread.start()
            self.assertTrue(delete_thread.is_alive())
            release.set()
            reconcile_thread.join(2)
            delete_thread.join(2)

        self.assertFalse(reconcile_thread.is_alive())
        self.assertFalse(delete_thread.is_alive())
        self.assertEqual(deleted, [{"publication_id": committed["publication_id"], "deleted": True}])
        self.assertFalse(os.path.exists(os.path.join(self.root, committed["publication_id"])))
        self.assertFalse(os.path.exists(os.path.join(self.root, "publication_manifest.json")))

    def test_commit_and_reconcile_do_not_double_transition(self):
        entered = threading.Event()
        release = threading.Event()
        committed = []
        reconciled = []

        def register(path, manifest):
            entered.set()
            release.wait(1)

        def commit():
            committed.append(commit_staging(self.staging, self.root, self._manifest(), register))

        def reconcile():
            reconciled.append(reconcile_publications(self.root, lambda path, manifest: None))

        with mock.patch("app.model_publication.inspect_serving_compatibility", return_value={"backend": "vllm", "status": "ready", "tested_version": "0.7.0"}):
            commit_thread = threading.Thread(target=commit)
            commit_thread.start()
            self.assertTrue(entered.wait(1))
            reconcile_thread = threading.Thread(target=reconcile)
            reconcile_thread.start()
            release.set()
            commit_thread.join(2)
            reconcile_thread.join(2)

        self.assertFalse(commit_thread.is_alive())
        self.assertFalse(reconcile_thread.is_alive())
        self.assertEqual(committed[0]["publication_state"], "published")
        self.assertEqual(reconciled[0]["published"], 0)
        self.assertEqual(validate_published_asset(os.path.join(self.root, self.publication_id))["publication_state"], "published")

    def test_delete_rejects_symlinked_asset_and_trash(self):
        committed = commit_staging(self.staging, self.root, self._manifest(), self._register)
        final = os.path.join(self.root, committed["publication_id"])
        shutil.rmtree(final)
        os.symlink(self.tmpdir.name, final)
        with self.assertRaisesRegex(PublicationError, "unsafe_path"):
            delete_published_asset(committed["publication_id"], self.root, lambda publication_id, path: False, lambda path: True)

        os.unlink(final)
        staging, manifest = self._new_staging(committed["publication_id"])
        commit_staging(staging, self.root, manifest, self._register)
        os.symlink(self.tmpdir.name, os.path.join(self.root, ".trash"))
        with self.assertRaisesRegex(PublicationError, "unsafe_path"):
            delete_published_asset(committed["publication_id"], self.root, lambda publication_id, path: False, lambda path: True)

    def test_two_reconcilers_transition_pending_once(self):
        with self.assertRaisesRegex(RuntimeError, "pending"):
            commit_staging(self.staging, self.root, self._manifest(), lambda path, manifest: (_ for _ in ()).throw(RuntimeError("pending")))
        calls = []
        results = []
        start = threading.Event()

        def register(path, manifest):
            start.wait(1)
            calls.append(manifest["publication_id"])

        def reconcile():
            results.append(reconcile_publications(self.root, register))

        with mock.patch("app.model_publication.inspect_serving_compatibility", return_value={"backend": "vllm", "status": "ready", "tested_version": "0.7.0"}):
            threads = [threading.Thread(target=reconcile) for _ in range(2)]
            for thread in threads:
                thread.start()
            start.set()
            for thread in threads:
                thread.join(2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(sum(result["published"] for result in results), 1)
        self.assertEqual(validate_published_asset(os.path.join(self.root, self.publication_id))["publication_state"], "published")


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
    def test_publication_task_active_check_is_fail_closed(self):
        from app.repositories import publication_task_is_active

        def check(rows):
            query = mock.Mock()
            query.filter.return_value = query
            query.all.return_value = rows
            with mock.patch("app.repositories.db.session.query", return_value=query):
                return publication_task_is_active("publication-001")

        active = SimpleNamespace(id="task-001", task_type="model_publication", status="validating", config={"publication_id": "publication-001"})
        complete = SimpleNamespace(id="task-001", task_type="model_publication", status="completed", config={"publication_id": "publication-001"})
        other = SimpleNamespace(id="task-001", task_type="merge", status="running", config={"publication_id": "publication-001"})
        self.assertTrue(check([active]))
        self.assertFalse(check([complete]))
        self.assertFalse(check([other]))
        with mock.patch("app.repositories.db.session.query", side_effect=RuntimeError("db unavailable")):
            self.assertIsNone(publication_task_is_active("publication-001"))

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

    def test_recovered_registration_completes_registration_pending_task(self):
        from app.repositories import model_register_recovered_publication

        task = SimpleNamespace(id="task-001", task_type="model_publication", status="registration_pending")
        manifest = {
            "display_name": "Recovered model",
            "artifact_type": "text",
            "provenance": {"task_id": task.id},
            "model": {"model_type": "qwen2"},
            "files": {"total_bytes": 123},
        }
        registered = object()
        with mock.patch("app.repositories.model_register_published", return_value=registered):
            with mock.patch("app.repositories.db.session.get", return_value=task):
                with mock.patch("app.repositories.task_set_status") as set_status:
                    result = model_register_recovered_publication("/published/model", manifest)

        self.assertIs(result, registered)
        set_status.assert_called_once_with(
            task.id,
            "completed",
            error="",
            model_path="/published/model",
            config_patch={
                "commit_in_progress": False,
                "validation_enqueued": False,
                "error_code": None,
            },
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
