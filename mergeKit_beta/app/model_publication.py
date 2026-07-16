"""Atomic filesystem publication and startup recovery for formal model assets."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from typing import Callable, Sequence

from app.model_inspection import ModelInspection


_MANIFEST_NAME = "publication_manifest.json"
_STALE_STAGING_SECONDS = 24 * 60 * 60


class PublicationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@contextmanager
def publication_lock(root: str):
    os.makedirs(root, exist_ok=True)
    lock_path = os.path.join(root, ".publication.lock")
    with open(lock_path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_write_json(path: str, payload: dict) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _error(code: str, message: str) -> PublicationError:
    return PublicationError(code, "%s: %s" % (code, message))


def _publication_id(value: object) -> str:
    publication_id = str(value or "").strip()
    if not publication_id or publication_id in (".", "..") or os.path.basename(publication_id) != publication_id:
        raise _error("invalid_publication_id", "publication_id must be one directory name")
    return publication_id


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(path: str) -> tuple[list[dict], int]:
    if os.path.islink(path):
        raise _error("validation_failed", "published asset root must not be a symlink")
    entries = []
    total_bytes = 0
    for current, directories, files in os.walk(path, followlinks=False):
        directories.sort()
        files.sort()
        for name in directories:
            if os.path.islink(os.path.join(current, name)):
                raise _error("validation_failed", "published assets must not contain symlinks")
        for name in files:
            absolute = os.path.join(current, name)
            relative = os.path.relpath(absolute, path).replace(os.sep, "/")
            if relative == _MANIFEST_NAME:
                continue
            if os.path.islink(absolute) or not os.path.isfile(absolute):
                raise _error("validation_failed", "published assets must contain regular files only")
            size_bytes = os.path.getsize(absolute)
            entries.append({"path": relative, "sha256": _hash_file(absolute), "size_bytes": size_bytes})
            total_bytes += size_bytes
    return entries, total_bytes


def inspect_serving_compatibility(architectures: Sequence[str], recorded_version: str | None = None) -> dict:
    from vllm import __version__ as vllm_version
    from vllm.model_executor.models.registry import ModelRegistry

    result = {"backend": "vllm", "tested_version": vllm_version}
    if recorded_version is not None and recorded_version != vllm_version:
        result.update({"status": "stale", "reason_code": "version_changed"})
        return result
    try:
        ModelRegistry.inspect_model_cls(list(architectures))
    except ValueError as exc:
        result.update({
            "status": "blocked",
            "reason_code": "unsupported_architecture",
            "reason": str(exc),
        })
        return result
    result["status"] = "ready"
    return result


def build_manifest(
    staging: str,
    request: dict,
    inspection: ModelInspection,
    validation: dict,
    compatibility: dict,
) -> dict:
    publication_id = _publication_id(request.get("publication_id") or os.path.basename(os.path.abspath(staging)))
    entries, total_bytes = _inventory(staging)
    artifact_type = "vlm" if inspection.is_vlm else "text"
    manifest_validation = {"structural": {}, "functional": {}, "evaluation": {}}
    manifest_validation.update(validation or {})
    manifest_compatibility = {"transformers": {}, "lm_eval": {}, "lmms_eval": {}, "serving": {}}
    manifest_compatibility.update(compatibility or {})
    created_at = request.get("created_at") or _now()
    return {
        "schema_version": 1,
        "publication_id": publication_id,
        "display_name": str(request.get("display_name") or publication_id),
        "artifact_type": artifact_type,
        "capabilities": ["text_generation"] + (["vision_language"] if artifact_type == "vlm" else []),
        "publication_state": "registration_pending",
        "provenance": {
            "task_id": request.get("task_id"),
            "recipe_path": request.get("recipe_path"),
            "recipe_sha256": request.get("recipe_sha256"),
            "recipe_snapshot": request.get("recipe_snapshot") or {},
            "parents": request.get("parents") or [],
            "vlm_base": request.get("vlm_base") or {},
        },
        "model": {
            "model_type": inspection.model_type,
            "architectures": list(inspection.architectures),
            "tokenizer_class": request.get("tokenizer_class"),
            "processor_class": inspection.processor_class,
            "dtype": request.get("dtype"),
        },
        "files": {"hash_algorithm": "sha256", "total_bytes": total_bytes, "entries": entries},
        "validation": manifest_validation,
        "compatibility": manifest_compatibility,
        "timestamps": {"created_at": created_at, "validated_at": request.get("validated_at"), "published_at": None},
    }


def _load_manifest(path: str) -> dict:
    manifest_path = os.path.join(path, _MANIFEST_NAME)
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise _error("validation_failed", "manifest is missing or invalid") from exc
    if not isinstance(manifest, dict):
        raise _error("validation_failed", "manifest must be a JSON object")
    return manifest


def validate_published_asset(path: str, full_hash: bool = False) -> dict:
    asset_path = os.path.abspath(path)
    if not os.path.isdir(asset_path) or os.path.islink(asset_path):
        raise _error("validation_failed", "published asset directory is missing or unsafe")
    manifest = _load_manifest(asset_path)
    try:
        _publication_id(manifest.get("publication_id"))
        files = manifest["files"]
        expected_entries = files["entries"]
        expected_total = int(files["total_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _error("validation_failed", "manifest file inventory is invalid") from exc
    if manifest.get("schema_version") != 1 or manifest.get("publication_state") not in ("registration_pending", "published"):
        raise _error("validation_failed", "manifest schema or publication state is invalid")
    if not isinstance(expected_entries, list):
        raise _error("validation_failed", "manifest entries must be a list")
    if any(
        not isinstance(entry, dict)
        or not isinstance(entry.get("path"), str)
        or not isinstance(entry.get("sha256"), str)
        or not isinstance(entry.get("size_bytes"), int)
        for entry in expected_entries
    ):
        raise _error("validation_failed", "manifest entries must contain path, hash and size")
    actual_entries, actual_total = _inventory(asset_path)
    expected_paths = [entry["path"] for entry in expected_entries]
    actual_paths = [entry["path"] for entry in actual_entries]
    if expected_paths != sorted(expected_paths) or expected_paths != actual_paths or expected_total != actual_total:
        raise _error("validation_failed", "manifest file inventory does not match asset")
    if full_hash:
        expected_hashes = [(entry.get("path"), entry.get("sha256")) for entry in expected_entries]
        actual_hashes = [(entry["path"], entry["sha256"]) for entry in actual_entries]
        if expected_hashes != actual_hashes:
            raise _error("validation_failed", "manifest file hashes do not match asset")
    return manifest


def _fsync_directory(path: str) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def commit_staging(staging: str, root: str, manifest: dict, register_fn: Callable[[str, dict], object]) -> dict:
    root = os.path.abspath(root)
    staging = os.path.abspath(staging)
    publication_id = _publication_id(manifest.get("publication_id"))
    final = os.path.join(root, publication_id)
    if not os.path.isdir(staging) or os.path.islink(staging):
        raise _error("staging_missing", "staging directory is missing or unsafe")
    os.makedirs(root, exist_ok=True)
    if os.stat(staging).st_dev != os.stat(root).st_dev:
        raise _error("cross_device_staging", "staging and publication root must share a filesystem")
    pending_manifest = json.loads(json.dumps(manifest))
    pending_manifest["publication_state"] = "registration_pending"
    pending_manifest.setdefault("timestamps", {})["published_at"] = None
    with publication_lock(root):
        if os.path.exists(final):
            raise _error("publication_exists", "publication directory already exists")
        atomic_write_json(os.path.join(staging, _MANIFEST_NAME), pending_manifest)
        _fsync_directory(staging)
        os.replace(staging, final)
        _fsync_directory(root)
        try:
            register_fn(os.path.realpath(final), pending_manifest)
        except Exception:
            raise
        pending_manifest["publication_state"] = "published"
        pending_manifest["timestamps"]["published_at"] = _now()
        atomic_write_json(os.path.join(final, _MANIFEST_NAME), pending_manifest)
    return pending_manifest


def _quarantine_asset(root: str, path: str) -> None:
    quarantine = os.path.join(root, ".quarantine")
    os.makedirs(quarantine, exist_ok=True)
    name = "%s-%d" % (os.path.basename(path), int(time.time()))
    destination = os.path.join(quarantine, name)
    with publication_lock(root):
        if os.path.lexists(path):
            os.replace(path, destination)
            _fsync_directory(root)
            _fsync_directory(quarantine)


def _cleanup_staging(root: str) -> int:
    staging_root = os.path.join(root, ".staging")
    if not os.path.isdir(staging_root):
        return 0
    cleaned = 0
    cutoff = time.time() - _STALE_STAGING_SECONDS
    for name in sorted(os.listdir(staging_root)):
        if name.startswith("."):
            continue
        path = os.path.join(staging_root, name)
        if not os.path.isdir(path) or os.path.islink(path) or os.path.getmtime(path) > cutoff:
            continue
        diagnostic = os.path.join(root, ".diagnostics", "%s.json" % name)
        with publication_lock(root):
            if not os.path.isdir(path) or os.path.getmtime(path) > cutoff:
                continue
            atomic_write_json(diagnostic, {"publication_id": name, "reason": "stale_inactive_staging", "recorded_at": _now()})
            shutil.rmtree(path)
            _fsync_directory(staging_root)
        cleaned += 1
    return cleaned


def reconcile_publications(root: str, register_fn: Callable[[str, dict], object]) -> dict:
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    result = {"registered": 0, "published": 0, "quarantined": 0, "staging_cleaned": _cleanup_staging(root), "skipped": 0}
    seen_ids = set()
    seen_paths = set()
    for name in sorted(os.listdir(root)):
        if name.startswith("."):
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            manifest = validate_published_asset(path, full_hash=True)
        except PublicationError:
            _quarantine_asset(root, path)
            result["quarantined"] += 1
            continue
        publication_id = manifest["publication_id"]
        real_path = os.path.realpath(path)
        if publication_id in seen_ids or real_path in seen_paths:
            result["skipped"] += 1
            continue
        seen_ids.add(publication_id)
        seen_paths.add(real_path)
        serving = (manifest.get("compatibility") or {}).get("serving") or {}
        compatibility = inspect_serving_compatibility(
            manifest.get("model", {}).get("architectures") or (),
            recorded_version=serving.get("tested_version"),
        )
        if compatibility != serving:
            manifest.setdefault("compatibility", {})["serving"] = compatibility
            atomic_write_json(os.path.join(path, _MANIFEST_NAME), manifest)
        if manifest["publication_state"] != "registration_pending":
            continue
        with publication_lock(root):
            register_fn(real_path, manifest)
            manifest["publication_state"] = "published"
            manifest.setdefault("timestamps", {})["published_at"] = _now()
            atomic_write_json(os.path.join(path, _MANIFEST_NAME), manifest)
        result["registered"] += 1
        result["published"] += 1
    return result


def delete_published_asset(
    publication_id: str,
    root: str,
    reference_check_fn: Callable[[str, str], bool],
    delete_model_fn: Callable[[str], bool],
) -> dict:
    root = os.path.abspath(root)
    publication_id = _publication_id(publication_id)
    path = os.path.join(root, publication_id)
    if not os.path.isdir(path):
        raise _error("publication_missing", "publication directory does not exist")
    real_path = os.path.realpath(path)
    trash = os.path.join(root, ".trash", publication_id)
    with publication_lock(root):
        if reference_check_fn(publication_id, real_path):
            raise _error("publication_referenced", "publication is still referenced")
        os.makedirs(os.path.dirname(trash), exist_ok=True)
        if os.path.exists(trash):
            raise _error("trash_conflict", "trash destination already exists")
        os.replace(path, trash)
        _fsync_directory(root)
        try:
            deleted_model = bool(delete_model_fn(real_path))
        except Exception:
            os.replace(trash, path)
            _fsync_directory(root)
            raise
        if not deleted_model:
            os.replace(trash, path)
            _fsync_directory(root)
            raise _error("model_delete_failed", "model registration could not be deleted")
    shutil.rmtree(trash)
    return {"publication_id": publication_id, "deleted": True}
