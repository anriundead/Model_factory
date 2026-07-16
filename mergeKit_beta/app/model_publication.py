"""Atomic filesystem publication and restart recovery for formal model assets."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
import shutil
import tempfile
import time
from typing import Callable, Sequence

from sqlalchemy.orm import Session

from app.model_inspection import ModelInspection


_MANIFEST_NAME = "publication_manifest.json"
_CONTROL_DIRS = (".staging", ".diagnostics", ".trash", ".quarantine")
_STALE_STAGING_SECONDS = 24 * 60 * 60


class PublicationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _error(code: str, message: str) -> PublicationError:
    return PublicationError(code, "%s: %s" % (code, message))


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _lstat(path: str):
    try:
        return os.lstat(path)
    except OSError as exc:
        raise _error("unsafe_path", "cannot inspect %s" % path) from exc


def _directory(path: str, code: str = "unsafe_path") -> str:
    stat_result = _lstat(path)
    if os.path.islink(path) or (stat_result.st_mode & 0o170000) != 0o040000:
        raise _error(code, "directory is missing, not a directory, or a symlink")
    return os.path.realpath(os.path.abspath(path))


def _publication_root(root: str, create: bool = False) -> str:
    root = os.path.abspath(root)
    if create and not os.path.lexists(root):
        os.makedirs(root, exist_ok=True)
    return _directory(root)


def _publication_id(value: object) -> str:
    publication_id = str(value or "").strip()
    separators = [os.sep]
    if os.altsep:
        separators.append(os.altsep)
    if (
        not publication_id
        or publication_id.startswith(".")
        or publication_id in (".", "..")
        or any(separator in publication_id for separator in separators)
        or os.path.basename(publication_id) != publication_id
    ):
        raise _error("invalid_publication_id", "publication_id must be a safe non-dot directory name")
    return publication_id


def _child(root: str, name: str) -> str:
    candidate = os.path.abspath(os.path.join(root, name))
    if os.path.commonpath((root, candidate)) != root:
        raise _error("unsafe_path", "path escapes publication root")
    if os.path.lexists(candidate) and os.path.commonpath((root, os.path.realpath(candidate))) != root:
        raise _error("unsafe_path", "resolved path escapes publication root")
    return candidate


def _control_dir(root: str, name: str, create: bool = False) -> str:
    if name not in _CONTROL_DIRS:
        raise _error("unsafe_path", "unknown publication control directory")
    path = _child(root, name)
    if create and not os.path.lexists(path):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
    return _directory(path)


def _asset_dir(root: str, publication_id: str, required: bool = True) -> str | None:
    path = _child(root, _publication_id(publication_id))
    if not os.path.lexists(path):
        if required:
            raise _error("publication_missing", "publication directory does not exist")
        return None
    return _directory(path, "unsafe_path")


def _staging_dir(root: str, staging: str, publication_id: str) -> str:
    expected = _child(_control_dir(root, ".staging"), _publication_id(publication_id))
    supplied = os.path.abspath(staging)
    if supplied != expected:
        raise _error("staging_invalid", "staging must be root/.staging/publication_id")
    return _directory(supplied, "staging_invalid")


@contextmanager
def publication_lock(root: str):
    root = _publication_root(root, create=True)
    for name in _CONTROL_DIRS:
        if os.path.lexists(_child(root, name)):
            _control_dir(root, name)
    lock_path = _child(root, ".publication.lock")
    if os.path.lexists(lock_path) and os.path.islink(lock_path):
        raise _error("unsafe_path", "publication lock must not be a symlink")
    with open(lock_path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield root
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_write_json(path: str, payload: dict) -> None:
    parent = _directory(os.path.dirname(path))
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fsync_directory(path: str) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: str) -> bool:
    stat_result = _lstat(path)
    return (stat_result.st_mode & 0o170000) == 0o100000


def _inventory(path: str, include_hash: bool = True) -> tuple[list[dict], int]:
    path = _directory(path, "validation_failed")
    entries = []
    total_bytes = 0
    for current, directories, files in os.walk(path, followlinks=False):
        directories.sort()
        files.sort()
        for name in directories:
            directory = os.path.join(current, name)
            if os.path.islink(directory) or not os.path.isdir(directory):
                raise _error("validation_failed", "published assets must not contain unsafe directories")
        for name in files:
            absolute = os.path.join(current, name)
            relative = os.path.relpath(absolute, path).replace(os.sep, "/")
            if relative == _MANIFEST_NAME or ("/" not in relative and relative.startswith(".manifest-")):
                continue
            if os.path.islink(absolute) or not _regular_file(absolute):
                raise _error("validation_failed", "published assets must contain regular files only")
            size_bytes = os.path.getsize(absolute)
            entry = {"path": relative, "size_bytes": size_bytes}
            if include_hash:
                entry["sha256"] = _hash_file(absolute)
            entries.append(entry)
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
        result.update({"status": "blocked", "reason_code": "unsupported_architecture", "reason": str(exc)})
        return result
    result["status"] = "ready"
    return result


def current_serving_compatibility(manifest: dict) -> dict:
    """Return fail-closed serving metadata for the installed backend version."""
    serving = dict(manifest["compatibility"]["serving"])
    try:
        current_version = package_version("vllm")
    except PackageNotFoundError:
        serving.update({"status": "blocked", "reason_code": "serving_runtime_unavailable"})
        return serving
    if serving.get("tested_version") != current_version:
        serving.update({"status": "stale", "reason_code": "version_changed"})
    return serving


def build_manifest(
    staging: str,
    request: dict,
    inspection: ModelInspection,
    validation: dict,
    compatibility: dict,
) -> dict:
    publication_id = _publication_id(request.get("publication_id") or os.path.basename(os.path.abspath(staging)))
    entries, total_bytes = _inventory(staging, include_hash=True)
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
        "timestamps": {"created_at": created_at, "validated_at": request.get("validated_at") or created_at, "published_at": None},
    }


def _load_manifest(path: str) -> dict:
    manifest_path = os.path.join(path, _MANIFEST_NAME)
    try:
        if os.path.islink(manifest_path):
            raise OSError("manifest is a symlink")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise _error("validation_failed", "manifest is missing or invalid") from exc
    if not isinstance(manifest, dict):
        raise _error("validation_failed", "manifest must be a JSON object")
    return manifest


def _safe_entry_path(value: object) -> bool:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        return False
    parts = value.split("/")
    return all(part and part not in (".", "..") for part in parts)


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def _validate_manifest(path: str, manifest: dict, full_hash: bool) -> dict:
    try:
        publication_id = _publication_id(manifest.get("publication_id"))
        if os.path.basename(path) != publication_id:
            raise ValueError("directory name")
        if manifest.get("schema_version") != 1 or manifest.get("publication_state") not in ("registration_pending", "published"):
            raise ValueError("schema")
        if not isinstance(manifest.get("display_name"), str) or not manifest["display_name"].strip():
            raise ValueError("display_name")
        artifact_type = manifest.get("artifact_type")
        if artifact_type not in ("text", "vlm"):
            raise ValueError("artifact_type")
        expected_capabilities = ["text_generation"] + (["vision_language"] if artifact_type == "vlm" else [])
        if manifest.get("capabilities") != expected_capabilities:
            raise ValueError("capabilities")
        provenance = manifest["provenance"]
        model = manifest["model"]
        validation = manifest["validation"]
        compatibility = manifest["compatibility"]
        timestamps = manifest["timestamps"]
        if not isinstance(provenance, dict) or not isinstance(provenance.get("task_id"), str) or not provenance["task_id"].strip():
            raise ValueError("provenance")
        if not isinstance(model, dict) or not isinstance(model.get("model_type"), str) or not model["model_type"].strip():
            raise ValueError("model")
        architectures = model.get("architectures")
        if not isinstance(architectures, list) or not architectures or any(not isinstance(value, str) or not value.strip() for value in architectures):
            raise ValueError("architectures")
        if not isinstance(validation, dict) or any(not isinstance(validation.get(name), dict) for name in ("structural", "functional", "evaluation")):
            raise ValueError("validation")
        serving = compatibility.get("serving") if isinstance(compatibility, dict) else None
        if not isinstance(compatibility, dict) or any(not isinstance(compatibility.get(name), dict) for name in ("transformers", "lm_eval", "lmms_eval")) or not isinstance(serving, dict) or serving.get("status") not in ("ready", "blocked", "stale"):
            raise ValueError("compatibility")
        if not isinstance(timestamps, dict) or any(not isinstance(timestamps.get(name), str) or not timestamps[name] for name in ("created_at", "validated_at")):
            raise ValueError("timestamps")
        if manifest["publication_state"] == "published" and (not isinstance(timestamps.get("published_at"), str) or not timestamps["published_at"]):
            raise ValueError("published timestamp")
        files = manifest["files"]
        if not isinstance(files, dict) or files.get("hash_algorithm") != "sha256" or not isinstance(files.get("total_bytes"), int) or files["total_bytes"] < 0:
            raise ValueError("files")
        expected_entries = files.get("entries")
        if not isinstance(expected_entries, list) or any(
            not isinstance(entry, dict)
            or not _safe_entry_path(entry.get("path"))
            or not _sha256(entry.get("sha256"))
            or not isinstance(entry.get("size_bytes"), int)
            or entry["size_bytes"] < 0
            for entry in expected_entries
        ):
            raise ValueError("entries")
        expected_paths = [entry["path"] for entry in expected_entries]
        if expected_paths != sorted(expected_paths) or len(expected_paths) != len(set(expected_paths)):
            raise ValueError("entry ordering")
    except (KeyError, TypeError, ValueError) as exc:
        raise _error("validation_failed", "manifest contract is invalid") from exc
    actual_entries, actual_total = _inventory(path, include_hash=full_hash)
    actual_paths = [entry["path"] for entry in actual_entries]
    if expected_paths != actual_paths or files["total_bytes"] != actual_total:
        raise _error("validation_failed", "manifest file inventory does not match asset")
    if [(entry["path"], entry["size_bytes"]) for entry in expected_entries] != [
        (entry["path"], entry["size_bytes"]) for entry in actual_entries
    ]:
        raise _error("validation_failed", "manifest file sizes do not match asset")
    if full_hash and [(entry["path"], entry["sha256"]) for entry in expected_entries] != [(entry["path"], entry["sha256"]) for entry in actual_entries]:
        raise _error("validation_failed", "manifest file hashes do not match asset")
    return manifest


def validate_published_asset(path: str, full_hash: bool = False) -> dict:
    return _validate_manifest(_directory(os.path.abspath(path), "validation_failed"), _load_manifest(os.path.abspath(path)), full_hash)


def validate_formal_published_model(model, root: str, full_hash: bool = False) -> dict:
    """Validate a core registry row as a formal, published asset."""
    if getattr(model, "source", None) != "published":
        raise _error("model_not_published", "model is not a formal published asset")
    root = _publication_root(root)
    path = _directory(getattr(model, "path", ""), "validation_failed")
    if os.path.dirname(path) != root:
        raise _error("validation_failed", "published model path is outside the publication root")
    manifest = validate_published_asset(path, full_hash=full_hash)
    if manifest.get("publication_id") != os.path.basename(path) or manifest.get("publication_state") != "published":
        raise _error("validation_failed", "published model registry does not match its manifest")
    return manifest


def commit_staging(staging: str, root: str, manifest: dict, register_fn: Callable[[str, dict], object]) -> dict:
    root = _publication_root(root, create=True)
    publication_id = _publication_id(manifest.get("publication_id"))
    with publication_lock(root) as root:
        staging_parent = _control_dir(root, ".staging")
        staging = _staging_dir(root, staging, publication_id)
        if os.stat(staging).st_dev != os.stat(root).st_dev:
            raise _error("cross_device_staging", "staging and publication root must share a filesystem")
        final = _asset_dir(root, publication_id, required=False)
        if final is not None:
            raise _error("publication_exists", "publication directory already exists")
        pending_manifest = json.loads(json.dumps(manifest))
        pending_manifest["publication_state"] = "registration_pending"
        pending_manifest.setdefault("timestamps", {})["published_at"] = None
        _validate_manifest(staging, pending_manifest, full_hash=True)
        atomic_write_json(os.path.join(staging, _MANIFEST_NAME), pending_manifest)
        _fsync_directory(staging)
        final = _child(root, publication_id)
        os.replace(staging, final)
        _fsync_directory(staging_parent)
        _fsync_directory(root)
        _fsync_directory(final)
        try:
            register_fn(_directory(final), pending_manifest)
        except Exception:
            raise
        pending_manifest["publication_state"] = "published"
        pending_manifest["timestamps"]["published_at"] = _now()
        atomic_write_json(os.path.join(final, _MANIFEST_NAME), pending_manifest)
    return pending_manifest


def _quarantine_locked(root: str, path: str) -> None:
    quarantine = _control_dir(root, ".quarantine", create=True)
    base = os.path.basename(path)
    destination = _child(quarantine, "%s-%d" % (base, int(time.time())))
    suffix = 1
    while os.path.lexists(destination):
        destination = _child(quarantine, "%s-%d-%d" % (base, int(time.time()), suffix))
        suffix += 1
    os.replace(path, destination)
    _fsync_directory(root)
    _fsync_directory(quarantine)


def _cleanup_staging_locked(root: str, active_check_fn: Callable[[str], bool | None] | None) -> int:
    if active_check_fn is None:
        return 0
    staging_path = _child(root, ".staging")
    if not os.path.lexists(staging_path):
        return 0
    staging_root = _control_dir(root, ".staging", create=False)
    cutoff = time.time() - _STALE_STAGING_SECONDS
    cleaned = 0
    for name in sorted(os.listdir(staging_root)):
        if name.startswith("."):
            continue
        try:
            publication_id = _publication_id(name)
            path = _asset_dir(staging_root, publication_id)
            if os.path.getmtime(path) > cutoff or active_check_fn(publication_id) is not False:
                continue
            diagnostics = _control_dir(root, ".diagnostics", create=True)
            atomic_write_json(os.path.join(diagnostics, "%s.json" % publication_id), {
                "publication_id": publication_id,
                "reason": "stale_inactive_staging",
                "recorded_at": _now(),
            })
            shutil.rmtree(path)
            _fsync_directory(staging_root)
            _fsync_directory(diagnostics)
            cleaned += 1
        except (OSError, PublicationError):
            continue
    return cleaned


def reconcile_publications(
    root: str,
    register_fn: Callable[[str, dict], object],
    active_check_fn: Callable[[str], bool | None] | None = None,
) -> dict:
    root = _publication_root(root, create=True)
    result = {"registered": 0, "published": 0, "quarantined": 0, "staging_cleaned": 0, "skipped": 0, "errors": []}
    with publication_lock(root) as root:
        try:
            result["staging_cleaned"] = _cleanup_staging_locked(root, active_check_fn)
        except PublicationError as exc:
            result["errors"].append({"asset": ".staging", "code": exc.code})
        names = [name for name in sorted(os.listdir(root)) if not name.startswith(".")]
    seen_ids = set()
    seen_paths = set()
    for name in names:
        try:
            _publication_id(name)
            with publication_lock(root) as root:
                path = _asset_dir(root, name, required=False)
                if path is None:
                    continue
                try:
                    manifest = validate_published_asset(path, full_hash=True)
                except PublicationError as exc:
                    _quarantine_locked(root, path)
                    result["quarantined"] += 1
                    result["errors"].append({"asset": name, "code": exc.code})
                    continue
                publication_id = manifest["publication_id"]
                real_path = _directory(path)
                if publication_id in seen_ids or real_path in seen_paths:
                    result["skipped"] += 1
                    continue
                seen_ids.add(publication_id)
                seen_paths.add(real_path)
                try:
                    serving = manifest["compatibility"]["serving"]
                    compatibility = inspect_serving_compatibility(manifest["model"]["architectures"], serving.get("tested_version"))
                    if compatibility != serving:
                        manifest["compatibility"]["serving"] = compatibility
                        atomic_write_json(os.path.join(path, _MANIFEST_NAME), manifest)
                    register_fn(real_path, manifest)
                except Exception as exc:
                    result["errors"].append({"asset": publication_id, "code": "recovery_failed", "detail": str(exc)[:200]})
                    continue
                result["registered"] += 1
                if manifest["publication_state"] == "registration_pending":
                    manifest["publication_state"] = "published"
                    manifest["timestamps"]["published_at"] = _now()
                    atomic_write_json(os.path.join(path, _MANIFEST_NAME), manifest)
                    result["published"] += 1
        except PublicationError as exc:
            result["errors"].append({"asset": name, "code": exc.code})
        except OSError as exc:
            result["errors"].append({"asset": name, "code": "filesystem_error", "detail": str(exc)[:200]})
    return result


def delete_published_asset(
    publication_id: str,
    root: str,
    reference_check_fn: Callable[[str, str], bool],
    delete_model_fn: Callable[[str], bool],
) -> dict:
    root = _publication_root(root)
    publication_id = _publication_id(publication_id)
    with publication_lock(root) as root:
        path = _asset_dir(root, publication_id)
        trash_root = _control_dir(root, ".trash", create=True)
        trash = _child(trash_root, publication_id)
        if os.path.lexists(trash):
            raise _error("trash_conflict", "trash destination already exists")
        real_path = _directory(path)
        if reference_check_fn(publication_id, real_path):
            raise _error("publication_referenced", "publication is still referenced")
        os.replace(path, trash)
        _fsync_directory(root)
        _fsync_directory(trash_root)
        try:
            deleted_model = bool(delete_model_fn(real_path))
        except Exception:
            os.replace(trash, path)
            _fsync_directory(root)
            _fsync_directory(trash_root)
            raise
        if not deleted_model:
            os.replace(trash, path)
            _fsync_directory(root)
            _fsync_directory(trash_root)
            raise _error("model_delete_failed", "model registration could not be deleted")
        shutil.rmtree(trash)
        _fsync_directory(root)
        _fsync_directory(trash_root)
    return {"publication_id": publication_id, "deleted": True}


def _registered_asset_is_referenced(candidate_id: str, path: str) -> bool:
    from app.extensions import db
    from app.model_gateway.models import ServingModelService
    from app.models import Model, Task

    real_path = os.path.realpath(os.path.abspath(path.rstrip(os.sep)))
    active_statuses = {"queued", "materializing", "validating", "registration_pending", "running"}
    core_session = Session(bind=db.engine)
    try:
        models = core_session.query(Model).filter(Model.source == "published").all()
        model = next(
            (row for row in models if os.path.realpath(os.path.abspath(row.path.rstrip(os.sep))) == real_path),
            None,
        )
        try:
            tasks = (
                core_session.query(Task)
                .filter(Task.task_type == "model_publication", Task.status.in_(active_statuses))
                .all()
            )
        except Exception:
            core_session.rollback()
            return True
        task_active = any(
            task.id == candidate_id
            or (isinstance(task.config, dict) and task.config.get("publication_id") == candidate_id)
            for task in tasks
        )
        model_id = model.id if model is not None else None
        core_session.rollback()
    finally:
        core_session.close()
    if task_active:
        return True

    gateway_session = Session(bind=db.engines["model_gateway"])
    try:
        services = gateway_session.query(ServingModelService).filter(ServingModelService.status != "deleted").all()
        referenced = False
        for service in services:
            service_path = os.path.realpath(os.path.abspath(service.model_path.rstrip(os.sep)))
            if service_path == real_path or (model_id is not None and service.model_id == model_id):
                referenced = True
                break
        gateway_session.rollback()
        return referenced
    finally:
        gateway_session.close()


def _delete_core_model_by_canonical_path(path: str) -> bool:
    from app.extensions import db
    from app.models import Model

    real_path = os.path.realpath(os.path.abspath(path.rstrip(os.sep)))
    core_session = Session(bind=db.engine)
    try:
        model = next(
            (row for row in core_session.query(Model)
             .filter(Model.source == "published")
             .order_by(Model.id.asc())
             .all()
             if os.path.realpath(os.path.abspath(row.path.rstrip(os.sep))) == real_path),
            None,
        )
        if model is None:
            core_session.rollback()
            return False
        core_session.delete(model)
        core_session.commit()
        return True
    except Exception:
        core_session.rollback()
        raise
    finally:
        core_session.close()


def delete_registered_published_asset(publication_id: str, root: str) -> dict:
    """Delete a formal asset only when no active task or Gateway row references it."""
    try:
        return delete_published_asset(
            publication_id,
            root,
            _registered_asset_is_referenced,
            _delete_core_model_by_canonical_path,
        )
    except PublicationError as exc:
        if exc.code == "publication_referenced":
            raise _error("asset_in_use", "publication is still referenced") from exc
        raise
