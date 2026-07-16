"""Worker phases for formal model publication.

Heavy model loading is deliberately isolated in this module's CLI child process.
The Flask worker only prepares files and records state.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from app.model_inspection import inspect_model
from app.model_publication import (
    PublicationError,
    build_manifest,
    commit_staging,
    inspect_serving_compatibility,
)


def _error(code: str, message: str) -> PublicationError:
    return PublicationError(code, "%s: %s" % (code, message))


def _root(params: dict) -> str:
    if params.get("publication_root"):
        return os.path.abspath(params["publication_root"])
    from config import Config

    return Config.PUBLISHED_MODELS_PATH


def _publication_id(task_id: str, params: dict) -> str:
    value = str(params.get("publication_id") or task_id).strip()
    if not value or value.startswith(".") or os.path.basename(value) != value:
        raise _error("invalid_publication_id", "publication id is invalid")
    return value


def _staging_path(task_id: str, params: dict) -> str:
    return os.path.join(_root(params), ".staging", _publication_id(task_id, params))


def _set_status(task_id: str, status: str, **kwargs) -> None:
    try:
        from flask import has_app_context

        if not has_app_context():
            return
        from app.repositories import task_set_status

        task_set_status(task_id, status, **kwargs)
    except Exception:
        # File materialization must not be made unsafe by a transient status write.
        return


def _cancelled(task_control: dict) -> bool:
    return bool((task_control or {}).get("aborted"))


def _copy_tree_cooperatively(source: str, destination: str, task_control: dict) -> None:
    """Copy independently, checking cancellation at every directory and file."""
    for current, directories, files in os.walk(source, followlinks=False):
        if _cancelled(task_control):
            raise _error("canceled", "publication canceled")
        relative = os.path.relpath(current, source)
        target_dir = destination if relative == "." else os.path.join(destination, relative)
        os.makedirs(target_dir, exist_ok=True)
        for directory in directories:
            source_dir = os.path.join(current, directory)
            if os.path.islink(source_dir):
                raise _error("materialization_failed", "source model contains a symlink")
            os.makedirs(os.path.join(target_dir, directory), exist_ok=True)
        for name in files:
            if _cancelled(task_control):
                raise _error("canceled", "publication canceled")
            source_file = os.path.join(current, name)
            if os.path.islink(source_file):
                raise _error("materialization_failed", "source model contains a symlink")
            shutil.copy2(source_file, os.path.join(target_dir, name), follow_symlinks=False)


def _estimate_bytes(source: str) -> int:
    return sum(
        os.path.getsize(os.path.join(current, name))
        for current, _directories, files in os.walk(source)
        for name in files
        if not os.path.islink(os.path.join(current, name))
    )


def _check_space(root: str, estimated_bytes: int) -> None:
    required = estimated_bytes + max(5 * 1024**3, estimated_bytes // 10)
    if shutil.disk_usage(root).free < required:
        raise _error("insufficient_disk_space", "publication root has insufficient usable disk space")


def _resolve_source(params: dict) -> str:
    source = (params.get("source_path") or "").strip()
    if not source or not os.path.isdir(source):
        raise _error("invalid_recipe", "resolved publication source is unavailable")
    return os.path.realpath(source)


def _recipe_model_paths(params: dict) -> list[str]:
    recipe_path = (params.get("recipe_path") or "").strip()
    try:
        with open(recipe_path, encoding="utf-8") as handle:
            recipe = json.load(handle)
    except (OSError, ValueError) as exc:
        raise _error("invalid_recipe", "managed recipe cannot be read") from exc
    paths = [os.path.realpath(path) for path in recipe.get("model_paths") or []]
    if not paths or any(not os.path.isdir(path) for path in paths):
        raise _error("invalid_recipe", "managed recipe has unavailable parents")
    return paths


def _materialize_recipe(task_id: str, params: dict, staging: str, progress: Callable, task_control: dict) -> None:
    from merge_manager import run_recipe_apply_task

    try:
        with open(params["recipe_path"], encoding="utf-8") as handle:
            recipe = json.load(handle)
    except (KeyError, OSError, ValueError) as exc:
        raise _error("invalid_recipe", "managed recipe cannot be read") from exc
    is_vlm = recipe.get("artifact_type") == "vlm" or bool(recipe.get("vlm_path"))
    output_dir = staging if not is_vlm else "%s.language" % staging

    result = run_recipe_apply_task(
        task_id,
        {**params, "recipe_id": params.get("recipe_id")},
        progress,
        task_control,
        skip_register=True,
        output_dir_override=output_dir,
    )
    if result.get("status") != "success":
        raise _error("materialization_failed", result.get("error", "recipe materialization failed"))
    if not is_vlm:
        return
    vlm_base = params.get("vlm_base_path") or recipe.get("vlm_path") or (recipe.get("vlm_base") or {}).get("source_path")
    if not vlm_base or not os.path.isdir(vlm_base):
        shutil.rmtree(output_dir, ignore_errors=True)
        raise _error("vlm_base_missing", "recipe has no complete VLM base")
    try:
        from evolution.vendor.vlm_merge.model_composition import materialize_full_vlm

        materialize_full_vlm(output_dir, vlm_base, staging, recipe.get("dtype", "bfloat16"))
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def validate_staging_model(staging: str) -> dict:
    inspection = inspect_model(staging)
    for current, directories, files in os.walk(staging, followlinks=False):
        if any(os.path.islink(os.path.join(current, item)) for item in directories + files):
            raise _error("validation_failed", "published assets must not contain symlinks")
    return {"inspection": inspection, "structural": {"status": "passed"}}


def _structural_result(staging: str, structural_validate_fn: Callable) -> tuple[object, dict]:
    value = structural_validate_fn(staging)
    if isinstance(value, dict):
        return value.get("inspection"), value.get("structural") or {"status": "passed"}
    return value, {"status": "passed"}


def run_model_publication_task(
    task_id: str,
    params: dict,
    progress: Callable[[int, str], None],
    task_control: dict,
    *,
    copy_fn: Callable = shutil.copytree,
    structural_validate_fn: Callable = validate_staging_model,
) -> dict:
    """Materialize and structurally validate a publication, stopping before GPU validation."""
    root = _root(params)
    staging = _staging_path(task_id, params)
    try:
        if _cancelled(task_control):
            raise _error("canceled", "publication canceled")
        source = _resolve_source(params) if params.get("source_type") != "recipe" else None
        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root, ".staging"), exist_ok=True)
        estimated = _estimate_bytes(source) if source else sum(_estimate_bytes(path) for path in _recipe_model_paths(params))
        _check_space(root, estimated)
        _set_status(task_id, "materializing", config_patch={"staging_path": staging})
        progress(5, "Materializing publication")
        if os.path.lexists(staging):
            raise _error("publication_exists", "staging directory already exists")
        if source is None:
            _materialize_recipe(task_id, params, staging, progress, task_control)
        elif copy_fn is shutil.copytree:
            _copy_tree_cooperatively(source, staging, task_control)
        else:
            copy_fn(source, staging)
        if _cancelled(task_control):
            raise _error("canceled", "publication canceled")
        progress(65, "Validating publication structure")
        inspection, structural = _structural_result(staging, structural_validate_fn)
        _set_status(
            task_id,
            "validating",
            config_patch={"staging_path": staging, "structural_validation": structural},
        )
        progress(80, "Awaiting explicit GPU validation")
        return {"status": "validating", "staging_path": staging, "inspection": inspection}
    except PublicationError as exc:
        if exc.code == "canceled":
            shutil.rmtree(staging, ignore_errors=True)
            _set_status(task_id, "canceled", error=str(exc), config_patch={"error_code": exc.code})
            return {"status": "canceled", "error_code": exc.code, "error": str(exc)}
        shutil.rmtree(staging, ignore_errors=True)
        _set_status(task_id, "failed", error=str(exc), config_patch={"error_code": exc.code})
        return {"status": "error", "error_code": exc.code, "error": str(exc)}
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        _set_status(task_id, "failed", error=str(exc), config_patch={"error_code": "materialization_failed"})
        return {"status": "error", "error_code": "materialization_failed", "error": str(exc)}


def _normalize_gpu_ids(gpu_ids: list[int]) -> list[int]:
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise _error("gpu_selection_required", "explicit physical GPU ids are required")
    try:
        normalized = [int(value) for value in gpu_ids]
    except (TypeError, ValueError) as exc:
        raise _error("gpu_selection_required", "GPU ids must be physical integers") from exc
    if len(normalized) != len(set(normalized)):
        raise _error("gpu_selection_required", "GPU ids must not repeat")
    if 2 in normalized:
        raise _error("protected_gpu", "GPU 2 is protected")
    return normalized


def publication_gpu_preflight(gpu_ids: list[int]) -> list[dict]:
    """Read-only nvidia-smi check immediately before child-process validation."""
    gpu_ids = _normalize_gpu_ids(gpu_ids)
    from core.gpu_topology import query_gpus

    topology = {gpu.index: gpu for gpu in query_gpus()}
    command = [
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
    snapshots = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            snapshots[int(parts[0])] = {
                "index": int(parts[0]), "uuid": parts[1],
                "memory_used_mib": int(float(parts[2])), "memory_total_mib": int(float(parts[3])),
            }
    missing = [gpu_id for gpu_id in gpu_ids if gpu_id not in snapshots or gpu_id not in topology]
    if missing:
        raise _error("gpu_selection_required", "selected physical GPU is unavailable")
    process_result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5,
    )
    selected_uuids = {snapshots[gpu_id]["uuid"] for gpu_id in gpu_ids}
    for line in process_result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] in selected_uuids:
            raise _error("gpu_busy", "selected GPU has an external compute process")
    return [snapshots[gpu_id] for gpu_id in gpu_ids]


def validate_model_functionally(staging: str, gpu_ids: list[int], _task_control: dict) -> dict:
    """Run real validation in a child whose CUDA visibility is explicitly scoped."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in gpu_ids)
    completed = subprocess.run(
        [sys.executable, __file__, "--functional-validation", staging],
        env=env, capture_output=True, text=True, timeout=20 * 60,
    )
    if completed.returncode:
        raise _error("validation_failed", (completed.stderr or completed.stdout or "functional validation failed")[-1000:])
    return json.loads(completed.stdout)


def _load_task(task_id: str):
    from app.extensions import db
    from app.models import Task

    return db.session.get(Task, task_id)


def run_publication_validation(
    task_id: str,
    gpu_ids: list[int],
    progress: Callable[[int, str], None],
    task_control: dict,
    *,
    functional_validate_fn: Callable = validate_model_functionally,
) -> dict:
    """Resume a persisted validating task and atomically commit it after functional validation."""
    gpu_ids = _normalize_gpu_ids(gpu_ids)
    if _cancelled(task_control):
        return {"status": "canceled", "error_code": "canceled"}
    task = _load_task(task_id)
    if task is None or task.task_type != "model_publication":
        raise _error("publication_missing", "publication task does not exist")
    if task.status != "validating":
        raise _error("invalid_publication_state", "publication is not awaiting validation")
    params = dict(task.config or {})
    staging = params.get("staging_path") or _staging_path(task_id, params)
    if not os.path.isdir(staging):
        raise _error("publication_missing", "publication staging is unavailable")
    progress(82, "Checking selected GPUs")
    gpu_snapshot = publication_gpu_preflight(gpu_ids)
    if _cancelled(task_control):
        return {"status": "canceled", "error_code": "canceled"}
    progress(86, "Running functional validation")
    functional = functional_validate_fn(staging, gpu_ids, task_control) or {"status": "passed"}
    if _cancelled(task_control):
        return {"status": "canceled", "error_code": "canceled"}
    inspection = inspect_model(staging)
    compatibility = {"serving": inspect_serving_compatibility(inspection.architectures)}
    manifest = build_manifest(
        staging,
        {**params, "task_id": task_id, "publication_id": _publication_id(task_id, params)},
        inspection,
        {"structural": params.get("structural_validation") or {"status": "passed"}, "functional": functional},
        compatibility,
    )
    # Cancellation is rejected from this point because commit_staging is atomic.
    _set_status(task_id, "registration_pending", config_patch={"commit_in_progress": True, "gpu_snapshot": gpu_snapshot})
    progress(96, "Committing publication")
    from app.repositories import model_register_published

    committed = commit_staging(staging, _root(params), manifest, model_register_published)
    _set_status(
        task_id,
        "completed",
        model_path=os.path.join(_root(params), committed["publication_id"]),
        config_patch={"commit_in_progress": False, "publication_id": committed["publication_id"]},
    )
    progress(100, "Publication complete")
    return {"status": "success", "publication_id": committed["publication_id"]}


def _functional_validation_worker(path: str) -> dict:
    """Task 7 can extend this child contract with real image/CMMMU fixtures."""
    inspection = inspect_model(path)
    if inspection.is_vlm:
        from PIL import Image
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(path, trust_remote_code=True)
        image = Image.new("RGB", (8, 8), "white")
        inputs = processor(text="Describe this image.", images=image, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=4)
        if not processor.batch_decode(output, skip_special_tokens=True)[0].strip():
            raise RuntimeError("VLM image validation returned no output")
        # The result shape reserves the evaluation channel for Task 7's CMMMU sample.
        return {"status": "passed", "text_generation": {"status": "passed"}, "image": {"status": "passed"}, "evaluation": {}}
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    inputs = tokenizer("Reply with one word: ready", return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=4)
    if not tokenizer.decode(output[0], skip_special_tokens=True).strip():
        raise RuntimeError("text validation returned no output")
    return {"status": "passed", "text_generation": {"status": "passed"}}


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--functional-validation":
    print(json.dumps(_functional_validation_worker(sys.argv[2])))
