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
import time
import re
from contextlib import nullcontext
from typing import Callable

from app.model_inspection import inspect_model
from app.model_publication import (
    PublicationError,
    _inventory,
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
    from flask import has_app_context

    if not has_app_context():
        raise RuntimeError("model publication status writes require Flask app context")
    from app.repositories import task_set_status

    if task_set_status(task_id, status, **kwargs) is None:
        raise _error("publication_missing", "publication task does not exist")


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
    from app.repositories import model_get_by_id

    model = model_get_by_id(str(params.get("model_id") or "").strip())
    if model is None or not os.path.isdir(model.path):
        raise _error("invalid_model", "managed source model is unavailable")
    return os.path.realpath(model.path)


def _resolve_recipe(params: dict) -> tuple[str, dict]:
    import merge_manager

    recipe_id = str(params.get("recipe_id") or "").strip()
    if not recipe_id or recipe_id.startswith(".") or os.path.basename(recipe_id) != recipe_id:
        raise _error("invalid_recipe", "managed recipe id is invalid")
    root = os.path.realpath(os.path.abspath(merge_manager.RECIPES_DIR))
    recipe_path = os.path.realpath(os.path.join(root, "%s.json" % recipe_id))
    if os.path.commonpath((root, recipe_path)) != root or not os.path.isfile(recipe_path):
        raise _error("invalid_recipe", "managed recipe is unavailable")
    recorded_path = str(params.get("recipe_path") or "").strip()
    if recorded_path and os.path.realpath(recorded_path) != recipe_path:
        raise _error("invalid_recipe", "managed recipe identity changed")
    try:
        with open(recipe_path, encoding="utf-8") as handle:
            recipe = json.load(handle)
    except (OSError, ValueError) as exc:
        raise _error("invalid_recipe", "managed recipe cannot be read") from exc
    if not isinstance(recipe, dict):
        raise _error("invalid_recipe", "managed recipe must be an object")
    return recipe_path, recipe


def _recipe_model_paths(params: dict) -> list[str]:
    _recipe_path, recipe = _resolve_recipe(params)
    paths = [os.path.realpath(path) for path in recipe.get("model_paths") or []]
    if not paths or any(not os.path.isdir(path) for path in paths):
        raise _error("invalid_recipe", "managed recipe has unavailable parents")
    return paths


def _materialize_recipe(task_id: str, params: dict, staging: str, progress: Callable, task_control: dict) -> None:
    from merge_manager import run_recipe_apply_task

    _recipe_path, recipe = _resolve_recipe(params)
    is_vlm = recipe.get("artifact_type") == "vlm" or bool(recipe.get("vlm_path"))
    output_dir = staging if not is_vlm else "%s.language" % staging

    result = run_recipe_apply_task(
        task_id,
        {**params, "recipe_id": params.get("recipe_id")},
        progress,
        task_control,
        skip_register=True,
        output_dir_override=output_dir,
        metadata_type_override="model_publication",
        metadata_extra={
            "publication_id": _publication_id(task_id, params),
            "display_name": params.get("display_name"),
            "source_type": "recipe",
        },
    )
    if result.get("status") != "success":
        raise _error("materialization_failed", result.get("error", "recipe materialization failed"))
    if not is_vlm:
        return
    vlm_base_model_id = str(params.get("vlm_base_model_id") or "").strip()
    if vlm_base_model_id:
        from app.repositories import model_get_by_id
        from app.model_inspection import resolve_vlm_base

        model = model_get_by_id(vlm_base_model_id)
        try:
            vlm_base = resolve_vlm_base(recipe, override_path=model.path).path if model is not None else None
        except ValueError:
            vlm_base = None
    else:
        from app.model_inspection import resolve_vlm_base

        try:
            vlm_base = resolve_vlm_base(recipe).path
        except ValueError:
            vlm_base = None
    if not vlm_base:
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


def _manifest_matches_inventory(manifest: dict, baseline: object) -> bool:
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        return False
    return baseline == {
        "files": files.get("entries"),
        "total_bytes": files.get("total_bytes"),
    }


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
        if _cancelled(task_control):
            raise _error("canceled", "publication canceled")
        files, total_bytes = _inventory(staging, include_hash=True)
        if _cancelled(task_control):
            raise _error("canceled", "publication canceled")
        _set_status(
            task_id,
            "validating",
            config_patch={
                "staging_path": staging,
                "structural_validation": structural,
                "staging_inventory": {"files": files, "total_bytes": total_bytes},
            },
        )
        progress(80, "Awaiting explicit GPU validation")
        return {"status": "validating"}
    except PublicationError as exc:
        if exc.code == "canceled":
            shutil.rmtree(staging, ignore_errors=True)
            _set_status(task_id, "canceled", error=str(exc), config_patch={"error_code": exc.code})
            return {"status": "canceled", "error_code": exc.code, "error": str(exc)}
        shutil.rmtree(staging, ignore_errors=True)
        return {"status": "error", "error_code": exc.code, "error": str(exc)}
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        return {"status": "error", "error_code": "materialization_failed", "error": str(exc)}


def _normalize_gpu_ids(gpu_ids: list[int]) -> list[int]:
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise _error("gpu_selection_required", "explicit physical GPU ids are required")
    if any(type(value) is not int for value in gpu_ids):
        raise _error("gpu_selection_required", "GPU ids must be physical integers")
    normalized = list(gpu_ids)
    if len(normalized) != len(set(normalized)):
        raise _error("gpu_selection_required", "GPU ids must not repeat")
    if 2 in normalized:
        raise _error("protected_gpu", "GPU 2 is protected")
    return normalized


def _nonnegative_decimal(value: object) -> int:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError("not a nonnegative decimal integer")
    return int(value)


def publication_gpu_preflight(gpu_ids: list[int], *, required_bytes: int) -> list[dict]:
    """Fail-closed GPU check with model bytes + 10%/1 GiB inference headroom."""
    gpu_ids = _normalize_gpu_ids(gpu_ids)
    from core.gpu_topology import query_gpus

    try:
        topology_rows = query_gpus()
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
    except Exception as exc:
        raise _error("gpu_preflight_failed", "GPU inventory query failed") from exc
    if result.returncode != 0:
        raise _error("gpu_preflight_failed", "GPU inventory query failed")

    uuid_pattern = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")
    topology = {}
    for gpu in topology_rows:
        if gpu.index in topology or gpu.mem_total_mib <= 0:
            raise _error("gpu_preflight_failed", "GPU topology is invalid")
        topology[gpu.index] = gpu
    snapshots = {}
    uuids = set()
    try:
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                raise ValueError("invalid GPU row")
            index = _nonnegative_decimal(parts[0])
            used = _nonnegative_decimal(parts[2])
            total = _nonnegative_decimal(parts[3])
            uuid = parts[1]
            if (
                index in snapshots
                or uuid in uuids
                or uuid_pattern.fullmatch(uuid) is None
                or total <= 0
                or not 0 <= used <= total
            ):
                raise ValueError("invalid GPU row")
            uuids.add(uuid)
            snapshots[index] = {
                "index": index,
                "uuid": uuid,
                "memory_used_mib": used,
                "memory_total_mib": total,
                "memory_free_mib": total - used,
            }
    except (TypeError, ValueError) as exc:
        raise _error("gpu_preflight_failed", "GPU inventory output is invalid") from exc

    selected = []
    for gpu_id in gpu_ids:
        snapshot = snapshots.get(gpu_id)
        gpu = topology.get(gpu_id)
        if snapshot is None or gpu is None or snapshot["memory_total_mib"] != gpu.mem_total_mib:
            raise _error("gpu_preflight_failed", "selected GPU inventory changed")
        selected.append((snapshot, gpu))

    try:
        requirement = int(required_bytes)
        if requirement <= 0:
            raise ValueError("nonpositive model size")
        model_mib = (requirement + 1024**2 - 1) // 1024**2
    except (TypeError, ValueError) as exc:
        raise _error("gpu_preflight_failed", "model memory requirement is invalid") from exc
    # Weight bytes plus 10% (at least 1 GiB) covers loading overhead; every
    # selected device also retains a 1 GiB CUDA/runtime reserve.
    required_mib = model_mib + max(1024, (model_mib + 9) // 10)
    if any(snapshot["memory_free_mib"] < 1024 for snapshot, _gpu in selected):
        raise _error("insufficient_gpu_memory", "selected GPU lacks runtime reserve")
    if sum(snapshot["memory_free_mib"] - 1024 for snapshot, _gpu in selected) < required_mib:
        raise _error("insufficient_gpu_memory", "selected GPUs lack validated model headroom")

    try:
        process_result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception as exc:
        raise _error("gpu_preflight_failed", "compute-process query failed") from exc
    if process_result.returncode != 0:
        raise _error("gpu_preflight_failed", "compute-process query failed")
    selected_uuids = {snapshots[gpu_id]["uuid"] for gpu_id in gpu_ids}
    for line in process_result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        try:
            valid = (
                len(parts) == 2
                and uuid_pattern.fullmatch(parts[0]) is not None
                and _nonnegative_decimal(parts[1]) > 0
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise _error("gpu_preflight_failed", "compute-process output is invalid")
        if parts[0] in selected_uuids:
            raise _error("gpu_busy", "selected GPU has an external compute process")
    return [snapshots[gpu_id] for gpu_id in gpu_ids]


def validate_model_functionally(staging: str, gpu_ids: list[int], task_control: dict) -> dict:
    """Run real validation in a child whose CUDA visibility is explicitly scoped."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in gpu_ids)
    process = subprocess.Popen(
        [sys.executable, "-m", "app.model_publication_tasks", "--functional-validation", staging],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    task_control["process"] = process
    deadline = time.monotonic() + 20 * 60
    try:
        while process.poll() is None:
            if _cancelled(task_control):
                process.terminate()
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                raise _error("canceled", "publication canceled")
            if time.monotonic() >= deadline:
                process.kill()
                process.communicate()
                raise _error("validation_failed", "functional validation timed out")
            time.sleep(0.1)
        stdout, stderr = process.communicate()
        if _cancelled(task_control):
            raise _error("canceled", "publication canceled")
        if process.returncode:
            raise _error("validation_failed", (stderr or stdout or "functional validation failed")[-1000:])
        try:
            result = json.loads(stdout)
        except (TypeError, ValueError) as exc:
            raise _error("validation_failed", "functional validation returned invalid JSON") from exc
        if not isinstance(result, dict) or result.get("status") != "passed":
            raise _error("validation_failed", "functional validation did not pass")
        return result
    finally:
        if task_control.get("process") is process:
            task_control["process"] = None


def _load_task(task_id: str):
    from app.extensions import db
    from app.models import Task

    return db.session.get(Task, task_id)


def remove_publication_staging(params: dict) -> None:
    """Remove only the Task 3-authorized staging directory for this publication."""
    from app.model_publication import _control_dir, _publication_root, _staging_dir

    root = _publication_root(_root(params), create=True)
    staging_parent = os.path.join(root, ".staging")
    if not os.path.lexists(staging_parent):
        return
    _control_dir(root, ".staging")
    staging = params.get("staging_path") or _staging_path(str(params.get("task_id") or ""), params)
    if not os.path.lexists(staging):
        return
    safe_staging = _staging_dir(root, staging, _publication_id(str(params.get("task_id") or ""), params))
    shutil.rmtree(safe_staging)


def _cancel_validation(task_id: str, params: dict) -> dict:
    remove_publication_staging({**params, "task_id": task_id})
    message = "canceled: publication canceled"
    _set_status(task_id, "canceled", error=message, config_patch={"error_code": "canceled"})
    return {"status": "canceled", "error_code": "canceled", "error": message}


def _validation_failure(task_id: str, exc: Exception) -> dict:
    code = exc.code if isinstance(exc, PublicationError) else "validation_failed"
    message = str(exc)
    _set_status(task_id, "validating", error=message, config_patch={"error_code": code, "validation_enqueued": False})
    return {"status": "validating", "error_code": code, "error": message}


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
    task = _load_task(task_id)
    if task is None or task.task_type != "model_publication":
        raise _error("publication_missing", "publication task does not exist")
    if task.status != "validating":
        raise _error("invalid_publication_state", "publication is not awaiting validation")
    params = dict(task.config or {})
    staging = params.get("staging_path") or _staging_path(task_id, params)
    if not os.path.isdir(staging):
        raise _error("publication_missing", "publication staging is unavailable")
    if _cancelled(task_control):
        return _cancel_validation(task_id, params)
    try:
        baseline = params.get("staging_inventory")
        files, total_bytes = _inventory(staging, include_hash=True)
        if baseline != {"files": files, "total_bytes": total_bytes}:
            raise _error("staging_changed", "publication staging changed after structural validation")
        progress(82, "Checking selected GPUs")
        gpu_snapshot = publication_gpu_preflight(gpu_ids, required_bytes=_estimate_bytes(staging))
        if _cancelled(task_control):
            return _cancel_validation(task_id, params)
        progress(86, "Running functional validation")
        functional = functional_validate_fn(staging, gpu_ids, task_control) or {"status": "passed"}
        if _cancelled(task_control):
            return _cancel_validation(task_id, params)
        inspection = inspect_model(staging)
        compatibility = {"serving": inspect_serving_compatibility(inspection.architectures)}
        validation = {
            "structural": params.get("structural_validation") or {"status": "passed"},
            "functional": functional,
        }
        if isinstance(functional.get("evaluation"), dict) and functional["evaluation"]:
            validation["evaluation"] = functional["evaluation"]
        manifest = build_manifest(
            staging,
            {**params, "task_id": task_id, "publication_id": _publication_id(task_id, params)},
            inspection,
            validation,
            compatibility,
        )
        if not _manifest_matches_inventory(manifest, baseline):
            raise _error("staging_changed", "publication staging changed during validation")
    except PublicationError as exc:
        if exc.code == "canceled":
            return _cancel_validation(task_id, params)
        return _validation_failure(task_id, exc)
    except Exception as exc:
        return _validation_failure(task_id, exc)

    lock = task_control.get("lock")
    with (lock if lock is not None else nullcontext()):
        from app.extensions import db

        db.session.expire_all()
        current = _load_task(task_id)
        if _cancelled(task_control) or (current is not None and current.status == "canceled"):
            return _cancel_validation(task_id, params)
        if current is None or current.status != "validating":
            raise _error("invalid_publication_state", "publication is not awaiting validation")
        _set_status(
            task_id,
            "registration_pending",
            config_patch={"commit_in_progress": True, "gpu_snapshot": gpu_snapshot, "error_code": None},
        )

    progress(96, "Committing publication")
    from app.repositories import model_register_published

    try:
        committed = commit_staging(staging, _root(params), manifest, model_register_published)
        _set_status(
            task_id,
            "completed",
            model_path=os.path.join(_root(params), committed["publication_id"]),
            config_patch={"commit_in_progress": False, "publication_id": committed["publication_id"], "error_code": None},
        )
    except Exception as exc:
        _set_status(
            task_id,
            "registration_pending",
            error=str(exc),
            config_patch={"commit_in_progress": False, "error_code": "registration_pending"},
        )
        return {"status": "registration_pending", "error_code": "registration_pending", "error": str(exc)}
    progress(100, "Publication complete")
    return {"status": "success", "publication_id": committed["publication_id"]}


def _functional_validation_worker(path: str) -> dict:
    """Load on visible CUDA devices and run smoke generation plus one CMMMU row."""
    inspection = inspect_model(path)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("functional validation requires visible CUDA devices")
    visible_count = len([value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()])
    if visible_count < 1:
        raise RuntimeError("functional validation requires explicit visible CUDA devices")
    device_map = "auto" if visible_count > 1 else "cuda"
    torch_dtype = torch.bfloat16
    if inspection.is_vlm:
        import tempfile
        from PIL import Image
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.eval()
        image = Image.new("RGB", (224, 224), "white")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
        if hasattr(processor, "apply_chat_template"):
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "<|image_pad|>\nDescribe this image."
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        input_device = getattr(model, "device", torch.device("cuda"))
        moved = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved[key] = value
            elif isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                moved[key] = value.to(input_device, dtype=torch_dtype)
            else:
                moved[key] = value.to(input_device)
        with torch.no_grad():
            output = model.generate(**moved, max_new_tokens=4)
        if not processor.batch_decode(output, skip_special_tokens=True)[0].strip():
            raise RuntimeError("VLM image validation returned no output")
        del model, processor
        torch.cuda.empty_cache()

        from merge_manager import run_lmms_eval_stream

        with tempfile.TemporaryDirectory(prefix="publication-cmmmu-") as output_dir:
            evaluation = run_lmms_eval_stream(
                path,
                output_dir,
                lambda *_args: None,
                0,
                100,
                task_control={},
                limit=1,
                hf_dataset="m-a-p/CMMMU",
                hf_subset="health_and_medicine",
                hf_split="val",
                num_gpus=visible_count,
                absolute_limit=1,
            )
        if not isinstance(evaluation, dict) or int(evaluation.get("samples") or 0) < 1:
            raise RuntimeError("CMMMU validation evaluated no samples")
        return {
            "status": "passed",
            "text_generation": {"status": "passed"},
            "image": {"status": "passed"},
            "evaluation": {"cmmmu": evaluation},
        }
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    inputs = tokenizer("Reply with one word: ready", return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=4)
    if not tokenizer.decode(output[0], skip_special_tokens=True).strip():
        raise RuntimeError("text validation returned no output")
    return {"status": "passed", "text_generation": {"status": "passed"}}


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--functional-validation":
    print(json.dumps(_functional_validation_worker(sys.argv[2])))
