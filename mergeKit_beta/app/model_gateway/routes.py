"""HTTP routes for the published-model gateway."""
from __future__ import annotations

from datetime import datetime
from datetime import timedelta
import hashlib
import json
import os
import re
import threading
import uuid

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
import requests
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.extensions import db
from app.models import Model
from app.model_gateway.auth import (
    extract_bearer_token,
    find_active_api_key,
    generate_api_key,
    require_admin_token,
)
from app.model_gateway.models import (
    ServingApiKey,
    ServingModelService,
    ServingRequest,
    ServingUsageRecord,
    ResearchFile,
    ResearchJob,
)
from app.model_gateway.documents import save_upload, validate_public_source_url
from app.model_gateway.queue import enqueue_research_file, enqueue_research_job
from app.model_gateway.quotas import QuotaExceeded, release_quota, reserve_quota
from app.model_gateway.runtime import ServiceStateError, start_service, stop_service


model_gateway_bp = Blueprint("model_gateway", __name__)

ADMIN_PREFIX = "/api/model-gateway/admin"
SERVED_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SUPPORTED_CHAT_FIELDS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "stream",
    "stop",
    "presence_penalty",
    "frequency_penalty",
}
IMMEDIATE_CANCEL_STATES = {"pending", "queued", "paused_model_offline", "retrying"}
DEFERRED_CANCEL_STATES = {"running", "streaming"}
TERMINAL_REQUEST_STATES = {"success", "completed", "failed", "dead_letter", "expired", "canceled"}
RESEARCH_TASK_TYPES = {"document_qa", "summary", "compare", "extract"}


def _error(status: int, code: str, message: str):
    return jsonify({"error": {"code": code, "message": message}}), status


def _quota_error(exc: QuotaExceeded):
    response, status = _error(429, exc.code, "Quota exceeded; retry after the indicated interval")
    response.headers["Retry-After"] = str(exc.retry_after_seconds)
    response.headers["X-RateLimit-Limit"] = str(exc.limit)
    return response, status


def _quota_limit(name: str, default: int) -> int:
    return max(1, int(current_app.config.get(name, default) or default))


def _response_with_request_id(body: dict, request_id: str, status: int = 200):
    response = jsonify(body)
    response.status_code = status
    response.headers["X-Request-Id"] = request_id
    return response


def _parse_int_field(data: dict, name: str, default=None):
    value = data.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(name)


def _parse_float_field(data: dict, name: str, default=None):
    value = data.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(name)


def _admin_required():
    try:
        require_admin_token(current_app.config, request.headers.get("Authorization"))
    except RuntimeError:
        return _error(503, "serving_admin_token_not_configured", "Serving admin token is not configured")
    except PermissionError:
        return _error(401, "unauthorized", "Invalid admin token")
    return None


def _allowed_roots() -> list[str]:
    return [
        current_app.config.get("MODEL_POOL_PATH", ""),
        current_app.config.get("LOCAL_MODELS_PATH", ""),
        current_app.config.get("MERGE_DIR", ""),
        current_app.config.get("PUBLISHED_MODELS_PATH", ""),
        *(current_app.config.get("LOCAL_MODELS_EXTRA_PATHS") or []),
    ]


def _formal_published_asset(model: Model, full_hash: bool = False) -> dict:
    from app.model_publication import validate_formal_published_model

    return validate_formal_published_model(model, current_app.config.get("PUBLISHED_MODELS_PATH", ""), full_hash=full_hash)


def _asset_error(exc):
    return _error(409, exc.code, str(exc))


def _require_api_key():
    token = extract_bearer_token(request.headers.get("Authorization"))
    if not token:
        return None, _error(401, "unauthorized", "Missing API key")
    try:
        key = find_active_api_key(token)
    except ValueError:
        return None, _error(401, "unauthorized", "Invalid API key")
    if not key:
        return None, _error(401, "unauthorized", "Invalid API key")
    return key, None


def _key_allows_model(key: ServingApiKey, served_model_name: str) -> bool:
    allowlist = key.model_allowlist or []
    return not allowlist or served_model_name in allowlist


def _research_root() -> str:
    root = current_app.config.get("MERGEKIT_MODEL_GATEWAY_RESEARCH_ROOT")
    if not root:
        root = os.path.join(current_app.root_path, "..", "runtime", "model_gateway")
    return os.path.abspath(root)


def _research_expiry(now: datetime | None = None) -> datetime:
    hours = max(1 / 3600, float(current_app.config.get("MERGEKIT_MODEL_GATEWAY_SOURCE_TTL_HOURS", 24) or 24))
    return (now or datetime.utcnow()) + timedelta(hours=hours)


def _research_file_to_dict(file_row: ResearchFile) -> dict:
    return {
        "id": file_row.id,
        "name": file_row.original_name,
        "source_kind": file_row.source_kind,
        "source_url": file_row.source_url,
        "status": file_row.status,
        "expires_at": file_row.expires_at.isoformat(),
        "error_code": file_row.error_code,
    }


def _research_job_to_dict(job: ResearchJob) -> dict:
    data = {
        "id": job.id,
        "model": job.served_model_name,
        "task_type": job.task_type,
        "status": job.status,
        "file_ids": job.file_ids or [],
        "output_format": job.output_format,
        "require_citations": bool(job.require_citations),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "expires_at": job.expires_at.isoformat() if job.expires_at else None,
        "error_code": job.error_code,
    }
    if job.status == "completed" and job.result_path and os.path.isfile(job.result_path):
        try:
            with open(job.result_path, encoding="utf-8") as handle:
                result = json.load(handle)
            evidence = result.get("evidence") or []
            source_urls = {
                row.id: row.source_url
                for row in db.session.query(ResearchFile).filter(
                    ResearchFile.id.in_([item.get("file_id") for item in evidence if item.get("file_id")])
                ).all()
            }
            sources = []
            for item in evidence:
                source = {"file_id": item.get("file_id"), "locator": item.get("locator")}
                if source_urls.get(item.get("file_id")):
                    source["url"] = source_urls[item["file_id"]]
                sources.append(source)
            data["result"] = {
                "answer": result.get("answer", ""),
                "citations": result.get("citations") or [],
                "sources": sources,
            }
        except (OSError, ValueError, TypeError):
            data["error_code"] = data["error_code"] or "result_unavailable"
    return data


def _research_request_fingerprint(data: dict, file_ids: list, output_format: str) -> str:
    """Hash idempotency inputs without persisting the user's research text."""
    normalized = {
        "task_type": data.get("task_type"),
        "file_ids": file_ids,
        "input": data.get("input"),
        "output_format": output_format,
        "require_citations": bool(data.get("require_citations", True)),
    }
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _idempotent_research_job(key_id: str, served_model_name: str, idempotency_key: str):
    if not idempotency_key:
        return None
    return (
        db.session.query(ResearchJob)
        .filter_by(
            api_key_id=key_id,
            served_model_name=served_model_name,
            idempotency_key=idempotency_key,
        )
        .one_or_none()
    )


def _enqueue_research_job(job: ResearchJob) -> None:
    if current_app.config.get("MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND", "db") != "redis":
        return
    redis_url = current_app.config.get("MERGEKIT_MODEL_GATEWAY_REDIS_URL", "")
    if not redis_url:
        raise RuntimeError("research Redis URL is not configured")
    import redis

    client = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
    enqueue_research_job(client, job.id, job.model_service_id, job.task_type)


def _enqueue_research_file(file_row: ResearchFile) -> None:
    if current_app.config.get("MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND", "db") != "redis":
        return
    redis_url = current_app.config.get("MERGEKIT_MODEL_GATEWAY_REDIS_URL", "")
    if not redis_url:
        raise RuntimeError("research Redis URL is not configured")
    import redis

    client = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
    enqueue_research_file(client, file_row.id)


@model_gateway_bp.before_request
def _guard_admin_routes():
    if request.path.startswith(ADMIN_PREFIX):
        return _admin_required()
    return None


@model_gateway_bp.get("/api/model-gateway/admin/model-services")
def admin_list_model_services():
    services = db.session.query(ServingModelService).order_by(ServingModelService.created_at.desc()).all()
    return jsonify({"status": "success", "services": [svc.to_dict() for svc in services]})


@model_gateway_bp.get("/api/model-gateway/admin/publishable-models")
def admin_list_publishable_models():
    from app.model_publication import PublicationError, current_serving_compatibility

    models = db.session.query(Model).filter(Model.source == "published").order_by(Model.created_at.desc()).all()
    rows = []
    for model in models:
        selectable = False
        reason_code = None
        artifact_type = model.architecture or "text"
        try:
            manifest = _formal_published_asset(model, full_hash=False)
            artifact_type = manifest["artifact_type"]
            serving = current_serving_compatibility(manifest)
            selectable = serving.get("status") == "ready"
            reason_code = None if selectable else serving.get("reason_code") or serving.get("status")
        except PublicationError as exc:
            reason_code = exc.code
        rows.append({
            "model_id": model.id,
            "display_name": model.name,
            "artifact_type": artifact_type,
            "selectable": selectable,
            "blocked_reason_code": reason_code,
        })
    return jsonify({"status": "success", "models": rows})


@model_gateway_bp.post("/api/model-gateway/admin/model-services")
def admin_create_model_service():
    from app.model_publication import PublicationError, current_serving_compatibility, publication_lock

    data = request.get_json(silent=True) or {}
    model_id = (data.get("model_id") or "").strip()
    display_name = (data.get("display_name") or "").strip()
    served_model_name = (data.get("served_model_name") or "").strip()
    gpu_ids = data.get("gpu_ids") or []
    try:
        tp = _parse_int_field(data, "tensor_parallel_size", 1)
        gpu_mem = _parse_float_field(data, "gpu_memory_utilization", 0.85)
        max_model_len = _parse_int_field(data, "max_model_len")
        max_num_seqs = _parse_int_field(data, "max_num_seqs", 8)
        max_num_batched_tokens = _parse_int_field(data, "max_num_batched_tokens")
    except ValueError as exc:
        return _error(400, f"invalid_{exc.args[0]}", f"{exc.args[0]} must be numeric")

    if "model_path" in data:
        return _error(400, "invalid_request", "model_path is resolved from model_id")
    if not model_id or not display_name or not served_model_name:
        return _error(400, "invalid_request", "model_id, display_name and served_model_name are required")
    if not SERVED_NAME_RE.match(served_model_name):
        return _error(400, "invalid_served_model_name", "served_model_name supports letters, numbers, dot, underscore and dash")
    if not isinstance(gpu_ids, list) or not gpu_ids:
        return _error(400, "invalid_gpu_ids", "gpu_ids must be a non-empty list")
    if tp < 1 or tp > len(gpu_ids):
        return _error(400, "invalid_tensor_parallel_size", "tensor_parallel_size must be between 1 and selected GPU count")
    if gpu_mem < 0.50 or gpu_mem > 0.92:
        return _error(400, "invalid_gpu_memory_utilization", "gpu_memory_utilization must be between 0.50 and 0.92")

    root = current_app.config.get("PUBLISHED_MODELS_PATH", "")
    try:
        with publication_lock(root):
            core_session = Session(bind=db.engine)
            try:
                model = core_session.get(Model, model_id)
                if not model or model.source != "published":
                    raise PublicationError("asset_unavailable", "model_id is not a formal published asset")
                manifest = _formal_published_asset(model, full_hash=True)
                serving = current_serving_compatibility(manifest)
                if serving.get("status") != "ready":
                    code = serving.get("reason_code") or "asset_unavailable"
                    raise PublicationError(code, "published asset is not selectable")
                model_snapshot = {"id": model.id, "path": model.path}
                core_session.rollback()
            finally:
                core_session.close()

            gateway_session = Session(bind=db.engines["model_gateway"])
            try:
                if gateway_session.query(ServingModelService).filter(
                    ServingModelService.served_model_name == served_model_name,
                ).first():
                    return _error(409, "served_model_name_exists", "served_model_name already exists")
                service = ServingModelService(
                    model_id=model_snapshot["id"],
                    model_path=model_snapshot["path"],
                    display_name=display_name,
                    served_model_name=served_model_name,
                    model_type=manifest["artifact_type"],
                    backend_type="vllm",
                    status="stopped",
                    vllm_host="127.0.0.1",
                    gpu_ids=gpu_ids,
                    tensor_parallel_size=tp,
                    gpu_memory_utilization=gpu_mem,
                    dtype=(data.get("dtype") or "auto").strip() or "auto",
                    max_model_len=max_model_len,
                    max_num_seqs=max_num_seqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                    trust_remote_code=bool(data.get("trust_remote_code", False)),
                )
                gateway_session.add(service)
                gateway_session.commit()
                service_payload = service.to_dict()
            except Exception:
                gateway_session.rollback()
                raise
            finally:
                gateway_session.close()
    except PublicationError as exc:
        return _asset_error(exc)
    except IntegrityError:
        return _error(409, "served_model_name_exists", "served_model_name already exists")
    return jsonify({"status": "success", "service": service_payload}), 201


@model_gateway_bp.get("/api/model-gateway/admin/model-services/<service_id>")
def admin_get_model_service(service_id):
    service = db.session.get(ServingModelService, service_id)
    if not service:
        return _error(404, "not_found", "Model service not found")
    return jsonify({"status": "success", "service": service.to_dict()})


@model_gateway_bp.delete("/api/model-gateway/admin/model-services/<service_id>")
def admin_delete_model_service(service_id):
    claimed = db.session.execute(
        update(ServingModelService)
        .where(
            ServingModelService.id == service_id,
            ServingModelService.status.in_(("stopped", "failed")),
        )
        .values(status="deleted", vllm_pid=None, vllm_pgid=None)
        .execution_options(synchronize_session=False)
    )
    if claimed.rowcount != 1:
        db.session.rollback()
        db.session.expire_all()
        if db.session.get(ServingModelService, service_id) is None:
            return _error(404, "not_found", "Model service not found")
        return _error(409, "service_not_stopped", "Model service must be stopped before deletion")
    db.session.commit()
    db.session.expire_all()
    service = db.session.get(ServingModelService, service_id)
    return jsonify({"status": "success", "service": service.to_dict()})


@model_gateway_bp.post("/api/model-gateway/admin/model-services/<service_id>/start")
def admin_start_model_service(service_id):
    from app.model_publication import PublicationError

    try:
        service = start_service(service_id)
    except (PublicationError, ServiceStateError) as exc:
        return _error(409, exc.code, str(exc))
    except ValueError as exc:
        return _error(400, "start_failed", str(exc))
    except Exception as exc:
        return _error(500, "start_failed", str(exc))
    return jsonify({"status": "success", "service": service.to_dict()})


@model_gateway_bp.post("/api/model-gateway/admin/model-services/<service_id>/stop")
def admin_stop_model_service(service_id):
    try:
        service = stop_service(service_id)
    except ServiceStateError as exc:
        return _error(409, exc.code, str(exc))
    except ValueError as exc:
        return _error(400, "stop_failed", str(exc))
    except Exception as exc:
        return _error(500, "stop_failed", str(exc))
    return jsonify({"status": "success", "service": service.to_dict()})


@model_gateway_bp.post("/api/model-gateway/admin/api-keys")
def admin_create_api_key():
    data = request.get_json(silent=True) or {}
    owner_label = (data.get("owner_label") or "").strip()
    if not owner_label:
        return _error(400, "invalid_request", "owner_label is required")
    plaintext, digest, prefix, last4 = generate_api_key()
    key = ServingApiKey(
        key_hash=digest,
        prefix=prefix,
        last4=last4,
        owner_label=owner_label,
        model_allowlist=data.get("model_allowlist") or [],
        notes=data.get("notes"),
    )
    db.session.add(key)
    db.session.commit()
    return jsonify({"status": "success", "api_key": plaintext, "key": key.to_dict(), "last4": last4}), 201


@model_gateway_bp.get("/api/model-gateway/admin/api-keys")
def admin_list_api_keys():
    keys = db.session.query(ServingApiKey).order_by(ServingApiKey.created_at.desc()).all()
    return jsonify({"status": "success", "api_keys": [key.to_dict() for key in keys]})


def _set_api_key_status(key_id: str, status: str):
    key = db.session.get(ServingApiKey, key_id)
    if not key:
        return _error(404, "not_found", "API key not found")
    if key.status == "revoked" and status != "revoked":
        return _error(409, "key_not_mutable", "Revoked API keys cannot be restored")
    key.status = status
    db.session.add(key)
    db.session.commit()
    return jsonify({"status": "success", "api_key": key.to_dict()})


@model_gateway_bp.post("/api/model-gateway/admin/api-keys/<key_id>/disable")
def admin_disable_api_key(key_id):
    return _set_api_key_status(key_id, "disabled")


@model_gateway_bp.post("/api/model-gateway/admin/api-keys/<key_id>/revoke")
def admin_revoke_api_key(key_id):
    return _set_api_key_status(key_id, "revoked")


@model_gateway_bp.post("/api/model-gateway/files")
def create_research_file():
    key, err = _require_api_key()
    if err:
        return err
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return _error(400, "invalid_request", "file is required")
    try:
        quarantine_path, digest = save_upload(upload, _research_root())
    except ValueError as exc:
        return _error(413, "file_rejected", str(exc))
    upload_bytes = os.path.getsize(quarantine_path)
    try:
        reserve_quota(
            db.session, key.id, "import_bytes", amount=upload_bytes,
            limit=_quota_limit("MERGEKIT_MODEL_GATEWAY_IMPORT_BYTES_PER_DAY", 500 * 1024 * 1024),
            window_seconds=86400,
        )
    except QuotaExceeded as exc:
        db.session.rollback()
        try:
            os.unlink(quarantine_path)
        except FileNotFoundError:
            pass
        return _quota_error(exc)
    file_row = ResearchFile(
        api_key_id=key.id,
        original_name=os.path.basename(upload.filename),
        source_kind="upload",
        status="received",
        content_sha256=digest,
        quarantine_path=quarantine_path,
        expires_at=_research_expiry(),
    )
    db.session.add(file_row)
    db.session.commit()
    try:
        _enqueue_research_file(file_row)
    except Exception as exc:
        try:
            os.unlink(quarantine_path)
        except FileNotFoundError:
            pass
        db.session.delete(file_row)
        release_quota(db.session, key.id, "import_bytes", amount=upload_bytes, window_seconds=86400)
        db.session.commit()
        return _error(503, "queue_unavailable", str(exc))
    return jsonify({"status": "success", "file": _research_file_to_dict(file_row)}), 201


@model_gateway_bp.post("/api/model-gateway/sources/url")
def create_research_url_source():
    key, err = _require_api_key()
    if err:
        return err
    raw_url = ((request.get_json(silent=True) or {}).get("url") or "").strip()
    try:
        parsed = validate_public_source_url(raw_url)
    except ValueError as exc:
        return _error(400, "invalid_source_url", str(exc))
    name = os.path.basename(parsed.path) or parsed.hostname
    file_row = ResearchFile(
        api_key_id=key.id,
        original_name=name[:512],
        source_kind="url",
        source_url=parsed.geturl(),
        status="received",
        expires_at=_research_expiry(),
    )
    db.session.add(file_row)
    db.session.commit()
    try:
        _enqueue_research_file(file_row)
    except Exception as exc:
        db.session.delete(file_row)
        db.session.commit()
        return _error(503, "queue_unavailable", str(exc))
    return jsonify({"status": "accepted", "file": _research_file_to_dict(file_row)}), 202


@model_gateway_bp.get("/api/model-gateway/files/<file_id>")
def get_research_file(file_id):
    key, err = _require_api_key()
    if err:
        return err
    file_row = db.session.get(ResearchFile, file_id)
    if not file_row or file_row.api_key_id != key.id:
        return _error(404, "file_not_found", "Research file not found")
    return jsonify({"status": "success", "file": _research_file_to_dict(file_row)})


@model_gateway_bp.post("/api/model-gateway/research/jobs")
def create_research_job():
    key, err = _require_api_key()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    task_type = (data.get("task_type") or "").strip()
    served_model_name = (data.get("model") or "").strip()
    file_ids = data.get("file_ids") or []
    user_input = (data.get("input") or "").strip()
    output_format = (data.get("output_format") or "markdown").strip()
    if task_type not in RESEARCH_TASK_TYPES or not served_model_name or not user_input:
        return _error(400, "invalid_request", "model, supported task_type and input are required")
    if output_format not in {"markdown", "json"}:
        return _error(400, "invalid_request", "output_format must be markdown or json")
    if not isinstance(file_ids, list) or not file_ids or len(file_ids) > 10 or len(file_ids) != len(set(file_ids)):
        return _error(400, "invalid_request", "file_ids must contain between 1 and 10 files")
    idempotency_key = (request.headers.get("Idempotency-Key") or "").strip() or None
    fingerprint = _research_request_fingerprint(data, file_ids, output_format)
    existing = _idempotent_research_job(key.id, served_model_name, idempotency_key)
    if existing:
        if existing.request_fingerprint != fingerprint:
            return _error(409, "idempotency_conflict", "Idempotency-Key was already used with a different request")
        return jsonify({"status": "accepted", "idempotent_replay": True, "job": _research_job_to_dict(existing)}), 202
    service, service_err = _find_running_service_for_key(key, served_model_name)
    if service_err:
        return service_err
    files = db.session.query(ResearchFile).filter(
        ResearchFile.api_key_id == key.id,
        ResearchFile.id.in_(file_ids),
    ).all()
    if len(files) != len(set(file_ids)):
        return _error(404, "file_not_found", "One or more research files were not found")
    if any(file_row.status != "ready" for file_row in files):
        return _error(409, "source_not_ready", "Research sources must complete scanning and parsing before use")
    try:
        reserve_quota(
            db.session, key.id, "active_research_jobs", amount=1,
            limit=_quota_limit("MERGEKIT_MODEL_GATEWAY_MAX_ACTIVE_RESEARCH_JOBS", 1), window_seconds=0,
        )
        reserve_quota(
            db.session, key.id, "research_submissions", amount=1,
            limit=_quota_limit("MERGEKIT_MODEL_GATEWAY_RESEARCH_SUBMISSIONS_PER_HOUR", 4), window_seconds=3600,
        )
    except QuotaExceeded as exc:
        db.session.rollback()
        return _quota_error(exc)
    job = ResearchJob(
        id=str(uuid.uuid4()),
        api_key_id=key.id,
        model_service_id=service.id,
        served_model_name=served_model_name,
        task_type=task_type,
        status="queued",
        file_ids=file_ids,
        output_format=output_format,
        require_citations=bool(data.get("require_citations", True)),
        idempotency_key=idempotency_key,
        request_fingerprint=fingerprint,
        expires_at=min(file_row.expires_at for file_row in files),
    )
    payload_dir = os.path.join(_research_root(), "payloads")
    os.makedirs(payload_dir, exist_ok=True)
    job.payload_path = os.path.join(payload_dir, f"{job.id}.json")
    with open(job.payload_path, "x", encoding="utf-8") as handle:
        json.dump({"input": user_input, "file_ids": file_ids}, handle, ensure_ascii=False)
    db.session.add(job)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        try:
            os.unlink(job.payload_path)
        except FileNotFoundError:
            pass
        existing = _idempotent_research_job(key.id, served_model_name, idempotency_key)
        if existing and existing.request_fingerprint == fingerprint:
            return jsonify({"status": "accepted", "idempotent_replay": True, "job": _research_job_to_dict(existing)}), 202
        return _error(409, "idempotency_conflict", "Idempotency-Key was already used with a different request")
    try:
        _enqueue_research_job(job)
    except Exception as exc:
        if job.payload_path:
            try:
                os.unlink(job.payload_path)
            except FileNotFoundError:
                pass
        db.session.delete(job)
        release_quota(db.session, key.id, "active_research_jobs", amount=1, window_seconds=0)
        release_quota(db.session, key.id, "research_submissions", amount=1, window_seconds=3600)
        db.session.commit()
        return _error(503, "queue_unavailable", str(exc))
    return jsonify({"status": "accepted", "job": _research_job_to_dict(job)}), 202


@model_gateway_bp.get("/api/model-gateway/research/jobs/<job_id>")
def get_research_job(job_id):
    key, err = _require_api_key()
    if err:
        return err
    job = db.session.get(ResearchJob, job_id)
    if not job or job.api_key_id != key.id:
        return _error(404, "job_not_found", "Research job not found")
    return jsonify({"status": "success", "job": _research_job_to_dict(job)})


@model_gateway_bp.post("/api/model-gateway/research/jobs/<job_id>/cancel")
def cancel_research_job(job_id):
    key, err = _require_api_key()
    if err:
        return err
    job = db.session.get(ResearchJob, job_id)
    if not job or job.api_key_id != key.id:
        return _error(404, "job_not_found", "Research job not found")
    if job.status in {"received", "queued", "paused_model_offline", "retrying"}:
        job.status = "canceled"
        job.finished_at = datetime.utcnow()
        job.error_code = "user_canceled"
        if job.payload_path:
            try:
                os.unlink(job.payload_path)
            except FileNotFoundError:
                pass
            job.payload_path = None
        release_quota(db.session, key.id, "active_research_jobs", amount=1, window_seconds=0)
        db.session.add(job)
        db.session.commit()
        return jsonify({"status": "success", "job": _research_job_to_dict(job)})
    if job.status in {"running", "streaming"}:
        job.status = "cancel_requested"
        db.session.add(job)
        db.session.commit()
        return jsonify({"status": "accepted", "job": _research_job_to_dict(job)}), 202
    return _error(409, "job_not_cancelable", f"Research job status {job.status} cannot be canceled")


@model_gateway_bp.get("/v1/models")
def v1_models():
    key, err = _require_api_key()
    if err:
        return err
    services = db.session.query(ServingModelService).filter_by(status="running").all()
    data = [
        {"id": svc.served_model_name, "object": "model", "owned_by": "mergekit-beta"}
        for svc in services
        if _key_allows_model(key, svc.served_model_name)
    ]
    db.session.commit()
    return jsonify({"object": "list", "data": data})


def _find_running_service_for_key(key: ServingApiKey, served_model_name: str):
    if not _key_allows_model(key, served_model_name):
        return None, _error(403, "model_not_allowed", "API key is not allowed to use this model")
    service = db.session.query(ServingModelService).filter_by(
        served_model_name=served_model_name,
        status="running",
    ).first()
    if not service:
        return None, _error(503, "model_not_running", "Model is not running")
    return service, None


def _mark_request_failed(req_row: ServingRequest, message: str) -> None:
    if req_row.status == "canceled":
        return
    req_row.status = "failed"
    req_row.error_message = message
    req_row.finished_at = datetime.utcnow()
    db.session.add(req_row)
    db.session.commit()


def _abort_upstream_request(service: ServingModelService, request_id: str) -> bool:
    response = requests.post(
        f"http://127.0.0.1:{service.vllm_port}/internal/model-gateway/abort/chatcmpl-{request_id}",
        headers={"Authorization": f"Bearer {service.internal_api_key}"},
        timeout=5,
    )
    return 200 <= response.status_code < 300


def _request_to_dict(req_row: ServingRequest) -> dict:
    return {
        "id": req_row.id,
        "model": req_row.served_model_name,
        "status": req_row.status,
        "request_type": req_row.request_type,
        "stream": bool(req_row.stream),
        "created_at": req_row.created_at.isoformat() if req_row.created_at else None,
        "finished_at": req_row.finished_at.isoformat() if req_row.finished_at else None,
        "error_code": req_row.error_code,
        "error_message": req_row.error_message,
    }


def _usage_to_dict(req_row: ServingRequest) -> dict:
    usage = (
        db.session.query(ServingUsageRecord)
        .filter_by(request_id=req_row.id)
        .order_by(ServingUsageRecord.created_at.desc())
        .first()
    )
    if not usage:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "usage_source": "not_recorded",
        }
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "usage_source": usage.usage_source,
    }


def _complete_non_streaming_request(
    app,
    request_id: str,
    api_key_id: str,
    model_service_id: str,
    served_model_name: str,
    url: str,
    payload: dict,
    headers: dict,
    result_holder: dict,
    done: threading.Event,
) -> None:
    try:
        upstream = requests.post(url, json=payload, headers=headers, timeout=300)
        body = upstream.json()
    except Exception as exc:
        with app.app_context():
            req_row = db.session.get(ServingRequest, request_id)
            if req_row:
                _mark_request_failed(req_row, str(exc))
        result_holder["error"] = str(exc)
        done.set()
        return

    with app.app_context():
        req_row = db.session.get(ServingRequest, request_id)
        if req_row:
            if upstream.status_code >= 400:
                if req_row.status != "canceled":
                    req_row.status = "failed"
                    req_row.error_message = upstream.text
                    req_row.finished_at = datetime.utcnow()
                    db.session.add(req_row)
                    db.session.commit()
            else:
                usage = body.get("usage") or {}
                if req_row.status != "canceled":
                    req_row.status = "success"
                    req_row.finished_at = datetime.utcnow()
                    db.session.add(ServingUsageRecord(
                        request_id=req_row.id,
                        api_key_id=api_key_id,
                        model_service_id=model_service_id,
                        served_model_name=served_model_name,
                        prompt_tokens=int(usage.get("prompt_tokens") or 0),
                        completion_tokens=int(usage.get("completion_tokens") or 0),
                        total_tokens=int(usage.get("total_tokens") or 0),
                        usage_source="vllm_response",
                    ))
                    db.session.add(req_row)
                    db.session.commit()

    result_holder["body"] = body
    result_holder["status_code"] = upstream.status_code
    done.set()


@model_gateway_bp.get("/v1/requests/<request_id>")
def v1_get_request(request_id):
    key, err = _require_api_key()
    if err:
        return err
    req_row = db.session.get(ServingRequest, request_id)
    if not req_row or req_row.api_key_id != key.id:
        return _error(404, "request_not_found", "Request not found")
    return jsonify({
        "object": "serving.request",
        "request": _request_to_dict(req_row),
        "usage": _usage_to_dict(req_row),
    })


@model_gateway_bp.post("/v1/requests/<request_id>/cancel")
def v1_cancel_request(request_id):
    key, err = _require_api_key()
    if err:
        return err
    req_row = db.session.get(ServingRequest, request_id)
    if not req_row or req_row.api_key_id != key.id:
        return _error(404, "request_not_found", "Request not found")

    if req_row.status in IMMEDIATE_CANCEL_STATES:
        req_row.status = "canceled"
        req_row.error_code = "user_canceled"
        req_row.finished_at = datetime.utcnow()
        db.session.add(req_row)
        db.session.commit()
        return jsonify({"status": "success", "request": _request_to_dict(req_row)})

    if req_row.status in DEFERRED_CANCEL_STATES:
        service = db.session.get(ServingModelService, req_row.model_service_id)
        if not service or service.status != "running" or not service.internal_api_key:
            return _error(503, "model_not_running", "Model is not running")
        try:
            aborted = _abort_upstream_request(service, req_row.id)
        except requests.RequestException as exc:
            return _error(502, "upstream_abort_failed", str(exc))
        if not aborted:
            return _error(502, "upstream_abort_failed", "vLLM rejected the abort request")
        req_row.status = "canceled"
        req_row.error_code = "user_canceled"
        req_row.finished_at = datetime.utcnow()
        db.session.add(req_row)
        db.session.commit()
        return jsonify({"status": "success", "request": _request_to_dict(req_row)})

    if req_row.status in TERMINAL_REQUEST_STATES or req_row.status == "cancel_requested":
        return _error(409, "request_not_cancelable", f"Request status {req_row.status} cannot be canceled")

    return _error(409, "request_not_cancelable", f"Request status {req_row.status} cannot be canceled")


@model_gateway_bp.post("/v1/chat/completions")
def v1_chat_completions():
    key, err = _require_api_key()
    if err:
        return err
    payload = request.get_json(silent=True) or {}
    extra_fields = set(payload) - SUPPORTED_CHAT_FIELDS
    if extra_fields:
        return _error(400, "unsupported_request_fields", "Unsupported fields: " + ", ".join(sorted(extra_fields)))
    served_model_name = (payload.get("model") or "").strip()
    if not served_model_name:
        return _error(400, "invalid_request", "model is required")
    if not isinstance(payload.get("messages"), list) or not payload.get("messages"):
        return _error(400, "invalid_request", "messages must be a non-empty list")

    service, err = _find_running_service_for_key(key, served_model_name)
    if err:
        return err
    if not service.internal_api_key:
        return _error(503, "model_runtime_not_ready", "Model runtime internal key is missing")

    request_id = (request.headers.get("X-Request-Id") or "").strip()
    if request_id:
        try:
            request_id = str(uuid.UUID(request_id))
        except ValueError:
            return _error(400, "invalid_request_id", "X-Request-Id must be a UUID")
        if db.session.get(ServingRequest, request_id):
            return _error(409, "request_id_exists", "X-Request-Id already exists")

    try:
        reserve_quota(
            db.session, key.id, "chat_requests", amount=1,
            limit=_quota_limit("MERGEKIT_MODEL_GATEWAY_CHAT_REQUESTS_PER_MINUTE", 20), window_seconds=60,
        )
    except QuotaExceeded as exc:
        db.session.rollback()
        return _quota_error(exc)

    req_row = ServingRequest(
        id=request_id or None,
        api_key_id=key.id,
        model_service_id=service.id,
        served_model_name=served_model_name,
        request_type="chat",
        status="running",
        stream=bool(payload.get("stream")),
    )
    db.session.add(req_row)
    db.session.commit()

    headers = {
        "Authorization": f"Bearer {service.internal_api_key}",
        "X-Request-Id": req_row.id,
    }
    url = f"http://127.0.0.1:{service.vllm_port}/v1/chat/completions"

    if payload.get("stream"):
        try:
            upstream = requests.post(url, json=payload, headers=headers, timeout=None, stream=True)
        except Exception as exc:
            _mark_request_failed(req_row, str(exc))
            return _error(502, "upstream_error", str(exc))
        if upstream.status_code >= 400:
            _mark_request_failed(req_row, upstream.text)
            return Response(upstream.content, status=upstream.status_code, mimetype=upstream.headers.get("content-type", "text/event-stream"))

        def generate():
            try:
                for chunk in upstream.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
                db.session.expire_all()
                current = db.session.get(ServingRequest, req_row.id)
                if current and current.status != "canceled":
                    current.status = "success"
                    current.finished_at = datetime.utcnow()
                    db.session.add(ServingUsageRecord(
                        request_id=current.id,
                        api_key_id=key.id,
                        model_service_id=service.id,
                        served_model_name=served_model_name,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        usage_source="stream_usage_unavailable",
                    ))
                    db.session.add(current)
                    db.session.commit()
            except Exception as exc:
                _mark_request_failed(req_row, str(exc))
                raise

        response = Response(
            stream_with_context(generate()),
            status=upstream.status_code,
            mimetype=upstream.headers.get("content-type", "text/event-stream"),
        )
        response.headers["X-Request-Id"] = req_row.id
        return response

    result_holder = {}
    done = threading.Event()
    app = current_app._get_current_object()
    worker = threading.Thread(
        target=_complete_non_streaming_request,
        args=(app, req_row.id, key.id, service.id, served_model_name, url, payload, headers, result_holder, done),
        daemon=True,
    )
    worker.start()

    wait_s = float(current_app.config.get("MERGEKIT_MODEL_GATEWAY_SYNC_WAIT_SECONDS", 60) or 0)
    if not done.wait(max(0.0, wait_s)):
        return _response_with_request_id({
            "object": "serving.request",
            "status": "running",
            "request_id": req_row.id,
            "status_url": f"/v1/requests/{req_row.id}",
            "cancel_url": f"/v1/requests/{req_row.id}/cancel",
        }, req_row.id, 202)

    if "error" in result_holder:
        return _error(502, "upstream_error", result_holder["error"])
    return _response_with_request_id(
        result_holder.get("body") or {},
        req_row.id,
        int(result_holder.get("status_code") or 200),
    )
