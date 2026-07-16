"""vLLM process management for published model services."""
from __future__ import annotations

from datetime import datetime
import glob
import os
import secrets
import signal
import socket
import subprocess
import time
import urllib.request

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.extensions import db
from app.model_gateway.auth import hash_secret
from app.model_gateway.models import ServingModelService, ServingRequest
from core.process_manager import ProcessManager


RUNTIME_STATES = ("starting", "running", "stopping")
RECOVERY_REASON = "system_restarted_manual_recovery_required"
INFLIGHT_REQUEST_STATES = ("running", "streaming", "cancel_requested")
MAX_LIFECYCLE_ERROR_LENGTH = 2048


class ServiceStateError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def should_recover_services_on_start() -> bool:
    """Limit restart recovery to the process that owns the Flask runtime."""
    return os.environ.get("MERGEKIT_MODEL_GATEWAY_RUNTIME_PROCESS") == "1"


def _real(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _is_under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([_real(path), _real(root)]) == _real(root)
    except ValueError:
        return False


def validate_model_path(path: str, allowed_roots: list[str]) -> None:
    model_path = _real(path or "")
    if not os.path.isdir(model_path):
        raise ValueError("model path is not a directory")
    roots = [r for r in (allowed_roots or []) if r]
    if not roots or not any(_is_under(model_path, root) for root in roots):
        raise ValueError("model path is outside allowed roots")
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise ValueError("model config.json is missing")
    tokenizer_names = ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt")
    if not any(os.path.isfile(os.path.join(model_path, name)) for name in tokenizer_names):
        raise ValueError("model tokenizer file is missing")
    if not (glob.glob(os.path.join(model_path, "*.safetensors")) or glob.glob(os.path.join(model_path, "*.bin"))):
        raise ValueError("model weight file is missing")


def find_free_port(start: int, end: int, reserved: set[int] | None = None) -> int:
    reserved = reserved or set()
    for port in range(int(start), int(end) + 1):
        if port in reserved:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free serving port available")


def allowed_model_roots(config) -> list[str]:
    return [
        getattr(config, "MODEL_POOL_PATH", ""),
        getattr(config, "LOCAL_MODELS_PATH", ""),
        getattr(config, "MERGE_DIR", ""),
        getattr(config, "PUBLISHED_MODELS_PATH", ""),
        *(getattr(config, "LOCAL_MODELS_EXTRA_PATHS", None) or []),
    ]


def _validate_formal_service_asset(service: ServingModelService, config) -> None:
    """Revalidate formal assets before they reserve a GPU or spawn vLLM."""
    from app.model_publication import PublicationError, current_serving_compatibility, validate_formal_published_model
    from app.models import Model

    root = getattr(config, "PUBLISHED_MODELS_PATH", "")
    formal_path = bool(root and _is_under(service.model_path, root))
    if not service.model_id:
        if formal_path:
            raise PublicationError("asset_unavailable", "asset_unavailable: published asset is not bound to a core model")
        return
    core_session = Session(bind=db.engine)
    try:
        model = core_session.get(Model, service.model_id)
        if not model or model.source != "published":
            raise PublicationError("asset_unavailable", "asset_unavailable: formal published model is unavailable")
        if _real(service.model_path) != _real(model.path):
            raise PublicationError("asset_identity_mismatch", "asset_identity_mismatch: service path does not match its formal model")
        manifest = validate_formal_published_model(model, root, full_hash=True)
        core_session.rollback()
    finally:
        core_session.close()
    serving = current_serving_compatibility(manifest)
    if serving.get("status") != "ready":
        code = serving.get("reason_code") or serving.get("status") or "asset_unavailable"
        raise PublicationError(code, "%s: published asset is not selectable" % code)


def validate_gpu_availability(
    service: ServingModelService,
    query_fn=None,
    max_used_mib: int = 1024,
    min_free_mib: int = 4096,
) -> None:
    gpu_ids = service.gpu_ids or []
    if not gpu_ids:
        return
    if query_fn is None:
        from core.gpu_topology import query_gpus
        query_fn = query_gpus
    gpus = {gpu.index: gpu for gpu in query_fn()}
    for raw_id in gpu_ids:
        idx = int(raw_id)
        gpu = gpus.get(idx)
        if not gpu:
            raise ValueError(f"GPU {idx} is not visible")
        used_mib = max(0, int(gpu.mem_total_mib) - int(gpu.mem_free_mib))
        if used_mib > int(max_used_mib):
            raise ValueError(f"GPU {idx} already has {used_mib} MiB in use")
        if int(gpu.mem_free_mib) < int(min_free_mib):
            raise ValueError(f"GPU {idx} has only {gpu.mem_free_mib} MiB free")


def build_vllm_command(service: ServingModelService, config) -> list[str]:
    internal_key = (service.internal_api_key or "").strip()
    if not internal_key:
        raise ValueError("internal vLLM API key is missing")
    host = service.vllm_host or "127.0.0.1"
    if host != "127.0.0.1":
        raise ValueError("vLLM host must be 127.0.0.1")
    if not service.vllm_port:
        raise ValueError("vLLM port is missing")

    if getattr(config, "MERGEKIT_MODEL_GATEWAY_VLLM_USE_COMPAT_WRAPPER", True):
        cmd = [
            getattr(config, "MERGEKIT_MODEL_GATEWAY_PYTHON", "python"),
            "-m",
            "app.model_gateway.vllm_entrypoint",
            "serve",
            service.model_path,
        ]
    else:
        cmd = [
            getattr(config, "MERGEKIT_MODEL_GATEWAY_VLLM_BIN", "vllm"),
            "serve",
            service.model_path,
        ]
    cmd.extend([
        "--host",
        "127.0.0.1",
        "--port",
        str(service.vllm_port),
        "--served-model-name",
        service.served_model_name,
        "--api-key",
        internal_key,
        "--tensor-parallel-size",
        str(service.tensor_parallel_size or 1),
        "--gpu-memory-utilization",
        str(service.gpu_memory_utilization or 0.85),
        "--dtype",
        service.dtype or "auto",
        "--disable-log-requests",
        "--enable-request-id-headers",
    ])
    if service.max_model_len:
        cmd.extend(["--max-model-len", str(service.max_model_len)])
    if service.max_num_seqs:
        cmd.extend(["--max-num-seqs", str(service.max_num_seqs)])
    if service.max_num_batched_tokens:
        cmd.extend(["--max-num-batched-tokens", str(service.max_num_batched_tokens)])
    if service.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def mark_services_stopped_after_restart(db_session) -> int:
    services = (
        db_session.query(ServingModelService)
        .filter(ServingModelService.status.in_(RUNTIME_STATES))
        .all()
    )
    now = datetime.utcnow()
    for service in services:
        service.status = "stopped"
        service.vllm_pid = None
        service.vllm_pgid = None
        service.last_exit_reason = RECOVERY_REASON
        service.stopped_at = now
        db_session.add(service)
    db_session.commit()
    return len(services)


def recover_inflight_requests_after_restart(db_session) -> int:
    """Finalize requests whose in-memory execution disappeared during restart.

    The first serving slice intentionally does not persist request bodies or
    replay work. Replaying an in-flight request would risk duplicated model
    execution and token accounting, so restart recovery makes its terminal
    outcome explicit instead.
    """
    requests = (
        db_session.query(ServingRequest)
        .filter(ServingRequest.status.in_(INFLIGHT_REQUEST_STATES))
        .all()
    )
    now = datetime.utcnow()
    for req in requests:
        if req.status == "streaming":
            req.status = "failed"
            req.error_code = "stream_interrupted_by_restart"
            req.error_message = "Streaming response interrupted by service restart."
        elif req.status == "cancel_requested":
            req.status = "canceled"
            req.error_code = "canceled_by_restart"
            req.error_message = "Cancellation finalized after service restart; request was not replayed."
        else:
            req.status = "failed"
            req.error_code = "request_interrupted_by_restart"
            req.error_message = "Request interrupted by service restart before background execution completed."
        req.finished_at = now
        db_session.add(req)
    db_session.commit()
    return len(requests)


def _healthcheck(service: ServingModelService, timeout_s: int = 1) -> bool:
    req = urllib.request.Request(
        f"http://127.0.0.1:{service.vllm_port}/v1/models",
        headers={"Authorization": f"Bearer {service.internal_api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _reserved_ports(exclude_service_id: str | None = None) -> set[int]:
    q = db.session.query(ServingModelService).filter(ServingModelService.status.in_(RUNTIME_STATES))
    if exclude_service_id:
        q = q.filter(ServingModelService.id != exclude_service_id)
    return {svc.vllm_port for svc in q.all() if svc.vllm_port}


def _ensure_internal_key(service: ServingModelService) -> None:
    if service.internal_api_key:
        return
    service.internal_api_key = f"mk_internal_{secrets.token_urlsafe(32)}"
    service.internal_api_key_hash = hash_secret(service.internal_api_key)


def _with_model_gateway_pythonpath(env: dict[str, str], config) -> dict[str, str]:
    root = getattr(config, "PROJECT_ROOT", None) or os.getcwd()
    hooks_dir = os.path.join(root, "app", "model_gateway", "runtime_hooks")
    old = env.get("PYTHONPATH", "")
    # Keep spawn-based vLLM engine processes patched without exposing project
    # modules (notably queue.py) as stdlib import candidates.
    env["PYTHONPATH"] = hooks_dir if not old else hooks_dir + os.pathsep + old
    env["MERGEKIT_CLI_SCRIPT"] = "1"
    return env


def _bounded_lifecycle_error(exc: BaseException) -> str:
    try:
        message = str(exc).strip() or exc.__class__.__name__
    except Exception:
        message = exc.__class__.__name__
    return message[:MAX_LIFECYCLE_ERROR_LENGTH]


def _reload_service(service_id: str) -> ServingModelService | None:
    db.session.expire_all()
    return db.session.get(ServingModelService, service_id)


def _cas_service_transition(service_id: str, expected_status: str, **values) -> ServingModelService | None:
    changed = db.session.execute(
        update(ServingModelService)
        .where(ServingModelService.id == service_id, ServingModelService.status == expected_status)
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    if changed.rowcount != 1:
        db.session.rollback()
        return None
    db.session.commit()
    return _reload_service(service_id)


def _record_start_failure(service_id: str, exc: BaseException) -> None:
    """Best-effort compensation using only a fresh Gateway transaction."""
    session = None
    try:
        session = Session(bind=db.engines["model_gateway"])
        session.execute(
            update(ServingModelService)
            .where(
                ServingModelService.id == service_id,
                ServingModelService.status.in_(("starting", "running")),
            )
            .values(
                status="failed",
                vllm_pid=None,
                vllm_pgid=None,
                last_error=_bounded_lifecycle_error(exc),
                stopped_at=datetime.utcnow(),
            )
            .execution_options(synchronize_session=False)
        )
        session.commit()
    except Exception:
        if session is not None:
            try:
                session.rollback()
            except Exception:
                pass
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def start_service(service_id: str, config=None, timeout_s: int = 120) -> ServingModelService:
    from config import Config

    config = config or Config
    service = db.session.get(ServingModelService, service_id)
    if not service:
        raise ValueError("serving model service not found")
    if service.status == "deleted":
        raise ServiceStateError("service_deleted", "deleted service is terminal")
    if service.status == "running":
        return service
    if service.status not in ("stopped", "failed"):
        raise ServiceStateError("service_state_conflict", "service lifecycle transition is already in progress")

    claimed = db.session.execute(
        update(ServingModelService)
        .where(
            ServingModelService.id == service_id,
            ServingModelService.status.in_(("stopped", "failed")),
        )
        .values(status="starting", last_error=None)
        .execution_options(synchronize_session=False)
    )
    if claimed.rowcount != 1:
        db.session.rollback()
        db.session.expire_all()
        current = db.session.get(ServingModelService, service_id)
        if current and current.status == "running":
            return current
        if current and current.status == "deleted":
            raise ServiceStateError("service_deleted", "deleted service is terminal")
        raise ServiceStateError("service_state_conflict", "service lifecycle transition is already in progress")
    db.session.commit()
    db.session.expire_all()
    service = db.session.get(ServingModelService, service_id)

    proc = None
    proc_pgid = None
    try:
        _validate_formal_service_asset(service, config)
        validate_model_path(service.model_path, allowed_model_roots(config))
        validate_gpu_availability(
            service,
            max_used_mib=getattr(config, "MERGEKIT_MODEL_GATEWAY_GPU_MAX_USED_MIB", 1024),
            min_free_mib=getattr(config, "MERGEKIT_MODEL_GATEWAY_GPU_MIN_FREE_MIB", 4096),
        )
        if not service.vllm_port:
            service.vllm_port = find_free_port(
                getattr(config, "MERGEKIT_MODEL_GATEWAY_PORT_START", 18000),
                getattr(config, "MERGEKIT_MODEL_GATEWAY_PORT_END", 18999),
                _reserved_ports(exclude_service_id=service.id),
            )
        service.vllm_host = "127.0.0.1"
        _ensure_internal_key(service)

        log_dir = getattr(config, "MERGEKIT_MODEL_GATEWAY_LOG_DIR", os.path.join(os.getcwd(), "logs", "model_gateway"))
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{service.id}.log")
        env = _with_model_gateway_pythonpath(os.environ.copy(), config)
        env["MERGEKIT_MODEL_GATEWAY_SERVICE_ID"] = service.id
        env["MERGEKIT_MODEL_GATEWAY_INTERNAL_API_KEY"] = service.internal_api_key
        gpu_ids = service.gpu_ids or []
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)

        cmd = build_vllm_command(service, config)
        runtime_values = {
            "vllm_host": service.vllm_host,
            "vllm_port": service.vllm_port,
            "internal_api_key": service.internal_api_key,
            "internal_api_key_hash": service.internal_api_key_hash,
        }
        db.session.rollback()
        with open(log_path, "ab") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                env=env,
                cwd=getattr(config, "PROJECT_ROOT", None) or os.getcwd(),
                **ProcessManager.create_process_group_kwargs(),
            )

        try:
            proc_pgid = os.getpgid(proc.pid)
        except Exception:
            proc_pgid = proc.pid
        service = _cas_service_transition(
            service_id,
            "starting",
            **runtime_values,
            vllm_pid=proc.pid,
            vllm_pgid=proc_pgid,
        )
        if service is None:
            raise ServiceStateError("service_state_conflict", "service lifecycle state changed during start")

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                service = _cas_service_transition(
                    service_id,
                    "starting",
                    status="failed",
                    vllm_pid=None,
                    vllm_pgid=None,
                    last_error=f"vLLM exited early with code {proc.returncode}",
                    stopped_at=datetime.utcnow(),
                )
                if service is None:
                    raise ServiceStateError("service_state_conflict", "service lifecycle state changed during start")
                return service
            if _healthcheck(service):
                service = _cas_service_transition(
                    service_id,
                    "starting",
                    status="running",
                    started_at=datetime.utcnow(),
                )
                if service is None:
                    raise ServiceStateError("service_state_conflict", "service lifecycle state changed during start")
                return service
            time.sleep(2)

        _terminate_process_group(proc_pgid, proc.pid, timeout_s=10)
        service = _cas_service_transition(
            service_id,
            "starting",
            status="failed",
            vllm_pid=None,
            vllm_pgid=None,
            last_error="vLLM healthcheck timeout",
            stopped_at=datetime.utcnow(),
        )
        if service is None:
            raise ServiceStateError("service_state_conflict", "service lifecycle state changed during start")
        return service
    except Exception as exc:
        try:
            db.session.rollback()
        except Exception:
            pass
        if proc is not None:
            try:
                _terminate_process_group(proc_pgid or proc.pid, proc.pid, timeout_s=10)
            except Exception:
                pass
        _record_start_failure(service_id, exc)
        raise


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_file:
            fields = stat_file.read().rsplit(")", 1)[-1].lstrip().split()
        return not fields or fields[0] != "Z"
    except OSError:
        return False


def _pid_has_service_marker(pid: int, service_id: str) -> bool:
    env_path = f"/proc/{pid}/environ"
    if not os.path.exists(env_path):
        return False
    try:
        with open(env_path, "rb") as f:
            data = f.read().split(b"\x00")
    except OSError:
        return False
    marker = f"MERGEKIT_MODEL_GATEWAY_SERVICE_ID={service_id}".encode("utf-8")
    return marker in data


def _find_marked_service_pids(service_id: str) -> list[int]:
    """Find every live vLLM process that carries this service marker."""
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    pids = []
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if _pid_alive(pid) and _pid_has_service_marker(pid, service_id):
            pids.append(pid)
    return pids


def _kill_process_group(pgid: int | None, sig: signal.Signals) -> None:
    if not pgid:
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return


def _terminate_process_group(pgid: int | None, pid: int | None, timeout_s: int) -> None:
    _kill_process_group(pgid, signal.SIGTERM)
    deadline = time.time() + max(0, int(timeout_s))
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.5)
    if _pid_alive(pid):
        _kill_process_group(pgid, signal.SIGKILL)


def _terminate_service_processes(pids: list[int], timeout_s: int) -> None:
    pgids = set()
    for pid in pids:
        try:
            pgids.add(os.getpgid(pid))
        except OSError:
            continue
    for pgid in pgids:
        _kill_process_group(pgid, signal.SIGTERM)

    deadline = time.time() + max(0, int(timeout_s))
    while time.time() < deadline and any(_pid_alive(pid) for pid in pids):
        time.sleep(0.5)
    for pgid in pgids:
        _kill_process_group(pgid, signal.SIGKILL)


def stop_service(service_id: str, timeout_s: int = 30) -> ServingModelService:
    service = _reload_service(service_id)
    if not service:
        raise ValueError("serving model service not found")
    if service.status == "deleted":
        raise ServiceStateError("service_deleted", "deleted service is terminal")
    if service.status == "stopped":
        return service
    if service.status in ("starting", "stopping"):
        raise ServiceStateError("service_state_conflict", "service lifecycle transition is already in progress")
    if service.status == "failed":
        stopped = _cas_service_transition(
            service_id,
            "failed",
            status="stopped",
            vllm_pid=None,
            vllm_pgid=None,
            stopped_at=datetime.utcnow(),
        )
        if stopped is not None:
            return stopped
        current = _reload_service(service_id)
        if current and current.status == "deleted":
            raise ServiceStateError("service_deleted", "deleted service is terminal")
        raise ServiceStateError("service_state_conflict", "service lifecycle state changed during stop")
    if service.status != "running":
        raise ServiceStateError("service_state_conflict", "service cannot be stopped from its current state")

    service = _cas_service_transition(service_id, "running", status="stopping")
    if service is None:
        current = _reload_service(service_id)
        if current and current.status == "deleted":
            raise ServiceStateError("service_deleted", "deleted service is terminal")
        raise ServiceStateError("service_state_conflict", "service lifecycle state changed during stop")

    stored_pid = service.vllm_pid
    pids = _find_marked_service_pids(service.id)
    if stored_pid and _pid_alive(stored_pid):
        if not _pid_has_service_marker(stored_pid, service.id):
            failed = _cas_service_transition(
                service_id,
                "stopping",
                status="failed",
                last_error="stored PID does not match serving service marker; manual action required",
                stopped_at=datetime.utcnow(),
            )
            if failed is not None:
                return failed
            raise ServiceStateError("service_state_conflict", "service lifecycle state changed during stop")
        if stored_pid not in pids:
            pids.append(stored_pid)

    if not pids:
        stopped = _cas_service_transition(
            service_id,
            "stopping",
            status="stopped",
            vllm_pid=None,
            vllm_pgid=None,
            stopped_at=datetime.utcnow(),
        )
        if stopped is not None:
            return stopped
        raise ServiceStateError("service_state_conflict", "service lifecycle state changed during stop")

    _terminate_service_processes(pids, timeout_s=timeout_s)

    if _find_marked_service_pids(service.id):
        failed = _cas_service_transition(
            service_id,
            "stopping",
            status="failed",
            last_error="vLLM process did not exit after stop request; manual action required",
            stopped_at=datetime.utcnow(),
        )
        if failed is not None:
            return failed
        raise ServiceStateError("service_state_conflict", "service lifecycle state changed during stop")

    stopped = _cas_service_transition(
        service_id,
        "stopping",
        status="stopped",
        vllm_pid=None,
        vllm_pgid=None,
        stopped_at=datetime.utcnow(),
    )
    if stopped is not None:
        return stopped
    raise ServiceStateError("service_state_conflict", "service lifecycle state changed during stop")
