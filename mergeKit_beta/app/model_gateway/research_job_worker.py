"""Main-container Redis consumer for durable research jobs."""
from __future__ import annotations

import json
import os
import socket
import threading
import time

from app.model_gateway.embeddings import OnnxDenseEmbedder
from app.model_gateway.lifecycle import claim_research_job, complete_research_job, fail_research_job, reconcile_research_jobs
from app.model_gateway.models import ResearchJob, ServingModelService
from app.model_gateway.queue import RESEARCH_STREAM, enqueue_research_job
from app.model_gateway.research_execution import build_research_messages, call_research_model, validate_research_answer
from app.model_gateway.retrieval import retrieve_research_evidence


RESEARCH_CONSUMER_GROUP = "model-gateway-research-workers"
_encoder = None


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value or "")


def _get_encoder(config):
    global _encoder
    if _encoder is None:
        _encoder = OnnxDenseEmbedder(config.MERGEKIT_MODEL_GATEWAY_EMBEDDING_MODEL_PATH)
    return _encoder


def execute_research_job(session, job_id: str, worker_id: str, config=None) -> str:
    """Execute one claimed job and commit only citation-validated results."""
    from config import Config

    config = config or Config
    job = session.get(ResearchJob, job_id)
    if not job or job.status != "running" or job.lease_owner != worker_id:
        return "not_owner"
    try:
        with open(job.payload_path or "", encoding="utf-8") as handle:
            payload = json.load(handle)
        question = (payload.get("input") or "").strip()
        if not question:
            raise ValueError("missing_research_payload")
        service = session.get(ServingModelService, job.model_service_id)
        if not service or service.status != "running":
            raise ValueError("model_offline")
        evidence = retrieve_research_evidence(
            session, _get_encoder(config), job.api_key_id, job.file_ids or [], question
        )
        messages = build_research_messages(job.task_type, question, evidence, output_format=job.output_format)
        answer = call_research_model(service, messages)
        citations = validate_research_answer(answer, evidence, bool(job.require_citations), output_format=job.output_format)
        result_dir = os.path.join(config.MERGEKIT_MODEL_GATEWAY_RESEARCH_ROOT, "results")
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"{job.id}.json")
        temporary_path = result_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump({"answer": answer, "citations": citations, "evidence": evidence}, handle, ensure_ascii=False)
        os.replace(temporary_path, result_path)
        outcome = complete_research_job(session, job.id, worker_id, result_path=result_path)
        if outcome != "completed":
            try:
                os.unlink(result_path)
            except FileNotFoundError:
                pass
        return outcome
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return fail_research_job(session, job_id, worker_id, str(exc))


def consume_research_message(session, redis_client, message_id, fields: dict, worker_id: str) -> str:
    job_id = _text(fields.get(b"job_id", fields.get("job_id"))).strip()
    if not job_id:
        outcome = "invalid_message"
    else:
        claim = claim_research_job(session, job_id, worker_id)
        outcome = execute_research_job(session, job_id, worker_id) if claim == "claimed" else claim
    redis_client.xack(RESEARCH_STREAM, RESEARCH_CONSUMER_GROUP, message_id)
    return outcome


def ensure_research_consumer_group(redis_client) -> None:
    try:
        redis_client.xgroup_create(RESEARCH_STREAM, RESEARCH_CONSUMER_GROUP, id="0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def run_research_worker(app) -> None:
    """Run only inside the main container so the private vLLM loopback is reachable."""
    import redis
    from app.extensions import db

    config = app.config
    client = redis.Redis.from_url(config["MERGEKIT_MODEL_GATEWAY_REDIS_URL"], socket_connect_timeout=3, socket_timeout=5)
    client.ping()
    ensure_research_consumer_group(client)
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    next_reconcile = 0.0
    while True:
        now = time.monotonic()
        if now >= next_reconcile:
            with app.app_context():
                for job_id in reconcile_research_jobs(db.session):
                    enqueue_research_job(client, job_id, None, "research")
            next_reconcile = now + max(1, int(config.get("MERGEKIT_MODEL_GATEWAY_RESEARCH_RECONCILE_SECONDS", 30)))
        for _, messages in client.xreadgroup(RESEARCH_CONSUMER_GROUP, worker_id, {RESEARCH_STREAM: ">"}, count=1, block=1000):
            for message_id, fields in messages:
                with app.app_context():
                    consume_research_message(db.session, client, message_id, fields, worker_id)


def start_research_worker(app) -> None:
    if os.environ.get("MERGEKIT_MODEL_GATEWAY_RUNTIME_PROCESS") != "1":
        return
    if app.config.get("MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND") != "redis" or not app.config.get("MERGEKIT_MODEL_GATEWAY_REDIS_URL"):
        return
    if getattr(app, "_model_gateway_research_worker_started", False):
        return
    app._model_gateway_research_worker_started = True
    threading.Thread(target=run_research_worker, args=(app,), daemon=True, name="model-gateway-research").start()
