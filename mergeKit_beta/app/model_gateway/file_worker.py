"""Dedicated no-GPU Redis Streams worker for scan-first file processing."""
from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("MERGEKIT_CLI_SCRIPT", "1")

from flask import Flask

from app.extensions import db
from app.model_gateway.file_processor import process_research_file
from app.model_gateway.models import ResearchFile
from app.model_gateway.queue import FILE_CONSUMER_GROUP, FILE_STREAM, enqueue_research_file


CONDA_CXX_RUNTIME = "/opt/conda/envs/mergenetic/lib/libstdc++.so.6"


def require_worker_preload() -> None:
    """Keep pip ONNX Runtime from loading an incompatible system C++ runtime."""
    if CONDA_CXX_RUNTIME not in os.environ.get("LD_PRELOAD", "").split(":"):
        raise RuntimeError("model gateway worker requires the Conda C++ runtime preload")


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value or "")


def consume_file_message(session, redis_client, message_id, fields: dict) -> str:
    file_id = _text(fields.get(b"file_id", fields.get("file_id"))).strip()
    if not file_id:
        outcome = "invalid_message"
    else:
        outcome = process_research_file(session, file_id)
    redis_client.xack(FILE_STREAM, FILE_CONSUMER_GROUP, message_id)
    return outcome


def reconcile_received_files(session, redis_client) -> list[str]:
    """DB is authoritative, so a restart can safely re-deliver received files."""
    files = (
        session.query(ResearchFile)
        .filter_by(source_kind="upload", status="received")
        .order_by(ResearchFile.created_at.asc())
        .all()
    )
    for source in files:
        enqueue_research_file(redis_client, source.id)
    return [source.id for source in files]


def ensure_file_consumer_group(redis_client) -> None:
    try:
        redis_client.xgroup_create(FILE_STREAM, FILE_CONSUMER_GROUP, id="0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def create_file_worker_app():
    """Create only the DB context needed by this worker; do not start Flask tasks."""
    from config import Config

    Config.setup_environment()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    app = Flask(__name__, static_folder=os.path.join(root, "static"), template_folder=os.path.join(root, "templates"))
    app.config.from_object(Config)
    db.init_app(app)
    with app.app_context():
        from app.model_gateway import models  # noqa: F401
    return app


def run_file_worker() -> None:
    from config import Config
    import redis

    require_worker_preload()
    if not Config.MERGEKIT_MODEL_GATEWAY_WORKER_TOKEN:
        raise RuntimeError("MERGEKIT_MODEL_GATEWAY_WORKER_TOKEN is required")
    if Config.MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND != "redis" or not Config.MERGEKIT_MODEL_GATEWAY_REDIS_URL:
        raise RuntimeError("Redis research queue is not configured")

    app = create_file_worker_app()
    client = redis.Redis.from_url(Config.MERGEKIT_MODEL_GATEWAY_REDIS_URL, socket_connect_timeout=3, socket_timeout=5)
    client.ping()
    ensure_file_consumer_group(client)
    consumer = f"{socket.gethostname()}:{os.getpid()}"
    interval_s = max(1, int(Config.MERGEKIT_MODEL_GATEWAY_FILE_RECONCILE_SECONDS))
    next_reconcile = 0.0
    while True:
        now = time.monotonic()
        if now >= next_reconcile:
            with app.app_context():
                reconcile_received_files(db.session, client)
            next_reconcile = now + interval_s
        deliveries = client.xreadgroup(FILE_CONSUMER_GROUP, consumer, {FILE_STREAM: ">"}, count=1, block=1000)
        for _, messages in deliveries:
            for message_id, fields in messages:
                with app.app_context():
                    consume_file_message(db.session, client, message_id, fields)


if __name__ == "__main__":
    run_file_worker()
