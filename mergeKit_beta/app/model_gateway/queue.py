"""Durable research-job delivery over Redis Streams."""
from __future__ import annotations


RESEARCH_STREAM = "model_gateway:research"
FILE_STREAM = "model_gateway:files"
FILE_CONSUMER_GROUP = "model-gateway-file-workers"


def enqueue_research_job(redis_client, job_id: str, service_id: str | None, task_type: str) -> str:
    message_id = redis_client.xadd(RESEARCH_STREAM, {
        "job_id": str(job_id),
        "service_id": str(service_id or ""),
        "task_type": str(task_type),
    })
    return message_id.decode() if isinstance(message_id, bytes) else str(message_id)


def enqueue_research_file(redis_client, file_id: str) -> str:
    message_id = redis_client.xadd(FILE_STREAM, {"file_id": str(file_id)})
    return message_id.decode() if isinstance(message_id, bytes) else str(message_id)
