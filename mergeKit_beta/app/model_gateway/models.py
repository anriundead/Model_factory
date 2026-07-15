"""ORM models for published model serving."""
from __future__ import annotations

from datetime import datetime
import uuid

from app.extensions import db


def _uuid() -> str:
    return str(uuid.uuid4())


class ServingModelService(db.Model):
    __bind_key__ = "model_gateway"
    __tablename__ = "serving_model_services"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    # The model factory owns its SQLite registry. This is a snapshot, not a cross-database FK.
    model_id = db.Column(db.String(36), nullable=True, index=True)
    model_path = db.Column(db.String(1024), nullable=False)
    display_name = db.Column(db.String(256), nullable=False)
    served_model_name = db.Column(db.String(128), nullable=False, unique=True, index=True)
    model_type = db.Column(db.String(32), default="text", nullable=False)
    backend_type = db.Column(db.String(32), default="vllm", nullable=False)
    status = db.Column(db.String(32), default="stopped", nullable=False, index=True)

    vllm_host = db.Column(db.String(64), default="127.0.0.1", nullable=False)
    vllm_port = db.Column(db.Integer, nullable=True, index=True)
    vllm_pid = db.Column(db.Integer, nullable=True)
    vllm_pgid = db.Column(db.Integer, nullable=True)

    gpu_ids = db.Column(db.JSON, nullable=True)
    gpu_uuids = db.Column(db.JSON, nullable=True)
    tensor_parallel_size = db.Column(db.Integer, default=1, nullable=False)
    gpu_memory_utilization = db.Column(db.Float, default=0.85, nullable=False)
    dtype = db.Column(db.String(32), default="auto", nullable=False)
    max_model_len = db.Column(db.Integer, nullable=True)
    max_num_seqs = db.Column(db.Integer, default=8, nullable=True)
    max_num_batched_tokens = db.Column(db.Integer, nullable=True)
    trust_remote_code = db.Column(db.Boolean, default=False, nullable=False)

    internal_api_key = db.Column(db.String(128), nullable=True)
    internal_api_key_hash = db.Column(db.String(64), nullable=True)
    last_error = db.Column(db.Text, nullable=True)
    last_exit_reason = db.Column(db.String(256), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    started_at = db.Column(db.DateTime, nullable=True)
    stopped_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "display_name": self.display_name,
            "served_model_name": self.served_model_name,
            "model_type": self.model_type,
            "backend_type": self.backend_type,
            "status": self.status,
            "vllm_host": self.vllm_host,
            "vllm_port": self.vllm_port,
            "gpu_ids": self.gpu_ids or [],
            "gpu_uuids": self.gpu_uuids or [],
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "trust_remote_code": self.trust_remote_code,
            "last_error": self.last_error,
            "last_exit_reason": self.last_exit_reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
        }


class ServingApiKey(db.Model):
    __bind_key__ = "model_gateway"
    __tablename__ = "serving_api_keys"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    key_hash = db.Column(db.String(64), unique=True, nullable=False, index=True)
    prefix = db.Column(db.String(32), default="mk_live", nullable=False)
    last4 = db.Column(db.String(4), nullable=False)
    owner_label = db.Column(db.String(128), nullable=False)
    status = db.Column(db.String(32), default="active", nullable=False, index=True)
    model_allowlist = db.Column(db.JSON, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    last_used_at = db.Column(db.DateTime, nullable=True)
    expires_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "prefix": self.prefix,
            "last4": self.last4,
            "owner_label": self.owner_label,
            "status": self.status,
            "model_allowlist": self.model_allowlist or [],
            "notes": self.notes,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ServingRequest(db.Model):
    __bind_key__ = "model_gateway"
    __tablename__ = "serving_requests"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    api_key_id = db.Column(db.String(36), db.ForeignKey("serving_api_keys.id", ondelete="SET NULL"), nullable=True, index=True)
    model_service_id = db.Column(db.String(36), db.ForeignKey("serving_model_services.id", ondelete="SET NULL"), nullable=True, index=True)
    served_model_name = db.Column(db.String(128), nullable=False, index=True)
    request_type = db.Column(db.String(32), default="chat", nullable=False)
    status = db.Column(db.String(32), default="pending", nullable=False, index=True)
    stream = db.Column(db.Boolean, default=False, nullable=False)
    idempotency_key = db.Column(db.String(128), nullable=True, index=True)
    error_code = db.Column(db.String(64), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    finished_at = db.Column(db.DateTime, nullable=True)


class ServingUsageRecord(db.Model):
    __bind_key__ = "model_gateway"
    __tablename__ = "serving_usage_records"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    request_id = db.Column(db.String(36), db.ForeignKey("serving_requests.id", ondelete="SET NULL"), nullable=True, index=True)
    api_key_id = db.Column(db.String(36), db.ForeignKey("serving_api_keys.id", ondelete="SET NULL"), nullable=True, index=True)
    model_service_id = db.Column(db.String(36), db.ForeignKey("serving_model_services.id", ondelete="SET NULL"), nullable=True, index=True)
    served_model_name = db.Column(db.String(128), nullable=False, index=True)
    prompt_tokens = db.Column(db.Integer, default=0, nullable=False)
    completion_tokens = db.Column(db.Integer, default=0, nullable=False)
    total_tokens = db.Column(db.Integer, default=0, nullable=False)
    usage_source = db.Column(db.String(64), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class ServingEvent(db.Model):
    __bind_key__ = "model_gateway"
    __tablename__ = "serving_events"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    model_service_id = db.Column(db.String(36), db.ForeignKey("serving_model_services.id", ondelete="SET NULL"), nullable=True, index=True)
    event_type = db.Column(db.String(64), nullable=False, index=True)
    message = db.Column(db.Text, nullable=True)
    payload = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class ResearchFile(db.Model):
    """Short-lived, API-key-owned research source metadata."""

    __bind_key__ = "model_gateway"
    __tablename__ = "research_files"

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    api_key_id = db.Column(db.String(36), db.ForeignKey("serving_api_keys.id", ondelete="CASCADE"), nullable=False, index=True)
    original_name = db.Column(db.String(512), nullable=False)
    source_kind = db.Column(db.String(16), nullable=False)  # upload | url
    source_url = db.Column(db.Text, nullable=True)
    media_type = db.Column(db.String(128), nullable=True)
    status = db.Column(db.String(32), default="received", nullable=False, index=True)
    content_sha256 = db.Column(db.String(64), nullable=True, index=True)
    quarantine_path = db.Column(db.String(1024), nullable=True)
    parsed_text_path = db.Column(db.String(1024), nullable=True)
    source_locator = db.Column(db.JSON, nullable=True)
    error_code = db.Column(db.String(64), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class ResearchJob(db.Model):
    """Durable job state. Payload and result files are always TTL-bound."""

    __bind_key__ = "model_gateway"
    __tablename__ = "research_jobs"
    __table_args__ = (
        db.UniqueConstraint(
            "api_key_id",
            "served_model_name",
            "idempotency_key",
            name="uq_research_job_api_model_idempotency",
        ),
    )

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    api_key_id = db.Column(db.String(36), db.ForeignKey("serving_api_keys.id", ondelete="CASCADE"), nullable=False, index=True)
    model_service_id = db.Column(db.String(36), db.ForeignKey("serving_model_services.id", ondelete="SET NULL"), nullable=True, index=True)
    served_model_name = db.Column(db.String(128), nullable=False, index=True)
    task_type = db.Column(db.String(32), nullable=False)
    status = db.Column(db.String(32), default="received", nullable=False, index=True)
    file_ids = db.Column(db.JSON, nullable=False, default=list)
    output_format = db.Column(db.String(16), default="markdown", nullable=False)
    require_citations = db.Column(db.Boolean, default=True, nullable=False)
    idempotency_key = db.Column(db.String(128), nullable=True, index=True)
    request_fingerprint = db.Column(db.String(64), nullable=True)
    payload_path = db.Column(db.String(1024), nullable=True)
    result_path = db.Column(db.String(1024), nullable=True)
    error_code = db.Column(db.String(64), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    lease_owner = db.Column(db.String(128), nullable=True)
    lease_expires_at = db.Column(db.DateTime, nullable=True, index=True)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    finished_at = db.Column(db.DateTime, nullable=True)


class ResearchChunk(db.Model):
    """TTL-bound parsed text with a precise source locator."""

    __bind_key__ = "model_gateway"
    __tablename__ = "research_chunks"

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    file_id = db.Column(db.String(36), db.ForeignKey("research_files.id", ondelete="CASCADE"), nullable=False, index=True)
    api_key_id = db.Column(db.String(36), db.ForeignKey("serving_api_keys.id", ondelete="CASCADE"), nullable=False, index=True)
    ordinal = db.Column(db.Integer, nullable=False)
    text = db.Column(db.Text, nullable=False)
    locator = db.Column(db.JSON, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
