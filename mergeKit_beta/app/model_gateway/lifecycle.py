"""Database-backed lifecycle rules for short-lived research jobs.

Redis can deliver a message more than once. The database row and its lease are
therefore the authority for claiming, recovering and cancelling work.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import os

from app.model_gateway.models import ResearchJob, ServingModelService


RUNNABLE_JOB_STATES = {"queued", "retrying", "paused_model_offline"}
TERMINAL_JOB_STATES = {"completed", "failed", "dead_letter", "canceled", "expired"}


def _clear_job_paths(job: ResearchJob) -> None:
    """Remove short-lived payload/result files and forget their locations."""
    for path in (job.payload_path, job.result_path):
        if path:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
    job.payload_path = None
    job.result_path = None


def _clear_lease(job: ResearchJob) -> None:
    job.lease_owner = None
    job.lease_expires_at = None


def _service_is_running(session, job: ResearchJob) -> bool:
    if not job.model_service_id:
        return False
    service = session.get(ServingModelService, job.model_service_id)
    return bool(service and service.status == "running")


def claim_research_job(session, job_id: str, worker_id: str, *, now: datetime | None = None, lease_seconds: int = 120) -> str:
    """Atomically claim one runnable job, or return why it was skipped.

    PostgreSQL applies the row lock; SQLite test mode accepts the same API and
    remains adequate for deterministic single-process tests.
    """
    now = now or datetime.utcnow()
    job = (
        session.query(ResearchJob)
        .filter(ResearchJob.id == job_id)
        .with_for_update()
        .one_or_none()
    )
    if not job:
        return "missing"
    if job.expires_at <= now:
        _clear_job_paths(job)
        _clear_lease(job)
        job.status = "expired"
        job.finished_at = now
        session.commit()
        return "expired"
    if job.status == "cancel_requested":
        _clear_job_paths(job)
        _clear_lease(job)
        job.status = "canceled"
        job.error_code = "user_canceled"
        job.finished_at = now
        session.commit()
        return "canceled"
    if job.status not in RUNNABLE_JOB_STATES:
        return "not_runnable"
    if not _service_is_running(session, job):
        _clear_lease(job)
        job.status = "paused_model_offline"
        session.commit()
        return "paused_model_offline"

    job.status = "running"
    job.lease_owner = worker_id
    job.lease_expires_at = now + timedelta(seconds=max(1, int(lease_seconds)))
    session.commit()
    return "claimed"


def complete_research_job(session, job_id: str, worker_id: str, *, now: datetime | None = None, result_path: str | None = None) -> str:
    """Commit an idempotent terminal result, honoring a concurrent cancel."""
    now = now or datetime.utcnow()
    job = (
        session.query(ResearchJob)
        .filter(ResearchJob.id == job_id)
        .with_for_update()
        .one_or_none()
    )
    if not job:
        return "missing"
    if job.status in TERMINAL_JOB_STATES:
        return "already_terminal"
    if job.status == "cancel_requested":
        _clear_job_paths(job)
        job.status = "canceled"
        job.error_code = "user_canceled"
        outcome = "canceled"
    elif job.status == "running" and job.lease_owner == worker_id:
        job.status = "completed"
        job.result_path = result_path
        outcome = "completed"
    else:
        return "not_owner"
    _clear_lease(job)
    job.finished_at = now
    session.commit()
    return outcome


def fail_research_job(session, job_id: str, worker_id: str, error_code: str, *, now: datetime | None = None) -> str:
    """Finalize a claimed job failure without overwriting a concurrent cancel."""
    now = now or datetime.utcnow()
    job = (
        session.query(ResearchJob)
        .filter(ResearchJob.id == job_id)
        .with_for_update()
        .one_or_none()
    )
    if not job:
        return "missing"
    if job.status in TERMINAL_JOB_STATES:
        return "already_terminal"
    if job.status == "cancel_requested":
        _clear_job_paths(job)
        job.status = "canceled"
        job.error_code = "user_canceled"
        outcome = "canceled"
    elif job.status == "running" and job.lease_owner == worker_id:
        job.status = "failed"
        job.error_code = str(error_code)[:64]
        job.error_message = job.error_code
        outcome = "failed"
    else:
        return "not_owner"
    _clear_lease(job)
    job.finished_at = now
    session.commit()
    return outcome


def reconcile_research_jobs(session, *, now: datetime | None = None) -> list[str]:
    """Recover durable jobs after a worker or Redis restart.

    The returned IDs need a fresh Redis delivery. Terminal jobs are never
    returned, which keeps cancellation and completion idempotent.
    """
    now = now or datetime.utcnow()
    enqueue_ids: list[str] = []
    jobs = session.query(ResearchJob).all()
    changed = False
    for job in jobs:
        if job.expires_at <= now:
            if job.status != "expired":
                _clear_job_paths(job)
                _clear_lease(job)
                job.status = "expired"
                job.finished_at = now
                changed = True
            continue
        if job.status == "cancel_requested":
            _clear_job_paths(job)
            _clear_lease(job)
            job.status = "canceled"
            job.error_code = "user_canceled"
            job.finished_at = now
            changed = True
            continue
        if job.status == "running" and job.lease_expires_at and job.lease_expires_at <= now:
            _clear_lease(job)
            job.status = "queued" if _service_is_running(session, job) else "paused_model_offline"
            changed = True
        if job.status in RUNNABLE_JOB_STATES:
            if _service_is_running(session, job):
                if job.status == "paused_model_offline":
                    job.status = "queued"
                    changed = True
                enqueue_ids.append(job.id)
            elif job.status != "paused_model_offline":
                _clear_lease(job)
                job.status = "paused_model_offline"
                changed = True
    if changed:
        session.commit()
    return enqueue_ids
