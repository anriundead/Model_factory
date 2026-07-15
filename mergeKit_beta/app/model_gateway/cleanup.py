"""Physical expiry cleanup for short-lived research source content."""
from __future__ import annotations

from datetime import datetime
import os

from app.model_gateway.models import ResearchChunk, ResearchFile, ResearchJob
from app.model_gateway.quotas import release_quota
from app.model_gateway.vectors import vector_path


_ACTIVE_STATES = {"received", "queued", "retrying", "paused_model_offline", "running", "streaming", "cancel_requested"}


def _unlink_managed(path: str | None, root: str) -> None:
    if not path:
        return
    try:
        root_real = os.path.realpath(root)
        path_real = os.path.realpath(path)
        if os.path.commonpath((root_real, path_real)) != root_real:
            return
    except (OSError, ValueError):
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except IsADirectoryError:
        pass


def purge_expired_research_data(session, root: str, *, now: datetime | None = None) -> dict[str, int]:
    """Delete expired source content and metadata without touching external paths."""
    now = now or datetime.utcnow()
    removed = {"files": 0, "jobs": 0, "chunks": 0}

    for job in session.query(ResearchJob).filter(ResearchJob.expires_at <= now).all():
        _unlink_managed(job.payload_path, root)
        _unlink_managed(job.result_path, root)
        if job.status in _ACTIVE_STATES:
            release_quota(session, job.api_key_id, "active_research_jobs", amount=1, window_seconds=0, now=now)
        session.delete(job)
        removed["jobs"] += 1

    for source in session.query(ResearchFile).filter(ResearchFile.expires_at <= now).all():
        _unlink_managed(source.quarantine_path, root)
        _unlink_managed(source.parsed_text_path, root)
        _unlink_managed(vector_path(root, source.id), root)
        removed["chunks"] += session.query(ResearchChunk).filter_by(file_id=source.id).delete(synchronize_session=False)
        session.delete(source)
        removed["files"] += 1

    orphaned = session.query(ResearchChunk).filter(ResearchChunk.expires_at <= now).delete(synchronize_session=False)
    removed["chunks"] += orphaned
    session.commit()
    return removed
