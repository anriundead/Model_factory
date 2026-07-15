"""Database-backed, fixed-window quota accounting for Gateway API Keys."""
from __future__ import annotations

from datetime import datetime, timedelta
import math

from sqlalchemy.exc import IntegrityError

from app.model_gateway.models import GatewayQuotaBucket


_CODES = {
    "chat_requests": "chat_rate_limit_exceeded",
    "research_submissions": "research_submission_limit_exceeded",
    "active_research_jobs": "research_concurrency_limit_exceeded",
    "import_bytes": "daily_import_quota_exceeded",
}


class QuotaExceeded(RuntimeError):
    def __init__(self, scope: str, limit: int, retry_after_seconds: int):
        self.scope = scope
        self.limit = int(limit)
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        self.code = _CODES.get(scope, "quota_exceeded")
        super().__init__(self.code)


def _window_start(now: datetime, window_seconds: int) -> datetime:
    if int(window_seconds) <= 0:
        return datetime(1970, 1, 1)
    seconds = int(window_seconds)
    epoch = int(now.timestamp())
    return datetime.utcfromtimestamp(epoch - (epoch % seconds))


def _get_bucket(session, api_key_id: str, scope: str, window_start: datetime) -> GatewayQuotaBucket:
    query = session.query(GatewayQuotaBucket).filter_by(
        api_key_id=api_key_id, scope=scope, window_start=window_start
    )
    bucket = query.with_for_update().one_or_none()
    if bucket:
        return bucket
    try:
        with session.begin_nested():
            bucket = GatewayQuotaBucket(api_key_id=api_key_id, scope=scope, window_start=window_start, used_units=0)
            session.add(bucket)
            session.flush()
    except IntegrityError:
        bucket = query.with_for_update().one()
    return bucket


def reserve_quota(
    session,
    api_key_id: str,
    scope: str,
    *,
    amount: int,
    limit: int,
    window_seconds: int,
    now: datetime | None = None,
) -> GatewayQuotaBucket:
    """Reserve usage in the caller's transaction or raise before accepting work."""
    amount, limit, window_seconds = int(amount), int(limit), int(window_seconds)
    if amount < 1 or limit < 1:
        raise ValueError("quota amount and limit must be positive")
    now = now or datetime.utcnow()
    start = _window_start(now, window_seconds)
    bucket = _get_bucket(session, api_key_id, scope, start)
    if bucket.used_units + amount > limit:
        retry_after = 60 if window_seconds <= 0 else math.ceil((start + timedelta(seconds=window_seconds) - now).total_seconds())
        raise QuotaExceeded(scope, limit, retry_after)
    bucket.used_units += amount
    session.add(bucket)
    return bucket


def release_quota(
    session,
    api_key_id: str,
    scope: str,
    *,
    amount: int,
    window_seconds: int,
    now: datetime | None = None,
) -> None:
    """Release a previously reserved concurrent slot without going negative."""
    amount = int(amount)
    if amount < 1:
        raise ValueError("quota amount must be positive")
    now = now or datetime.utcnow()
    bucket = _get_bucket(session, api_key_id, scope, _window_start(now, window_seconds))
    bucket.used_units = max(0, bucket.used_units - amount)
    session.add(bucket)
