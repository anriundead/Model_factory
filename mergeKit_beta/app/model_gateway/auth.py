"""Authentication helpers for model serving."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime


API_KEY_PREFIX = "mk_live"


def hash_secret(secret: str) -> str:
    value = (secret or "").strip()
    if not value:
        raise ValueError("secret is required")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str, str, str]:
    plaintext = f"{API_KEY_PREFIX}_{secrets.token_urlsafe(32)}"
    return plaintext, hash_secret(plaintext), API_KEY_PREFIX, plaintext[-4:]


def extract_bearer_token(header_value: str | None) -> str | None:
    value = (header_value or "").strip()
    parts = value.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        return None
    return parts[1].strip()


def require_admin_token(config, authorization_header: str | None) -> None:
    if hasattr(config, "get"):
        expected = (config.get("MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN", "") or "").strip()
    else:
        expected = (getattr(config, "MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN", "") or "").strip()
    if not expected:
        raise RuntimeError("serving admin token is not configured")
    supplied = extract_bearer_token(authorization_header)
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise PermissionError("invalid serving admin token")


def find_active_api_key(raw_key: str):
    from app.extensions import db
    from app.model_gateway.models import ServingApiKey

    digest = hash_secret(raw_key)
    key = db.session.query(ServingApiKey).filter_by(key_hash=digest, status="active").first()
    if not key:
        return None
    if key.expires_at and key.expires_at <= datetime.utcnow():
        return None
    key.last_used_at = datetime.utcnow()
    db.session.add(key)
    return key
