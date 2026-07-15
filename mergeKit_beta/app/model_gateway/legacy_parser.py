"""Private Worker client for the isolated legacy Office text parser."""
from __future__ import annotations

from pathlib import Path

import requests

from app.model_gateway.documents import DocumentParseError


class LegacyParserUnavailableError(RuntimeError):
    """The private parser cannot be reached and the source can be retried."""


def legacy_source_format(path: str) -> tuple[str, str]:
    suffix = Path(path).suffix.lower()
    if suffix == ".doc":
        return "doc", "paragraph"
    if suffix == ".ppt":
        return "ppt", "slide"
    raise ValueError("unsupported_legacy_format")


def _settings(url, token, timeout_seconds, max_bytes):
    if url is None or token is None or timeout_seconds is None or max_bytes is None:
        from config import Config

        url = Config.MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_URL if url is None else url
        token = Config.MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_TOKEN if token is None else token
        timeout_seconds = Config.MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        max_bytes = Config.MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_MAX_MIB * 1024 * 1024 if max_bytes is None else max_bytes
    if not str(url or "").strip() or not str(token or "").strip():
        raise LegacyParserUnavailableError("legacy_parser_unavailable")
    return str(url).rstrip("/"), str(token), max(1, int(timeout_seconds)), max(1, int(max_bytes))


def _validated_sections(payload, expected_kind: str, max_bytes: int) -> list[tuple[str, int, str]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("sections"), list):
        raise DocumentParseError("legacy_parse_invalid_output")
    sections = []
    total = 0
    for item in payload["sections"]:
        if not isinstance(item, dict) or item.get("kind") != expected_kind:
            raise DocumentParseError("legacy_parse_invalid_output")
        value, text = item.get("value"), item.get("text")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1 or not isinstance(text, str) or not text.strip():
            raise DocumentParseError("legacy_parse_invalid_output")
        total += len(text.encode("utf-8"))
        if total > max_bytes:
            raise DocumentParseError("legacy_parse_output_too_large")
        sections.append((expected_kind, value, text.strip()))
    if not sections:
        raise DocumentParseError("no_text_layer")
    return sections


def parse_legacy_office(path: str, *, url=None, token=None, timeout_seconds=None, max_bytes=None, post=requests.post) -> list[tuple[str, int, str]]:
    """Return citation-safe text sections for one clean legacy DOC or PPT."""
    source_format, expected_kind = legacy_source_format(path)
    url, token, timeout_seconds, max_bytes = _settings(url, token, timeout_seconds, max_bytes)
    try:
        with open(path, "rb") as source:
            response = post(
                f"{url}/parse",
                data=source,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Worker-Token": token,
                    "X-Source-Format": source_format,
                },
                timeout=(3, timeout_seconds),
            )
    except (OSError, requests.RequestException) as exc:
        raise LegacyParserUnavailableError("legacy_parser_unavailable") from exc
    if response.status_code == 503:
        raise LegacyParserUnavailableError("legacy_parser_unavailable")
    if response.status_code != 200:
        raise DocumentParseError(response.headers.get("X-Parser-Error", "legacy_parse_failed"))
    try:
        return _validated_sections(response.json(), expected_kind, max_bytes)
    except (TypeError, ValueError) as exc:
        raise DocumentParseError("legacy_parse_invalid_output") from exc
