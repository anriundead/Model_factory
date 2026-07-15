"""Admission checks shared by uploaded and remotely fetched research sources."""
from __future__ import annotations

import ipaddress
import os
import socket
import hashlib
import uuid
from pathlib import Path
from urllib.parse import ParseResult, urlparse


MAX_SOURCE_BYTES = 50 * 1024 * 1024
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx"}


class DocumentParseError(ValueError):
    """A safe, machine-readable reason why a source cannot be parsed."""



def _is_public_host(hostname: str) -> bool:
    if not hostname or hostname.lower() in {"localhost", "localhost.localdomain"}:
        return False
    try:
        addresses = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("source URL host could not be resolved") from exc
    for _, _, _, _, sockaddr in addresses:
        if not ipaddress.ip_address(sockaddr[0]).is_global:
            return False
    return bool(addresses)


def validate_public_source_url(raw_url: str) -> ParseResult:
    parsed = urlparse((raw_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not _is_public_host(parsed.hostname):
        raise ValueError("source URL must resolve to a public HTTP(S) host")
    return parsed


def validate_upload_metadata(filename: str, content_length: int) -> str:
    suffix = os.path.splitext((filename or "").strip())[1].lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise ValueError("file type is not supported")
    if int(content_length) < 0 or int(content_length) > MAX_SOURCE_BYTES:
        raise ValueError("file exceeds the 50 MiB upload limit")
    return suffix


def save_upload(file_storage, root: str) -> tuple[str, str]:
    """Stream an admitted upload into a private quarantine directory."""
    suffix = validate_upload_metadata(file_storage.filename, 0)
    target_dir = Path(root) / "quarantine"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid.uuid4()}{suffix}"
    digest = hashlib.sha256()
    size = 0
    try:
        with target.open("xb") as out:
            while chunk := file_storage.stream.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_SOURCE_BYTES:
                    raise ValueError("file exceeds the 50 MiB upload limit")
                digest.update(chunk)
                out.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return str(target), digest.hexdigest()


def chunk_document_text(sections: list[tuple[str, int, str]], max_chars: int = 2400) -> list[dict]:
    """Create bounded, citation-safe chunks without joining source locators."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks: list[dict] = []
    for kind, value, raw_text in sections:
        text = (raw_text or "").strip()
        for start in range(0, len(text), max_chars):
            part = text[start:start + max_chars].strip()
            if part:
                chunks.append({
                    "ordinal": len(chunks),
                    "text": part,
                    "locator": {"kind": str(kind), "value": int(value)},
                })
    return chunks


def _require_text(sections: list[tuple[str, int, str]]) -> list[tuple[str, int, str]]:
    cleaned = [(kind, value, text.strip()) for kind, value, text in sections if (text or "").strip()]
    if not cleaned:
        raise DocumentParseError("no_text_layer")
    return cleaned


def _parse_pdf(path: str) -> list[tuple[str, int, str]]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(path)
    except Exception as exc:
        raise DocumentParseError("invalid_pdf") from exc
    if reader.is_encrypted:
        raise DocumentParseError("password_protected")
    try:
        return _require_text([("page", index, page.extract_text() or "") for index, page in enumerate(reader.pages, start=1)])
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError("invalid_pdf") from exc


def _parse_docx(path: str) -> list[tuple[str, int, str]]:
    from docx import Document

    try:
        document = Document(path)
        return _require_text([("paragraph", index, paragraph.text) for index, paragraph in enumerate(document.paragraphs, start=1)])
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError("invalid_docx") from exc


def _parse_pptx(path: str) -> list[tuple[str, int, str]]:
    from pptx import Presentation

    try:
        presentation = Presentation(path)
        sections = []
        for index, slide in enumerate(presentation.slides, start=1):
            text = "\n".join(shape.text for shape in slide.shapes if getattr(shape, "has_text_frame", False))
            sections.append(("slide", index, text))
        return _require_text(sections)
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError("invalid_pptx") from exc


def parse_document_sections(path: str) -> list[tuple[str, int, str]]:
    """Extract text with stable citation locators from a single staged file."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix == ".pptx":
        return _parse_pptx(path)
    if suffix in {".doc", ".ppt"}:
        raise DocumentParseError("legacy_parser_required")
    raise DocumentParseError("unsupported_file_type")
