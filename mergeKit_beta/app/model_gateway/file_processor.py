"""CPU-only, scan-first processing of private staged research files."""
from __future__ import annotations

import os

from app.model_gateway.documents import DocumentParseError, chunk_document_text, parse_document_sections
from app.model_gateway.embeddings import OnnxDenseEmbedder
from app.model_gateway.legacy_parser import LegacyParserUnavailableError, legacy_source_format, parse_legacy_office
from app.model_gateway.models import ResearchChunk, ResearchFile
from app.model_gateway.quotas import QuotaExceeded, reserve_quota
from app.model_gateway.scanner import InfectedFileError, ScannerRejectedError, ScannerUnavailableError, scan_quarantined_file
from app.model_gateway.web_sources import WebFetchError, extract_html_sections, fetch_public_web_source
from app.model_gateway.vectors import encode_chunk_vectors, save_chunk_vectors


_embedding_encoder = None


def _get_embedding_encoder():
    global _embedding_encoder
    if _embedding_encoder is None:
        from config import Config

        _embedding_encoder = OnnxDenseEmbedder(Config.MERGEKIT_MODEL_GATEWAY_EMBEDDING_MODEL_PATH)
    return _embedding_encoder


def _remove_file(path: str | None) -> None:
    if path:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _reject_file(session, source: ResearchFile, error_code: str) -> str:
    _remove_file(source.quarantine_path)
    source.quarantine_path = None
    source.status = "rejected"
    source.error_code = error_code
    source.error_message = error_code
    session.add(source)
    session.commit()
    return "rejected"


def process_research_file(session, file_id: str) -> str:
    """Scan, parse and persist one staged source. Never invokes a model."""
    source = session.get(ResearchFile, file_id)
    if not source:
        return "missing"
    if source.status == "ready":
        return "already_ready"
    if source.source_kind == "url" and source.status == "queued_download":
        source.status = "received"
        session.add(source)
        session.commit()
    if source.status != "received":
        return "not_processable"

    if source.source_kind == "url" and not source.quarantine_path:
        try:
            fetched = fetch_public_web_source(source.source_url or "", _runtime_root(source))
        except WebFetchError as exc:
            source.error_code = str(exc)
            source.error_message = str(exc)
            source.status = "received" if str(exc) == "source_fetch_failed" else "rejected"
            session.add(source)
            session.commit()
            return "retry" if source.status == "received" else "rejected"
        from config import Config
        try:
            reserve_quota(
                session, source.api_key_id, "import_bytes", amount=os.path.getsize(fetched.path),
                limit=max(1, int(Config.MERGEKIT_MODEL_GATEWAY_IMPORT_BYTES_PER_DAY)), window_seconds=86400,
            )
        except QuotaExceeded as exc:
            _remove_file(fetched.path)
            source.status = "rejected"
            source.error_code = exc.code
            source.error_message = exc.code
            session.add(source)
            session.commit()
            return "rejected"
        source.source_url = fetched.canonical_url
        source.media_type = fetched.media_type
        source.content_sha256 = fetched.content_sha256
        source.quarantine_path = fetched.path
        session.add(source)
        session.commit()
    if not source.quarantine_path:
        return "not_processable"

    path = source.quarantine_path
    source.status = "processing"
    source.error_code = None
    source.error_message = None
    session.add(source)
    session.commit()
    try:
        scan_quarantined_file(path)
        if source.media_type in {"text/html", "application/xhtml+xml"}:
            with open(path, "rb") as handle:
                sections = extract_html_sections(handle.read(), source.source_url or "")
            if not sections:
                raise DocumentParseError("no_text_layer")
            chunks = chunk_document_text(sections)
        else:
            try:
                legacy_source_format(path)
            except ValueError:
                chunks = chunk_document_text(parse_document_sections(path))
            else:
                chunks = chunk_document_text(parse_legacy_office(path))
    except LegacyParserUnavailableError as exc:
        source.status = "received"
        source.error_code = str(exc)
        source.error_message = str(exc)
        session.add(source)
        session.commit()
        return "retry"
    except InfectedFileError:
        return _reject_file(session, source, "infected_file")
    except DocumentParseError as exc:
        return _reject_file(session, source, str(exc))
    except (ScannerUnavailableError, ScannerRejectedError) as exc:
        source.status = "received"
        source.error_code = str(exc)
        source.error_message = str(exc)
        session.add(source)
        session.commit()
        return "retry"
    except OSError:
        return _reject_file(session, source, "source_missing")
    try:
        vectors = encode_chunk_vectors(_get_embedding_encoder(), [chunk["text"] for chunk in chunks])
        save_chunk_vectors(_runtime_root(source), source.id, vectors)
    except (OSError, RuntimeError, ValueError):
        source.status = "received"
        source.error_code = "embedding_unavailable"
        source.error_message = source.error_code
        session.add(source)
        session.commit()
        return "retry"
    session.query(ResearchChunk).filter_by(file_id=source.id).delete()
    for chunk in chunks:
        session.add(ResearchChunk(
            file_id=source.id,
            api_key_id=source.api_key_id,
            ordinal=chunk["ordinal"],
            text=chunk["text"],
            locator=chunk["locator"],
            expires_at=source.expires_at,
        ))
    _remove_file(path)
    source.quarantine_path = None
    source.status = "ready"
    session.add(source)
    session.commit()
    return "ready"


def _runtime_root(source: ResearchFile) -> str:
    """Infer the common runtime root from the configured temporary source path."""
    del source
    from config import Config

    return Config.MERGEKIT_MODEL_GATEWAY_RESEARCH_ROOT
