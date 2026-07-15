"""CPU-only, scan-first processing of private staged research files."""
from __future__ import annotations

import os

from app.model_gateway.documents import DocumentParseError, chunk_document_text, parse_document_sections
from app.model_gateway.legacy_parser import LegacyParserUnavailableError, legacy_source_format, parse_legacy_office
from app.model_gateway.models import ResearchChunk, ResearchFile
from app.model_gateway.scanner import InfectedFileError, ScannerRejectedError, ScannerUnavailableError, scan_quarantined_file


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
    if source.status != "received" or not source.quarantine_path:
        return "not_processable"

    path = source.quarantine_path
    source.status = "processing"
    source.error_code = None
    source.error_message = None
    session.add(source)
    session.commit()
    try:
        scan_quarantined_file(path)
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
