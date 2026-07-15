import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class ResearchFileProcessorTestCase(unittest.TestCase):
    def setUp(self):
        from flask import Flask
        from app.extensions import db
        from app.model_gateway import models as gateway_models  # noqa: F401

        self.root = tempfile.mkdtemp(prefix="research_file_processor_")
        db_path = os.path.join(self.root, "gateway.sqlite")
        self.app = Flask(__name__)
        self.app.config.update(
            SQLALCHEMY_DATABASE_URI=f"sqlite:///{db_path}",
            SQLALCHEMY_BINDS={"model_gateway": f"sqlite:///{db_path}"},
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
        )
        self.db = db
        db.init_app(self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()

    def tearDown(self):
        self.db.session.remove()
        self.db.drop_all()
        self.ctx.pop()
        shutil.rmtree(self.root, ignore_errors=True)

    def make_file(self):
        from docx import Document
        from app.model_gateway.models import ResearchFile, ServingApiKey

        path = os.path.join(self.root, "report.docx")
        document = Document()
        document.add_paragraph("verified finding")
        document.save(path)
        key = ServingApiKey(
            id="key-1", key_hash="digest", prefix="mk_live", last4="1234", owner_label="test"
        )
        source = ResearchFile(
            id="file-1",
            api_key_id=key.id,
            original_name="report.docx",
            source_kind="upload",
            status="received",
            quarantine_path=path,
            expires_at=datetime.utcnow() + timedelta(hours=24),
        )
        self.db.session.add_all([key, source])
        self.db.session.commit()
        return source, path

    def make_legacy_file(self, suffix):
        from app.model_gateway.models import ResearchFile, ServingApiKey

        path = os.path.join(self.root, "legacy" + suffix)
        with open(path, "wb") as handle:
            handle.write(b"legacy payload")
        key = ServingApiKey(
            id="legacy-key", key_hash="legacy-digest", prefix="mk_live", last4="5678", owner_label="legacy"
        )
        source = ResearchFile(
            id="legacy-file", api_key_id=key.id, original_name="legacy" + suffix,
            source_kind="upload", status="received", quarantine_path=path,
            expires_at=datetime.utcnow() + timedelta(hours=24),
        )
        self.db.session.add_all([key, source])
        self.db.session.commit()
        return source, path


class TestResearchFileProcessor(ResearchFileProcessorTestCase):
    @patch("app.model_gateway.file_processor._runtime_root")
    @patch("app.model_gateway.file_processor.save_chunk_vectors", create=True)
    @patch("app.model_gateway.file_processor.encode_chunk_vectors", create=True)
    @patch("app.model_gateway.file_processor._get_embedding_encoder", create=True)
    @patch("app.model_gateway.file_processor.scan_quarantined_file")
    def test_ready_source_persists_batched_vectors_in_research_runtime(self, scan, get_encoder, encode, save, runtime_root):
        from app.model_gateway.file_processor import process_research_file

        runtime_root.return_value = self.root
        encode.return_value = np.asarray([[1.0, 0.0]], dtype=np.float32)
        _, path = self.make_file()

        self.assertEqual(process_research_file(self.db.session, "file-1"), "ready")

        get_encoder.assert_called_once()
        encode.assert_called_once_with(get_encoder.return_value, ["verified finding"])
        save.assert_called_once_with(self.root, "file-1", encode.return_value)
        scan.assert_called_once_with(path)

    @patch("app.model_gateway.file_processor.scan_quarantined_file")
    @patch("app.model_gateway.file_processor.reserve_quota", create=True)
    @patch("app.model_gateway.file_processor.fetch_public_web_source")
    def test_web_source_over_daily_quota_is_removed_before_scan(self, fetch, reserve, scan):
        from app.model_gateway.models import ResearchFile, ServingApiKey
        from app.model_gateway.quotas import QuotaExceeded
        from app.model_gateway.web_sources import FetchedWebSource
        from app.model_gateway.file_processor import process_research_file

        path = os.path.join(self.root, "oversize.html")
        with open(path, "wb") as handle:
            handle.write(b"<p>quota limited source</p>")
        key = ServingApiKey(id="quota-key", key_hash="quota-digest", prefix="mk_live", last4="0000", owner_label="quota")
        source = ResearchFile(
            id="quota-file", api_key_id=key.id, original_name="article", source_kind="url",
            source_url="https://example.org/article", status="received",
            expires_at=datetime.utcnow() + timedelta(hours=24),
        )
        self.db.session.add_all([key, source])
        self.db.session.commit()
        fetch.return_value = FetchedWebSource(
            canonical_url=source.source_url, media_type="text/html", path=path, content_sha256="b" * 64,
        )
        reserve.side_effect = QuotaExceeded("import_bytes", 1, 60)

        self.assertEqual(process_research_file(self.db.session, source.id), "rejected")
        row = self.db.session.get(ResearchFile, source.id)

        self.assertEqual(row.status, "rejected")
        self.assertEqual(row.error_code, "daily_import_quota_exceeded")
        self.assertFalse(os.path.exists(path))
        scan.assert_not_called()

    @patch("app.model_gateway.file_processor.scan_quarantined_file")
    @patch("app.model_gateway.file_processor.fetch_public_web_source", create=True)
    def test_web_source_is_fetched_scanned_chunked_and_original_removed(self, fetch, scan):
        from app.model_gateway.models import ResearchChunk, ResearchFile, ServingApiKey
        from app.model_gateway.web_sources import FetchedWebSource
        from app.model_gateway.file_processor import process_research_file

        path = os.path.join(self.root, "article.html")
        with open(path, "wb") as handle:
            handle.write(b"<title>Alloy study</title><p>Mass remained stable.</p><img alt='surface image'>")
        key = ServingApiKey(id="web-key", key_hash="web-digest", prefix="mk_live", last4="9999", owner_label="web")
        source = ResearchFile(
            id="web-file", api_key_id=key.id, original_name="article", source_kind="url",
            source_url="https://example.org/article", status="received",
            expires_at=datetime.utcnow() + timedelta(hours=24),
        )
        self.db.session.add_all([key, source])
        self.db.session.commit()
        fetch.return_value = FetchedWebSource(
            canonical_url="https://papers.example.org/article",
            media_type="text/html", path=path, content_sha256="a" * 64,
        )

        self.assertEqual(process_research_file(self.db.session, source.id), "ready")
        row = self.db.session.get(ResearchFile, source.id)
        chunks = self.db.session.query(ResearchChunk).filter_by(file_id=source.id).all()

        self.assertEqual(row.status, "ready")
        self.assertEqual(row.source_url, "https://papers.example.org/article")
        self.assertEqual(row.media_type, "text/html")
        self.assertIsNone(row.quarantine_path)
        self.assertFalse(os.path.exists(path))
        self.assertTrue(any(chunk.locator["kind"] == "web_title" for chunk in chunks))
        self.assertTrue(any("surface image" in chunk.text for chunk in chunks))
        scan.assert_called_once_with(path)

    @patch("app.model_gateway.file_processor.scan_quarantined_file")
    def test_clean_file_becomes_ready_with_ttl_chunks_and_no_original(self, scan):
        from app.model_gateway.file_processor import process_research_file
        from app.model_gateway.models import ResearchChunk, ResearchFile

        _, path = self.make_file()
        outcome = process_research_file(self.db.session, "file-1")
        row = self.db.session.get(ResearchFile, "file-1")
        chunk = self.db.session.query(ResearchChunk).one()

        self.assertEqual(outcome, "ready")
        self.assertEqual(row.status, "ready")
        self.assertIsNone(row.quarantine_path)
        self.assertFalse(os.path.exists(path))
        self.assertEqual(chunk.text, "verified finding")
        self.assertEqual(chunk.locator, {"kind": "paragraph", "value": 1})
        self.assertEqual(chunk.expires_at, row.expires_at)
        scan.assert_called_once_with(path)

    @patch("app.model_gateway.file_processor.scan_quarantined_file")
    def test_infected_file_is_rejected_and_removed(self, scan):
        from app.model_gateway.file_processor import process_research_file
        from app.model_gateway.models import ResearchChunk, ResearchFile
        from app.model_gateway.scanner import InfectedFileError

        _, path = self.make_file()
        scan.side_effect = InfectedFileError("infected_file")

        outcome = process_research_file(self.db.session, "file-1")
        row = self.db.session.get(ResearchFile, "file-1")

        self.assertEqual(outcome, "rejected")
        self.assertEqual(row.status, "rejected")
        self.assertEqual(row.error_code, "infected_file")
        self.assertIsNone(row.quarantine_path)
        self.assertFalse(os.path.exists(path))
        self.assertEqual(self.db.session.query(ResearchChunk).count(), 0)

    @patch("app.model_gateway.file_processor.parse_legacy_office", return_value=[("paragraph", 2, "legacy finding")])
    @patch("app.model_gateway.file_processor.scan_quarantined_file")
    def test_legacy_doc_is_scanned_once_parsed_and_persisted(self, scan, parse):
        from app.model_gateway.file_processor import process_research_file
        from app.model_gateway.models import ResearchChunk, ResearchFile

        _, original = self.make_legacy_file(".doc")

        self.assertEqual(process_research_file(self.db.session, "legacy-file"), "ready")
        row = self.db.session.get(ResearchFile, "legacy-file")
        chunk = self.db.session.query(ResearchChunk).one()

        self.assertEqual(row.status, "ready")
        self.assertEqual(chunk.locator, {"kind": "paragraph", "value": 2})
        scan.assert_called_once_with(original)
        parse.assert_called_once_with(original)
        self.assertFalse(os.path.exists(original))

    @patch("app.model_gateway.file_processor.parse_legacy_office")
    @patch("app.model_gateway.file_processor.scan_quarantined_file")
    def test_legacy_parser_unavailable_retries_without_removing_original(self, scan, parse):
        from app.model_gateway.file_processor import process_research_file
        from app.model_gateway.legacy_parser import LegacyParserUnavailableError
        from app.model_gateway.models import ResearchFile

        _, original = self.make_legacy_file(".ppt")
        parse.side_effect = LegacyParserUnavailableError("legacy_parser_unavailable")

        self.assertEqual(process_research_file(self.db.session, "legacy-file"), "retry")
        row = self.db.session.get(ResearchFile, "legacy-file")

        self.assertEqual(row.status, "received")
        self.assertEqual(row.error_code, "legacy_parser_unavailable")
        self.assertEqual(row.quarantine_path, original)
        self.assertTrue(os.path.exists(original))


if __name__ == "__main__":
    unittest.main()
