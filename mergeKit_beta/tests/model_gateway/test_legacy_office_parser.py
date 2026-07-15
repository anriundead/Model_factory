import os
import shutil
import tempfile
import unittest

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class FakeResponse:
    def __init__(self, status_code=200, headers=None, payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload

    def json(self):
        return self._payload


class TestLegacyOfficeParser(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="legacy_parser_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write_source(self, suffix, payload=b"legacy source"):
        path = os.path.join(self.root, f"source{suffix}")
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def test_legacy_source_format_keeps_doc_paragraph_and_ppt_slide_locators(self):
        from app.model_gateway.legacy_parser import legacy_source_format

        self.assertEqual(legacy_source_format("report.doc"), ("doc", "paragraph"))
        self.assertEqual(legacy_source_format("slides.PPT"), ("ppt", "slide"))
        with self.assertRaisesRegex(ValueError, "unsupported_legacy_format"):
            legacy_source_format("report.docx")

    def test_parser_sends_worker_token_and_returns_validated_doc_sections(self):
        from app.model_gateway.legacy_parser import parse_legacy_office

        source = self.write_source(".doc")
        captured = {}

        def post(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return FakeResponse(payload={"sections": [{"kind": "paragraph", "value": 2, "text": "verified finding"}]})

        sections = parse_legacy_office(
            source,
            url="http://parser:8090",
            token="worker-token",
            timeout_seconds=45,
            max_bytes=1024,
            post=post,
        )

        self.assertEqual(sections, [("paragraph", 2, "verified finding")])
        self.assertEqual(captured["url"], "http://parser:8090/parse")
        self.assertEqual(captured["headers"]["X-Worker-Token"], "worker-token")
        self.assertEqual(captured["headers"]["X-Source-Format"], "doc")

    def test_parser_rejects_locator_type_that_cannot_be_cited_for_source_format(self):
        from app.model_gateway.documents import DocumentParseError
        from app.model_gateway.legacy_parser import parse_legacy_office

        source = self.write_source(".doc")

        def post(*args, **kwargs):
            return FakeResponse(payload={"sections": [{"kind": "slide", "value": 1, "text": "wrong locator"}]})

        with self.assertRaisesRegex(DocumentParseError, "legacy_parse_invalid_output"):
            parse_legacy_office(source, url="http://parser", token="token", timeout_seconds=45, max_bytes=1024, post=post)

    def test_parser_maps_network_failure_to_retryable_unavailable(self):
        from app.model_gateway.legacy_parser import LegacyParserUnavailableError, parse_legacy_office

        source = self.write_source(".ppt")

        def post(*args, **kwargs):
            raise OSError("offline")

        with self.assertRaisesRegex(LegacyParserUnavailableError, "legacy_parser_unavailable"):
            parse_legacy_office(source, url="http://parser", token="token", timeout_seconds=45, max_bytes=1024, post=post)


if __name__ == "__main__":
    unittest.main()
