import ipaddress
import unittest
from unittest.mock import patch


class TestPublicSourceUrlValidation(unittest.TestCase):
    def test_rejects_loopback_and_non_http_urls(self):
        from app.model_gateway.documents import validate_public_source_url

        for url in (
            "http://127.0.0.1/private.pdf",
            "http://localhost/private.pdf",
            "file:///etc/passwd",
            "ftp://example.com/file.pdf",
        ):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, "public"):
                    validate_public_source_url(url)

    @patch("app.model_gateway.documents.socket.getaddrinfo")
    def test_rejects_private_dns_resolution(self, getaddrinfo):
        from app.model_gateway.documents import validate_public_source_url

        getaddrinfo.return_value = [
            (2, 1, 6, "", ("172.17.0.2", 443)),
        ]

        with self.assertRaisesRegex(ValueError, "public"):
            validate_public_source_url("https://example.test/report.pdf")

    @patch("app.model_gateway.documents.socket.getaddrinfo")
    def test_accepts_public_dns_resolution(self, getaddrinfo):
        from app.model_gateway.documents import validate_public_source_url

        getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ]

        parsed = validate_public_source_url("https://example.test/report.pdf")

        self.assertEqual(parsed.hostname, "example.test")


class TestResearchFileAdmission(unittest.TestCase):
    def test_rejects_unapproved_extension_and_oversize_upload(self):
        from app.model_gateway.documents import validate_upload_metadata

        with self.assertRaisesRegex(ValueError, "supported"):
            validate_upload_metadata("payload.exe", 128)
        with self.assertRaisesRegex(ValueError, "50 MiB"):
            validate_upload_metadata("report.pdf", 50 * 1024 * 1024 + 1)

    def test_accepts_supported_document_extension_within_limit(self):
        from app.model_gateway.documents import validate_upload_metadata

        self.assertEqual(validate_upload_metadata("slides.PPTX", 1024), ".pptx")


class TestResearchChunking(unittest.TestCase):
    def test_chunking_keeps_order_and_source_locator(self):
        from app.model_gateway.documents import chunk_document_text

        chunks = chunk_document_text([
            ("page", 1, "alpha beta gamma"),
            ("page", 2, "delta epsilon"),
        ], max_chars=12)

        self.assertEqual([chunk["ordinal"] for chunk in chunks], list(range(len(chunks))))
        self.assertEqual(chunks[0]["locator"], {"kind": "page", "value": 1})
        self.assertTrue(all(chunk["text"] for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
