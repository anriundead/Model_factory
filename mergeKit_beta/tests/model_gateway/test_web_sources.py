"""Regression tests for short-lived, citation-ready web research sources."""
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlparse

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class TestHtmlResearchExtraction(unittest.TestCase):
    def test_extracts_visible_text_table_and_image_context_with_web_locators(self):
        from app.model_gateway.web_sources import extract_html_sections

        sections = extract_html_sections(
            b"""<!doctype html><html><head><title>Corrosion study</title>
            <style>.hidden { display:none }</style><script>ignore_me()</script></head>
            <body><nav>Skip navigation</nav><main><h1>Alloy results</h1>
            <p>Sample A retained 91 percent mass after 30 days.</p>
            <table><tr><th>Sample</th><th>Mass</th></tr><tr><td>A</td><td>91%</td></tr></table>
            <figure><img alt="SEM image of corroded alloy A"><figcaption>Figure 1. Surface damage.</figcaption></figure>
            </main></body></html>""",
            "https://example.org/study",
        )

        self.assertEqual(sections[0], ("web_title", 1, "Corrosion study"))
        text = "\n".join(part[2] for part in sections)
        self.assertIn("Alloy results", text)
        self.assertIn("Sample A retained 91 percent mass after 30 days.", text)
        self.assertIn("Sample Mass A 91%", text)
        self.assertIn("SEM image of corroded alloy A", text)
        self.assertIn("Figure 1. Surface damage.", text)
        self.assertNotIn("ignore_me", text)
        self.assertNotIn("Skip navigation", text)
        self.assertTrue(all(kind.startswith("web_") and isinstance(value, int) for kind, value, _ in sections))


class _Response:
    def __init__(self, status, headers=None, body=b""):
        self.status = status
        self.headers = headers or {}
        self.body = body
        self.offset = 0

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read(self, size=-1):
        if size < 0:
            size = len(self.body) - self.offset
        chunk = self.body[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk

    def close(self):
        return None


class TestPublicWebFetch(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="web_source_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    @patch("app.model_gateway.web_sources._open_pinned_url", create=True)
    def test_follows_validated_redirect_and_stages_html(self, open_url):
        from app.model_gateway.web_sources import fetch_public_web_source

        with patch("app.model_gateway.web_sources.validate_public_source_url", create=True) as validate:
            validate.side_effect = [urlparse("https://example.org/start"), urlparse("https://papers.example.org/article")]
            open_url.side_effect = [
                _Response(302, {"Location": "https://papers.example.org/article"}),
                _Response(200, {"Content-Type": "text/html; charset=utf-8"}, b"<p>result</p>"),
            ]
            fetched = fetch_public_web_source("https://example.org/start", self.root)

        self.assertEqual(fetched.canonical_url, "https://papers.example.org/article")
        self.assertEqual(fetched.media_type, "text/html")
        self.assertTrue(fetched.path.endswith(".html"))
        with open(fetched.path, "rb") as handle:
            self.assertEqual(handle.read(), b"<p>result</p>")
        self.assertEqual(open_url.call_count, 2)

    @patch("app.model_gateway.web_sources._open_pinned_url", create=True)
    def test_rejects_private_redirect_before_opening_it(self, open_url):
        from app.model_gateway.web_sources import WebFetchError, fetch_public_web_source

        open_url.return_value = _Response(302, {"Location": "http://127.0.0.1/private"})
        with patch("app.model_gateway.web_sources.validate_public_source_url", create=True) as validate:
            validate.side_effect = [urlparse("https://example.org/start"), ValueError("source URL must resolve to a public HTTP(S) host")]
            with self.assertRaisesRegex(WebFetchError, "source_url_not_public"):
                fetch_public_web_source("https://example.org/start", self.root)

        self.assertEqual(open_url.call_count, 1)

    def test_rejects_malformed_url_port_before_opening_connection(self):
        from app.model_gateway.web_sources import WebFetchError, fetch_public_web_source

        with patch("app.model_gateway.web_sources.validate_public_source_url", create=True, return_value=urlparse("https://example.org:bad-port/study")):
            with self.assertRaisesRegex(WebFetchError, "source_url_not_public"):
                fetch_public_web_source("https://example.org:bad-port/study", self.root)


if __name__ == "__main__":
    unittest.main()
