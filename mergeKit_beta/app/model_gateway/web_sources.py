"""Extract citation-ready Stage 1 text from a fetched public HTML page."""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import hashlib
import http.client
import ipaddress
import os
import re
import socket
import ssl
from pathlib import Path
from urllib.parse import urljoin

from app.model_gateway.documents import MAX_SOURCE_BYTES, validate_public_source_url


_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript"}
_BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "figcaption"}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_WEB_MEDIA_TYPES = {
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "application/pdf": ".pdf",
}
_FETCH_CHUNK_SIZE = 1024 * 1024


class WebFetchError(ValueError):
    """A safe, machine-readable reason why a public web source was rejected."""


@dataclass(frozen=True)
class FetchedWebSource:
    canonical_url: str
    media_type: str
    path: str
    content_sha256: str


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, hostname: str, port: int, address: str, timeout: float):
        super().__init__(hostname, port=port, timeout=timeout)
        self._address = address

    def connect(self):
        self.sock = socket.create_connection((self._address, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname: str, port: int, address: str, timeout: float):
        super().__init__(hostname, port=port, timeout=timeout, context=ssl.create_default_context())
        self._address = address

    def connect(self):
        raw_socket = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(raw_socket, server_hostname=self.host)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _validated_public_url(raw_url: str):
    try:
        return validate_public_source_url(raw_url)
    except ValueError as exc:
        raise WebFetchError("source_url_not_public") from exc


def _resolve_public_address(hostname: str, port: int) -> str:
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise WebFetchError("source_url_not_public") from exc
    for _, _, _, _, sockaddr in addresses:
        address = sockaddr[0]
        if ipaddress.ip_address(address).is_global:
            return address
    raise WebFetchError("source_url_not_public")


def _open_pinned_url(parsed, timeout_seconds: float = 15):
    """Open a public URL at a freshly validated address, never via a proxy."""
    default_port = 443 if parsed.scheme == "https" else 80
    try:
        port = parsed.port or default_port
    except ValueError as exc:
        raise WebFetchError("source_url_not_public") from exc
    address = _resolve_public_address(parsed.hostname, port)
    connection_type = _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
    connection = connection_type(parsed.hostname, port, address, timeout_seconds)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    host = parsed.hostname if port == default_port else f"{parsed.hostname}:{port}"
    try:
        connection.request(
            "GET",
            path,
            headers={
                "Host": host,
                "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9",
                "Accept-Encoding": "identity",
                "User-Agent": "MergeneticResearch/1.0",
            },
        )
        return connection.getresponse()
    except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
        connection.close()
        raise WebFetchError("source_fetch_failed") from exc


def _stage_response(response, root: str, suffix: str) -> tuple[str, str]:
    content_length = response.getheader("Content-Length")
    try:
        if content_length and int(content_length) > MAX_SOURCE_BYTES:
            raise WebFetchError("source_too_large")
    except ValueError:
        raise WebFetchError("invalid_content_length") from None
    target_dir = Path(root) / "quarantine"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (hashlib.sha256(os.urandom(32)).hexdigest() + suffix)
    digest = hashlib.sha256()
    size = 0
    try:
        with target.open("xb") as handle:
            while chunk := response.read(_FETCH_CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_SOURCE_BYTES:
                    raise WebFetchError("source_too_large")
                digest.update(chunk)
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return str(target), digest.hexdigest()


class _VisibleTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.in_title = False
        self.title_parts: list[str] = []
        self.blocks: list[tuple[str, str]] = []
        self.block_stack: list[tuple[str, list[str]]] = []
        self.table_depth = 0
        self.table_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = True
        elif tag == "table":
            self.table_depth += 1
        elif tag in _BLOCK_TAGS and not self.table_depth:
            self.block_stack.append((tag, []))
        elif tag == "img":
            alt = _clean(dict(attrs).get("alt", ""))
            if alt:
                self.blocks.append(("web_image_context", alt))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = False
        elif tag == "table" and self.table_depth:
            self.table_depth -= 1
            if not self.table_depth:
                value = _clean(" ".join(self.table_parts))
                if value:
                    self.blocks.append(("web_table", value))
                self.table_parts = []
        elif tag in _BLOCK_TAGS and self.block_stack:
            opened, parts = self.block_stack.pop()
            value = _clean(" ".join(parts))
            if value:
                self.blocks.append(("web_figure" if opened == "figcaption" else "web_paragraph", value))

    def handle_data(self, data):
        if self.skip_depth:
            return
        value = _clean(data)
        if not value:
            return
        if self.in_title:
            self.title_parts.append(value)
        if self.table_depth:
            self.table_parts.append(value)
        elif self.block_stack:
            self.block_stack[-1][1].append(value)


def extract_html_sections(raw_html: bytes, canonical_url: str = "") -> list[tuple[str, int, str]]:
    """Return bounded semantic sections with stable, one-based web locators."""
    del canonical_url  # The URL belongs to the owner-scoped ResearchFile metadata.
    parser = _VisibleTextParser()
    parser.feed(raw_html.decode("utf-8", errors="replace"))
    parser.close()
    title = _clean(" ".join(parser.title_parts))
    parts: list[tuple[str, str]] = []
    if title:
        parts.append(("web_title", title))
    parts.extend(parser.blocks)
    return [(kind, index, text) for index, (kind, text) in enumerate(parts, start=1)]


def fetch_public_web_source(raw_url: str, root: str, *, max_redirects: int = 3) -> FetchedWebSource:
    """Fetch one public HTML/PDF source with redirect and response-size bounds."""
    current = (raw_url or "").strip()
    for _ in range(max(0, int(max_redirects)) + 1):
        parsed = _validated_public_url(current)
        response = _open_pinned_url(parsed)
        try:
            if response.status in _REDIRECT_STATUSES:
                location = (response.getheader("Location") or "").strip()
                if not location:
                    raise WebFetchError("redirect_missing_location")
                current = urljoin(parsed.geturl(), location)
                continue
            if response.status != 200:
                raise WebFetchError("source_fetch_failed")
            media_type = (response.getheader("Content-Type") or "").split(";", 1)[0].strip().lower()
            suffix = _WEB_MEDIA_TYPES.get(media_type)
            if not suffix:
                raise WebFetchError("unsupported_web_content_type")
            path, digest = _stage_response(response, root, suffix)
            return FetchedWebSource(parsed.geturl(), media_type, path, digest)
        finally:
            response.close()
    raise WebFetchError("too_many_redirects")
