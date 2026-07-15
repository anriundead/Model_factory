import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class FakeSocket:
    def __init__(self, response):
        self.response = response
        self.sent = []

    def sendall(self, data):
        self.sent.append(data)

    def settimeout(self, _timeout):
        return None

    def recv(self, _size):
        response, self.response = self.response, b""
        return response

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class TestResearchScanner(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="research_scanner_")
        self.path = os.path.join(self.root, "source.pdf")
        with open(self.path, "wb") as handle:
            handle.write(b"research source")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_clean_stream_is_accepted(self):
        from app.model_gateway.scanner import scan_quarantined_file

        fake = FakeSocket(b"stream: OK\0")
        with patch("app.model_gateway.scanner.socket.create_connection", return_value=fake):
            scan_quarantined_file(self.path, host="scanner", port=3310)

        self.assertEqual(fake.sent[0], b"zINSTREAM\0")
        self.assertEqual(fake.sent[-1], b"\0\0\0\0")

    def test_infected_stream_is_rejected(self):
        from app.model_gateway.scanner import InfectedFileError, scan_quarantined_file

        with patch(
            "app.model_gateway.scanner.socket.create_connection",
            return_value=FakeSocket(b"stream: Eicar-Test-Signature FOUND\0"),
        ):
            with self.assertRaisesRegex(InfectedFileError, "infected_file"):
                scan_quarantined_file(self.path)

    def test_unavailable_scanner_is_not_treated_as_clean(self):
        from app.model_gateway.scanner import ScannerUnavailableError, scan_quarantined_file

        with patch("app.model_gateway.scanner.socket.create_connection", side_effect=OSError("connection refused")):
            with self.assertRaisesRegex(ScannerUnavailableError, "scanner_unavailable"):
                scan_quarantined_file(self.path)


if __name__ == "__main__":
    unittest.main()
