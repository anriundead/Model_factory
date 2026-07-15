"""Minimal ClamAV INSTREAM client for private quarantined uploads."""
from __future__ import annotations

import socket
import struct


class ScannerUnavailableError(RuntimeError):
    pass


class InfectedFileError(ValueError):
    pass


class ScannerRejectedError(ValueError):
    pass


def _read_response(connection: socket.socket) -> bytes:
    parts = []
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            break
        parts.append(chunk)
        if b"\0" in chunk:
            break
    return b"".join(parts)


def scan_quarantined_file(path: str, *, host: str = "model-gateway-clamav", port: int = 3310, timeout_s: float = 30) -> None:
    """Raise unless clamd explicitly reports the staged file as clean."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s) as connection:
            connection.settimeout(timeout_s)
            connection.sendall(b"zINSTREAM\0")
            with open(path, "rb") as source:
                while chunk := source.read(1024 * 1024):
                    connection.sendall(struct.pack("!I", len(chunk)))
                    connection.sendall(chunk)
            connection.sendall(b"\0\0\0\0")
            response = _read_response(connection)
    except OSError as exc:
        raise ScannerUnavailableError("scanner_unavailable") from exc

    if b" FOUND" in response:
        raise InfectedFileError("infected_file")
    if b" OK" in response:
        return
    raise ScannerRejectedError("scanner_rejected")
