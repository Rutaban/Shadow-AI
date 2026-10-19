"""Instrumented client that drives one session across the tapped link."""
from __future__ import annotations

import json
import socket
import ssl
import time

_CONTEXT = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
_CONTEXT.check_hostname = False
_CONTEXT.verify_mode = ssl.CERT_NONE

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _read_response(fp) -> int:
    """Consume one HTTP/1.1 response, returning the body length in bytes."""
    total = 0
    headers = {}
    line = fp.readline()
    if not line:
        return 0
    while True:
        line = fp.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        key, _, value = line.decode("latin-1").partition(":")
        headers[key.strip().lower()] = value.strip()
    if headers.get("transfer-encoding", "").lower() == "chunked":
        while True:
            size_line = fp.readline()
            if not size_line:
                break
            try:
                size = int(size_line.strip().split(b";")[0], 16)
            except ValueError:
                break
            if size == 0:
                fp.readline()
                break
            body = fp.read(size)
            total += len(body)
            fp.read(2)
    else:
        length = int(headers.get("content-length", "0"))
        remaining = length
        while remaining > 0:
            block = fp.read(min(remaining, 65536))
            if not block:
                break
            remaining -= len(block)
            total += len(block)
    return total


def run_session(tap_port: int, token: int, session: dict, host: str = "127.0.0.1",
                path: str = "/v1/chat/completions") -> dict:
    raw = socket.create_connection((host, tap_port), timeout=180)
    raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    raw.sendall(token.to_bytes(8, "big"))
    tls = _CONTEXT.wrap_socket(raw, server_hostname="inference.local")
    fp = tls.makefile("rb")
    uploaded = 0
    downloaded = 0
    request_bytes = []
    request_times = []
    start = time.perf_counter()
    try:
        for step in session["steps"]:
            if step["gap"] > 0:
                time.sleep(step["gap"])
            body = step["body"]
            head = (
                f"POST {path} HTTP/1.1\r\n"
                f"Host: inference.local\r\n"
                f"User-Agent: {_UA}\r\n"
                f"Authorization: Bearer sk-proj-REDACTED\r\n"
                f"Content-Type: application/json\r\n"
                f"Accept: text/event-stream\r\n"
                f"Content-Length: {len(body)}\r\n\r\n"
            ).encode()
            request_times.append(time.perf_counter() - start)
            request_bytes.append(len(head) + len(body))
            tls.sendall(head)
            view = memoryview(body)
            step_size = 32768
            for offset in range(0, len(body), step_size):
                tls.sendall(view[offset:offset + step_size])
            uploaded += len(head) + len(body)
            downloaded += _read_response(fp)
    finally:
        try:
            fp.close(); tls.close()
        except OSError:
            pass
    return {
        "app_uploaded_bytes": uploaded,
        "app_downloaded_bytes": downloaded,
        "requests": len(request_bytes),
        "request_bytes": request_bytes,
        "request_times": request_times,
        "wall_seconds": time.perf_counter() - start,
    }
