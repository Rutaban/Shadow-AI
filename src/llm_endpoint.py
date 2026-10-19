"""A local TLS endpoint that speaks the wire format of commercial chat APIs.

The endpoint accepts the OpenAI /v1/chat/completions and Anthropic /v1/messages
request bodies and answers with a server sent event stream, one event per
generated token, so that the downlink of a session carries the same shape as a
real streaming inference response. Response text is drawn from a public
instruction corpus rather than being randomly generated.
"""
from __future__ import annotations

import json
import random
import ssl
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from config import CORPORA

_RESPONSES: list[str] = []


def _load_responses() -> list[str]:
    global _RESPONSES
    if not _RESPONSES:
        frame = pd.read_parquet(CORPORA / "alpaca.json")
        _RESPONSES = [str(v) for v in frame["output"] if isinstance(v, str) and len(v) > 40]
    return _RESPONSES


def make_self_signed(directory: Path) -> tuple[Path, Path]:
    cert = directory / "endpoint.crt"
    key = directory / "endpoint.key"
    if cert.exists() and key.exists():
        return cert, key
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(key), "-out", str(cert), "-days", "365",
         "-subj", "/CN=inference.local"],
        check=True, capture_output=True,
    )
    return cert, key


class InferenceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "inference-gateway/1.2"

    def log_message(self, fmt, *args):  # silence the default access log
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = b""
        remaining = length
        while remaining > 0:
            block = self.rfile.read(min(remaining, 65536))
            if not block:
                break
            body += block
            remaining -= len(block)
        try:
            payload = json.loads(body.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            payload = {}
        prompt_chars = len(body)
        rng = random.Random(prompt_chars ^ int(time.time() * 1e6) & 0xFFFF)
        stream = bool(payload.get("stream", True))
        corpus = _load_responses()
        answer = corpus[rng.randrange(len(corpus))]
        # A longer request does not produce a proportionally longer answer;
        # assistants summarise. The mapping below is bounded and sub linear.
        budget = int(min(900, max(24, 18 * (prompt_chars ** 0.28))))
        tokens = answer.split()
        while len(tokens) < budget:
            tokens = tokens + corpus[rng.randrange(len(corpus))].split()
        tokens = tokens[:budget]
        rate = rng.uniform(38.0, 92.0)

        if not stream:
            content = " ".join(tokens)
            data = json.dumps({
                "id": "chatcmpl-local", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": prompt_chars // 4, "completion_tokens": len(tokens)},
            }).encode()
            time.sleep(len(tokens) / rate)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        step = 1.0 / rate
        for index, token in enumerate(tokens):
            event = json.dumps({
                "id": "chatcmpl-local", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": " " + token}}],
            })
            frame = f"data: {event}\n\n".encode()
            chunk = f"{len(frame):X}\r\n".encode() + frame + b"\r\n"
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(step)
        try:
            tail = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(tail):X}\r\n".encode() + tail + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def start_endpoint(host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, int]:
    directory = Path(tempfile.gettempdir()) / "shadow_ai_tls"
    directory.mkdir(exist_ok=True)
    cert, key = make_self_signed(directory)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert), keyfile=str(key))
    ThreadingHTTPServer.request_queue_size = 512
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((host, port), InferenceHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]
