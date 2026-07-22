# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Slow HTTP DoS (Slowloris, CVE-2007-6750) protection."""

import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI

from ...core.http_protocol import RequestDeadlineMixin, create_hardened_http_protocol

REQUEST_TIMEOUT = 2.0
# Generous upper bound for CI machines under load.
CLOSE_SLACK = 4.0


@pytest.fixture
def server_port():
    app = FastAPI()

    @app.get("/")
    async def ok():
        return {"ok": True}

    @app.get("/slow")
    async def slow():
        # Response takes much longer than the request deadline.
        await asyncio.sleep(REQUEST_TIMEOUT * 2.5)
        return {"ok": True}

    protocol = create_hardened_http_protocol(REQUEST_TIMEOUT)
    assert isinstance(protocol, type)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, http=protocol, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        assert time.time() < deadline, "test server failed to start"
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield port
    server.should_exit = True
    thread.join(timeout=10)


def _connect(port: int) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.settimeout(REQUEST_TIMEOUT + CLOSE_SLACK + 5)
    return sock


def _wait_for_close(sock: socket.socket) -> float:
    """Read until the peer closes; return the elapsed time since call."""
    start = time.time()
    while True:
        data = sock.recv(4096)
        if not data:
            return time.time() - start


def _read_http_response(sock: socket.socket) -> bytes:
    """Read one non-chunked HTTP response based on Content-Length."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        assert chunk, f"connection closed before response completed: {buf!r}"
        buf += chunk
    header, _, body = buf.partition(b"\r\n\r\n")
    content_length = 0
    for line in header.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            content_length = int(line.split(b":", 1)[1])
    while len(body) < content_length:
        chunk = sock.recv(4096)
        assert chunk, "connection closed before body completed"
        body += chunk
    return header + b"\r\n\r\n" + body


def test_slow_headers_closed(server_port):
    """An incomplete request must be dropped around the deadline."""
    sock = _connect(server_port)
    try:
        sock.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n")  # no final CRLF
        elapsed = _wait_for_close(sock)
    finally:
        sock.close()
    assert REQUEST_TIMEOUT - 0.5 <= elapsed <= REQUEST_TIMEOUT + CLOSE_SLACK


def test_dripping_bytes_closed(server_port):
    """The deadline is absolute: trickling bytes must not reset it."""
    sock = _connect(server_port)
    start = time.time()
    try:
        payload = b"GET / HTTP/1.1\r\nHost: localhost\r\nX-A: "
        closed = False
        for byte in payload:
            try:
                sock.sendall(bytes([byte]))
            except OSError:
                closed = True
                break
            time.sleep(0.3)
        if not closed:
            _wait_for_close(sock)
        elapsed = time.time() - start
    finally:
        sock.close()
    assert REQUEST_TIMEOUT - 0.5 <= elapsed <= REQUEST_TIMEOUT + CLOSE_SLACK


def test_normal_request_ok(server_port):
    sock = _connect(server_port)
    try:
        sock.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        response = _read_http_response(sock)
    finally:
        sock.close()
    assert response.startswith(b"HTTP/1.1 200")


def test_slow_response_not_killed(server_port):
    """The deadline covers request receipt only, never response duration."""
    sock = _connect(server_port)
    try:
        sock.sendall(b"GET /slow HTTP/1.1\r\nHost: localhost\r\n\r\n")
        start = time.time()
        response = _read_http_response(sock)
        elapsed = time.time() - start
    finally:
        sock.close()
    assert response.startswith(b"HTTP/1.1 200")
    assert elapsed >= REQUEST_TIMEOUT * 2


def test_keep_alive_second_slow_request_closed(server_port):
    """A slow request on a reused keep-alive connection is also dropped."""
    sock = _connect(server_port)
    try:
        sock.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        response = _read_http_response(sock)
        assert response.startswith(b"HTTP/1.1 200")
        # Second request: incomplete headers.
        sock.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n")
        elapsed = _wait_for_close(sock)
    finally:
        sock.close()
    assert elapsed <= REQUEST_TIMEOUT + CLOSE_SLACK


def test_factory_disabled_returns_auto():
    assert create_hardened_http_protocol(0) == "auto"
    assert create_hardened_http_protocol(-1) == "auto"


def test_factory_returns_hardened_class():
    protocol = create_hardened_http_protocol(30)
    assert isinstance(protocol, type)
    assert issubclass(protocol, RequestDeadlineMixin)
    assert protocol.request_timeout == 30.0
