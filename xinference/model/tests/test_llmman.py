# Copyright 2022-2023 XProbe Inc.
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
"""The `llmman serve` client: the daemon protocol behind oci:// model paths.

Exercised against a real HTTP server on a loopback port rather than mocks, so
the NDJSON streaming contract is genuinely tested.
"""

import http.server
import json
import socketserver
import threading

import pytest

from .. import llmman


def _ndjson(*objs):
    return "".join(json.dumps(o) + "\n" for o in objs)


class _FakeDaemon:
    """A minimal stand-in for `llmman serve`, on a real loopback port."""

    def __init__(self):
        self.version = {"version": "0.1.0", "pid": 1}
        self.version_status = 200
        self.pull_body = _ndjson({"status": "success"})
        self.pull_status = 200
        self.last_request = None
        daemon = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body, ctype):
                raw = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                body = (
                    daemon.version
                    if isinstance(daemon.version, str)
                    else json.dumps(daemon.version)
                )
                self._send(daemon.version_status, body, "application/json")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                daemon.last_request = json.loads(self.rfile.read(length))
                self._send(daemon.pull_status, daemon.pull_body, "application/x-ndjson")

        self._server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def daemon():
    d = _FakeDaemon()
    yield d
    d.close()


def test_accepts_a_llmman_daemon(daemon):
    llmman.check_daemon(daemon.url)


def test_rejects_a_non_llmman_server(daemon):
    daemon.version = {"hello": "world"}
    with pytest.raises(RuntimeError, match="not an llmman daemon"):
        llmman.check_daemon(daemon.url)


def test_rejects_a_server_that_does_not_serve_the_api(daemon):
    daemon.version_status = 404
    with pytest.raises(RuntimeError, match="not an llmman daemon"):
        llmman.check_daemon(daemon.url)


def test_rejects_an_unparsable_version_response(daemon):
    daemon.version = "<html>hello</html>"
    with pytest.raises(RuntimeError, match="not an llmman daemon"):
        llmman.check_daemon(daemon.url)


def test_reports_nothing_listening_actionably():
    with pytest.raises(RuntimeError, match="llmman serve"):
        llmman.check_daemon("http://127.0.0.1:1")


def test_pull_succeeds_and_forwards_progress(daemon):
    daemon.pull_body = _ndjson(
        {"status": "pulling manifest"},
        {"status": "pulling blobs", "completed": 50, "total": 100},
        {"status": "success"},
    )
    seen = []
    llmman.pull(daemon.url, "ghcr.io/org/model:tag", lambda *a: seen.append(a))

    assert daemon.last_request == {"model": "ghcr.io/org/model:tag"}
    assert seen == [("pulling manifest", 0, 0), ("pulling blobs", 50, 100)]


def test_reports_an_in_band_error_at_http_200(daemon):
    # The daemon streams errors in-band, so a 200 does not mean success.
    daemon.pull_body = _ndjson({"status": "pulling"}, {"error": "unauthorized"})
    with pytest.raises(RuntimeError, match="unauthorized"):
        llmman.pull(daemon.url, "ref")


def test_rejects_a_stream_that_ends_without_success(daemon):
    daemon.pull_body = _ndjson({"status": "pulling blobs"})
    with pytest.raises(RuntimeError, match="without reporting success"):
        llmman.pull(daemon.url, "ref")


def test_reports_a_non_ok_status(daemon):
    daemon.pull_status = 400
    daemon.pull_body = '{"error":"bad request"}'
    with pytest.raises(RuntimeError):
        llmman.pull(daemon.url, "ref")


def test_tolerates_a_non_json_diagnostic_line(daemon):
    daemon.pull_body = "not json\n" + _ndjson({"status": "success"})
    llmman.pull(daemon.url, "ref")
