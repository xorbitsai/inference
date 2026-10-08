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
import subprocess
import threading
from unittest import mock

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
        self.show_body = json.dumps(
            {"model_info": {"digest": "sha256:daemon"}, "details": {}}
        )
        self.show_status = 200
        self.last_request = None
        self.last_path = None
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
                daemon.last_path = self.path
                if self.path == "/api/pull":
                    self._send(
                        daemon.pull_status,
                        daemon.pull_body,
                        "application/x-ndjson",
                    )
                elif self.path == "/api/show":
                    self._send(daemon.show_status, daemon.show_body, "application/json")
                else:
                    self._send(404, "{}", "application/json")

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


def test_reads_the_manifest_digest_from_the_daemon(daemon):
    assert llmman.daemon_model_digest(daemon.url, "org/model:tag") == "sha256:daemon"
    assert daemon.last_path == "/api/show"
    assert daemon.last_request == {"model": "org/model:tag"}


def test_rejects_a_show_response_without_a_digest(daemon):
    daemon.show_body = json.dumps({"model_info": {}})
    with pytest.raises(RuntimeError, match="did not return a manifest digest"):
        llmman.daemon_model_digest(daemon.url, "org/model:tag")


def test_pull_and_resolve_rejects_distinct_daemon_and_local_stores(daemon):
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="sha256:local\n", stderr=""
    )
    with (
        mock.patch.object(llmman, "endpoint", return_value=daemon.url),
        mock.patch.object(llmman, "_require_llmman_bin", return_value="llmman"),
        mock.patch.object(llmman.subprocess, "run", return_value=completed) as run,
        pytest.raises(RuntimeError, match="stores disagree"),
    ):
        llmman.pull_and_resolve("org/model:tag")

    run.assert_called_once_with(
        ["llmman", "list", "org/model:tag", "--format={{.Digest}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        check=False,
    )


def test_pull_and_resolve_uses_a_verified_shared_store(daemon, tmp_path):
    completed = [
        subprocess.CompletedProcess(
            args=[], returncode=0, stdout="sha256:daemon\n", stderr=""
        ),
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"path": str(tmp_path)}) + "\n",
            stderr="",
        ),
    ]
    with (
        mock.patch.object(llmman, "endpoint", return_value=daemon.url),
        mock.patch.object(llmman, "_require_llmman_bin", return_value="llmman"),
        mock.patch.object(llmman.subprocess, "run", side_effect=completed) as run,
    ):
        assert llmman.pull_and_resolve("org/model:tag") == str(tmp_path)

    assert run.call_args_list[1].args[0] == [
        "llmman",
        "resolve",
        "--no-pull",
        "org/model:tag",
    ]


@pytest.mark.parametrize(
    "reference,lookup_reference",
    [
        ("model", "model:latest"),
        ("org/model", "org/model:latest"),
        ("org/model:tag", "org/model:tag"),
        ("localhost:5000/org/model", "localhost:5000/org/model:latest"),
        ("[::1]:5000/org/model", "[::1]:5000/org/model:latest"),
        ("org/model@sha256:abc", "org/model@sha256:abc"),
    ],
)
def test_local_model_digest_uses_an_exact_reference(reference, lookup_reference):
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="sha256:local\n", stderr=""
    )
    with mock.patch.object(llmman.subprocess, "run", return_value=completed) as run:
        assert llmman.local_model_digest("llmman", reference) == "sha256:local"

    assert run.call_args.args[0] == [
        "llmman",
        "list",
        lookup_reference,
        "--format={{.Digest}}",
    ]


@pytest.mark.parametrize("stdout", ["", "sha256:a\nsha256:b\n"])
def test_local_model_digest_requires_exactly_one_manifest(stdout):
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=stdout, stderr=""
    )
    with (
        mock.patch.object(llmman.subprocess, "run", return_value=completed),
        pytest.raises(RuntimeError, match="same LLMMAN_MODELS store"),
    ):
        llmman.local_model_digest("llmman", "org/model:tag")
