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

import gc
import http.server
import threading

import pytest
import requests

from ..restful.restful_client import RESTfulChatModelHandle, RESTfulGenerateModelHandle


@pytest.fixture
def streaming_server():
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            payload = self.server.payload
            try:
                self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
                self.wfile.flush()
                self.server.finish.wait(timeout=5)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass
            finally:
                self.close_connection = True

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.finish = threading.Event()
    server.payload = b'data: {"choices": [{"text": "first"}]}\n\ndata: [DONE]\n\n'
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield server
    finally:
        server.finish.set()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "handle_type", [RESTfulGenerateModelHandle, RESTfulChatModelHandle]
)
@pytest.mark.parametrize(
    "finish",
    ["unstarted", "partial", "done", "error", "discarded", "discarded-unstarted"],
)
def test_sync_stream_releases_response(streaming_server, handle_type, finish):
    if finish == "error":
        streaming_server.payload = b'data: {"error": "stream failed"}\n\n'
    url = "http://127.0.0.1:%d" % streaming_server.server_port
    handle = handle_type("model", url, {})
    responses = []
    handle.session.hooks["response"].append(
        lambda response, *args, **kwargs: responses.append(response)
    )
    try:
        if handle_type is RESTfulChatModelHandle:
            stream = handle.chat(
                [{"role": "user", "content": "hello"}], generate_config={"stream": True}
            )
        else:
            stream = handle.generate("hello", generate_config={"stream": True})
        response = responses[0]
        assert not response.raw.closed

        if finish == "partial":
            assert next(stream)["choices"][0]["text"] == "first"
            stream.close()
        elif finish == "done":
            assert len(list(stream)) == 1
        elif finish == "error":
            with pytest.raises(Exception, match="stream failed"):
                next(stream)
        elif finish.startswith("discarded"):
            if finish == "discarded":
                next(stream)
            del stream
            gc.collect()
        else:
            stream.close()
            stream.close()

        assert response.raw.closed
    finally:
        handle.close()


def test_closing_unstarted_stream_releases_connection_pool(streaming_server):
    url = "http://127.0.0.1:%d" % streaming_server.server_port
    handle = RESTfulGenerateModelHandle("model", url, {})
    responses = []
    handle.session.hooks["response"].append(
        lambda response, *args, **kwargs: responses.append(response)
    )
    handle.session.mount(
        url, requests.adapters.HTTPAdapter(pool_maxsize=1, pool_block=True)
    )
    stream = handle.generate("hello", generate_config={"stream": True})
    first_response = responses[0]
    stream.close()

    completed = threading.Event()
    errors = []

    def next_request():
        try:
            response = handle.session.post(url, json={}, stream=True)
            response.close()
        except Exception as exc:
            errors.append(exc)
        finally:
            completed.set()

    thread = threading.Thread(target=next_request, daemon=True)
    thread.start()
    try:
        available = completed.wait(timeout=1)
        if not available:
            first_response.close()
        thread.join(timeout=5)
        assert available, "the closed stream kept the only connection checked out"
        assert not thread.is_alive()
        assert not errors
    finally:
        first_response.close()
        handle.close()
