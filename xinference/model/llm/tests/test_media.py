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

import base64
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import urlparse

import pytest
from PIL import Image

from ..media import (
    fetch_media,
    load_media_bytes,
    materialize_messages_media,
    media_workspace,
    validate_media_url,
    validate_messages_media,
)

PNG_BYTES = None


def _png_bytes():
    global PNG_BYTES
    if PNG_BYTES is None:
        buffer = BytesIO()
        Image.new("RGB", (2, 2), (1, 2, 3)).save(buffer, format="PNG")
        PNG_BYTES = buffer.getvalue()
    return PNG_BYTES


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        try:
            self._serve()
        except OSError:
            pass

    def _serve(self):
        if self.path == "/ok":
            body = _png_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/big":
            self.send_response(200)
            self.end_headers()
            for _ in range(8):
                self.wfile.write(b"x" * 4096)
                self.wfile.flush()
        elif self.path == "/lag":
            time.sleep(0.4)
            body = _png_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/gzip":
            import gzip

            body = gzip.compress(_png_bytes())
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/redirect-internal":
            self.send_response(302)
            self.send_header(
                "Location", f"http://127.0.0.1:{self.server.server_port}/ok"
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/redirect":
            # Stall in the header phase: only a per-hop budget catches this.
            time.sleep(3)
            self.send_response(302)
            self.send_header("Location", "/slow")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/slow":
            self.send_response(200)
            self.end_headers()
            # Dribble forever: defeats per-socket timeouts, not a total deadline.
            for _ in range(400):
                try:
                    self.wfile.write(b"x")
                    self.wfile.flush()
                except OSError:
                    return
                time.sleep(0.1)


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.fixture
def allow_loopback(monkeypatch):
    monkeypatch.setenv("XINFERENCE_MEDIA_BLOCK_PRIVATE_ADDRESS", "false")
    monkeypatch.setenv("XINFERENCE_MEDIA_FETCH_TIMEOUT", "1")


@pytest.mark.parametrize(
    "url",
    [
        "/etc/hostname",
        "/root/.xinference/config.json",
        "file:///etc/hostname",
        "FILE:///etc/hostname",
        "C:\\Windows\\win.ini",
        "../../etc/passwd",
    ],
)
def test_local_paths_rejected(url):
    with pytest.raises(ValueError, match="Local file"):
        validate_media_url(url)


def test_local_paths_allowed_by_opt_in(monkeypatch):
    monkeypatch.setenv("XINFERENCE_MEDIA_ALLOW_LOCAL_PATH", "true")
    assert validate_media_url("/etc/hostname") == "file"


@pytest.mark.parametrize("url", ["", None, 123, "   "])
def test_empty_urls_rejected(url):
    with pytest.raises(ValueError):
        validate_media_url(url)


def test_unsupported_scheme_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        validate_media_url("ftp://example.com/a.png")


@pytest.mark.parametrize("prefix", ["data", "DATA", "DaTa"])
def test_data_uri_scheme_is_case_insensitive(prefix):
    payload = base64.b64encode(b"hello").decode()
    assert validate_media_url(f"{prefix}:image/png;base64,{payload}") == "data"


@pytest.mark.parametrize(
    "url",
    [
        "data:image/png;base64,!!!!",
        "data:image/png;base64",  # no comma at all
        "data:image/png;base64,QUJD=",  # invalid padding
        "data:image;base64,QUJD",  # mediatype without subtype
        "datax:image/png;base64,QUJD",
    ],
)
def test_malformed_data_uris_rejected(url):
    with pytest.raises(ValueError):
        validate_media_url(url)


def test_data_uri_without_base64_marker():
    assert load_media_bytes("data:,hello%20world") == b"hello world"


def test_http_scheme_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("XINFERENCE_MEDIA_BLOCK_PRIVATE_ADDRESS", "false")
    assert validate_media_url("HTTP://example.com/a.png") == "http"
    assert validate_media_url("HttpS://example.com/a.png") == "https"


def test_private_address_blocked_by_default(server):
    with pytest.raises(ValueError):
        validate_media_url(f"{server}/ok")


def test_private_address_allowed_when_opted_out(server, allow_loopback):
    assert validate_media_url(f"{server}/ok") == "http"


def test_fetch_media_success(server, allow_loopback):
    assert fetch_media(f"{server}/ok") == _png_bytes()


def test_fetch_media_enforces_total_deadline(server, allow_loopback):
    started = time.monotonic()
    with pytest.raises(ValueError, match="time budget"):
        fetch_media(f"{server}/slow")
    assert time.monotonic() - started < 10


@pytest.fixture
def only_public_test_host(monkeypatch):
    """Treat one fake hostname as public and every literal address as private."""
    from ...image import utils as image_utils

    def fake(url, require_public=True):
        parsed = urlparse(url)
        if require_public and parsed.hostname != "public.test":
            raise ValueError("URL must resolve only to public addresses")
        return parsed, ["127.0.0.1"]

    monkeypatch.delenv("XINFERENCE_MEDIA_BLOCK_PRIVATE_ADDRESS", raising=False)
    monkeypatch.setenv("XINFERENCE_MEDIA_FETCH_TIMEOUT", "5")
    monkeypatch.setattr(image_utils, "_public_addresses", fake)


def test_fetch_media_deadline_spans_redirects(server, allow_loopback):
    started = time.monotonic()
    with pytest.raises(ValueError, match="time budget"):
        fetch_media(f"{server}/redirect")
    # The handler stalls 3s before the 302; a 1s budget must not wait it out.
    assert time.monotonic() - started < 2.5


def test_fetch_media_enforces_size_cap(server, allow_loopback, monkeypatch):
    monkeypatch.setenv("XINFERENCE_MEDIA_MAX_BYTES", "1024")
    with pytest.raises(ValueError, match="exceeds"):
        fetch_media(f"{server}/big")


def test_decode_image_rejects_local_path(tmp_path):
    from ..utils import _decode_image, _decode_image_without_rgb

    image_path = tmp_path / "secret.png"
    Image.new("RGB", (2, 2)).save(image_path)
    for decode in (_decode_image, _decode_image_without_rgb):
        with pytest.raises(ValueError, match="Local file"):
            decode(str(image_path))


def test_decode_image_accepts_data_uri_and_http(server, allow_loopback):
    from ..utils import _decode_image

    data_uri = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
    assert _decode_image(data_uri).size == (2, 2)
    assert _decode_image(f"{server}/ok").size == (2, 2)


@pytest.mark.parametrize(
    ("module", "cls_name", "key"),
    [
        ("deepseek_vl2", "DeepSeekVL2ChatModel", "image_url"),
        ("qwen2_audio", "Qwen2AudioChatModel", "audio_url"),
    ],
)
def test_multimodal_chat_rejects_local_media(module, cls_name, key):
    """deepseek_vl2 never calls _transform_messages, qwen2_audio overrides it."""
    import importlib

    model_cls = getattr(
        importlib.import_module(
            f"..transformers.multimodal.{module}", package=__package__
        ),
        cls_name,
    )
    model = object.__new__(model_cls)
    messages = [
        {
            "role": "user",
            "content": [{"type": key, key: {"url": "file:///etc/hostname"}}],
        }
    ]
    with pytest.raises(ValueError, match="Local file"):
        model.chat(messages, {"stream": False})


class _Blocked(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "cls_name", "method"),
    [
        ("..vllm.core", "VLLMMultiModel", "async_chat"),
        ("..sglang.core", "SGLANGVisionModel", "async_chat"),
        ("..lmdeploy.core", "LMDeployChatModel", "_get_prompt_input"),
    ],
)
async def test_media_validation_runs_off_event_loop(
    module, cls_name, method, monkeypatch
):
    """validate_media_url resolves DNS per part; a bare call blocks the actor."""
    import asyncio
    import importlib
    import sys
    import types
    from unittest.mock import MagicMock

    def recorder(messages, *args):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            raise _Blocked()
        raise AssertionError("media handling ran on the event loop")

    qwen_vl_utils = types.ModuleType("qwen_vl_utils")
    qwen_vl_utils.process_vision_info = MagicMock()
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_vl_utils)
    qwen_omni_utils = types.ModuleType("qwen_omni_utils")
    for name in ("process_audio_info", "process_mm_info", "process_vision_info"):
        setattr(qwen_omni_utils, name, MagicMock())
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", qwen_omni_utils)

    mod = importlib.import_module(module, package=__package__)
    patched = 0
    for name in ("materialize_messages_media", "validate_messages_media"):
        if hasattr(mod, name):
            monkeypatch.setattr(mod, name, recorder)
            patched += 1
    assert patched, f"{module} no longer handles media"
    model = object.__new__(getattr(mod, cls_name))
    model.model_family = MagicMock()

    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
    ]
    with pytest.raises(_Blocked):
        await getattr(model, method)(messages, {})


def test_fetch_media_decodes_content_encoding(server, allow_loopback):
    """Reading the raw stream must not hand back the still-compressed body."""
    assert fetch_media(f"{server}/gzip") == _png_bytes()


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image", "url": "/etc/hostname"},
        # HF processing_utils pulls "path" out of a part whatever its type, and
        # minicpm-v forwards unrecognised parts to apply_chat_template verbatim.
        {"type": "image", "path": "/etc/hostname"},
        {"type": "audio", "path": "/etc/hostname"},
        {"type": "video", "video": ["/etc/hostname"]},
        # load_image tries os.path.isfile() before decoding base64.
        {"type": "image", "base64": "/etc/hostname"},
        {"type": "video", "video": [["/etc/hostname"]]},
        {"type": "image_url", "image_url": {"url": ["/etc/hostname"]}},
    ],
)
def test_media_reference_keys_are_validated(part):
    messages = [{"role": "user", "content": [part]}]
    with pytest.raises(ValueError, match="Local file|non-empty string"):
        validate_messages_media(messages)


def test_materialize_replaces_remote_url_with_local_copy(server, allow_loopback):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"{server}/ok"}},
                {"type": "video_url", "video_url": {"url": f"{server}/ok"}},
            ],
        }
    ]
    with media_workspace() as workspace:
        materialize_messages_media(messages, workspace)
        image_url = messages[0]["content"][0]["image_url"]["url"]
        video_url = messages[0]["content"][1]["video_url"]["url"]
        assert image_url.startswith(workspace + os.sep)
        # qwen's video reader needs a URI, its image reader a bare path.
        assert video_url.startswith("file://")
        assert open(image_url, "rb").read() == _png_bytes()
        # Prompt builders re-validate what materialization handed them.
        assert load_media_bytes(image_url) == _png_bytes()
        assert load_media_bytes(video_url) == _png_bytes()
    assert not os.path.exists(image_url)
    with pytest.raises(ValueError, match="Local file"):
        validate_media_url(image_url)


def test_materialize_blocks_redirect_into_private_network(
    server, only_public_test_host
):
    """The bypass pre-flight validation alone cannot close."""
    port = urlparse(server).port
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"http://public.test:{port}/redirect-internal"
                    },
                }
            ],
        }
    ]
    # Validation passes: the first hop really is "public".
    validate_messages_media(messages)
    with media_workspace() as workspace:
        with pytest.raises(ValueError, match="public addresses"):
            materialize_messages_media(messages, workspace)


def test_import_creates_no_threads():
    """A module-level thread pool here deadlocks vLLM's forked EngineCore."""
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import threading, xinference.model.llm.media, xinference.model.llm.utils;"
            "assert threading.active_count() == 1, threading.enumerate()",
        ],
        check=True,
    )


@pytest.fixture
def dribble_header_server():
    """Serve a response whose headers arrive one byte at a time, forever."""
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=drip, args=(conn,), daemon=True).start()

    def drip(conn):
        try:
            conn.recv(65536)
            conn.sendall(b"HTTP/1.1 200 OK\r\n")
            # One byte per 0.1s: each one resets the recv-level timeout.
            conn.sendall(b"X-Pad: ")
            # Bounded: an endless drip would hang the run instead of failing it
            # if the watchdog ever regresses.
            for _ in range(600):
                if stop.is_set():
                    return
                conn.sendall(b"a")
                time.sleep(0.1)
        except OSError:
            pass
        finally:
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    yield f"http://127.0.0.1:{listener.getsockname()[1]}/x"
    stop.set()
    listener.close()


def test_fetch_media_bounds_the_header_phase(dribble_header_server, allow_loopback):
    """urllib3 timeouts are per recv(); only closing the socket bounds this."""
    started = time.monotonic()
    with pytest.raises(ValueError, match="time budget"):
        fetch_media(dribble_header_server)
    # allow_loopback sets a 1s budget.
    assert time.monotonic() - started < 4


def _lag_messages(server, count):
    return [
        {
            "role": "user",
            # Distinct dicts: repeating one would be rewritten once and skipped.
            "content": [
                {"type": "image_url", "image_url": {"url": f"{server}/lag"}}
                for _ in range(count)
            ],
        }
    ]


def test_materialize_allows_several_slow_parts(server, allow_loopback):
    """A per-url budget shared across parts starves legitimate many-image requests."""
    with media_workspace() as workspace:
        # 4 x 0.4s is over the 1s per-url budget but under the request cap.
        materialize_messages_media(_lag_messages(server, 4), workspace)


def test_materialize_caps_the_whole_request(server, allow_loopback):
    """Per-url budgets alone multiply by part count, each holding a thread."""
    started = time.monotonic()
    with media_workspace() as workspace:
        with pytest.raises(ValueError, match="time budget"):
            materialize_messages_media(_lag_messages(server, 20), workspace)
    # allow_loopback sets a 1s per-url budget; the request cap is 3x that.
    assert time.monotonic() - started < 4


def test_materialize_rewrites_list_valued_media(server, allow_loopback):
    messages = [
        {
            "role": "user",
            "content": [{"type": "video", "video": [f"{server}/ok", f"{server}/ok"]}],
        }
    ]
    with media_workspace() as workspace:
        materialize_messages_media(messages, workspace)
        frames = messages[0]["content"][0]["video"]
    assert all(frame.startswith("file://") for frame in frames)
    assert frames[0] != frames[1]


def test_deep_nesting_is_a_client_error():
    """json.loads accepts nesting far deeper than the walker can recurse."""
    import json

    deep = json.loads('{"video": ' + "[" * 3000 + '"/etc/hostname"' + "]" * 3000 + "}")
    messages = [{"role": "user", "content": [dict(type="video", **deep)]}]
    with pytest.raises(ValueError, match="too deep"):
        validate_messages_media(messages)
    with media_workspace() as workspace:
        with pytest.raises(ValueError, match="too deep"):
            materialize_messages_media(messages, workspace)
