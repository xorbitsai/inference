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

import asyncio
import io
import json
import threading
import time
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiohttp
import pytest
import requests
from aiohttp import web
from aiohttp.test_utils import TestServer

from xinference.client.restful.async_restful_client import AsyncRESTfulImageModelHandle
from xinference.client.restful.restful_client import RESTfulImageModelHandle


async def _echo_form(request: web.Request) -> web.Response:
    """Echo multipart fields; file parts carry their content."""
    fields: dict = {}
    for key, value in (await request.post()).items():
        if isinstance(value, web.FileField):
            value = value.file.read().decode()
        fields.setdefault(key, []).append(value)
    return web.json_response(fields)


@asynccontextmanager
async def _image_handle():
    app = web.Application()
    app.router.add_post("/v1/images/variations", _echo_form)
    app.router.add_post("/v1/images/inpainting", _echo_form)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        handle = AsyncRESTfulImageModelHandle.__new__(AsyncRESTfulImageModelHandle)
        handle._model_uid = "sd"
        handle._base_url = str(server.make_url("")).rstrip("/")
        handle.auth_headers = {}
        handle.session = session
        yield handle


async def test_async_image_to_image_sends_multipart_form():
    async with _image_handle() as handle:
        fields = await handle.image_to_image(
            [b"image-1", b"image-2"], "a cat", n=2, size="64*64"
        )

    assert fields["model"] == ["sd"]
    assert fields["prompt"] == ["a cat"]
    assert fields["n"] == ["2"]
    assert fields["size"] == ["64*64"]
    assert fields["image"] == ["image-1", "image-2"]


async def test_async_inpainting_sends_multipart_form():
    async with _image_handle() as handle:
        fields = await handle.inpainting(b"image", b"mask", "a cat", n=2)

    assert fields["model"] == ["sd"]
    assert fields["prompt"] == ["a cat"]
    assert fields["n"] == ["2"]
    assert fields["image"] == ["image"]
    assert fields["mask_image"] == ["mask"]


@asynccontextmanager
async def _edit_server():
    """Serves /a.png and echoes /v1/images/edits; records what reached each."""
    server_state = SimpleNamespace(downloads=[], edits=[])

    async def image(request: web.Request) -> web.Response:
        server_state.downloads.append(request.headers.get("Authorization"))
        return web.Response(body=b"downloaded")

    async def missing(request: web.Request) -> web.Response:
        return web.Response(status=404)

    async def edit(request: web.Request) -> web.Response:
        fields = json.loads((await _echo_form(request)).text)
        fields["authorization"] = [request.headers.get("Authorization")]
        server_state.edits.append(fields)
        return web.json_response(fields)

    app = web.Application()
    app.router.add_get("/a.png", image)
    app.router.add_get("/missing.png", missing)
    app.router.add_post("/v1/images/edits", edit)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        handle = AsyncRESTfulImageModelHandle.__new__(AsyncRESTfulImageModelHandle)
        handle._model_uid = "sd"
        handle._base_url = str(server.make_url("")).rstrip("/")
        handle.auth_headers = {"Authorization": "Bearer secret"}
        handle.session = session
        server_state.handle = handle
        server_state.url = handle._base_url
        yield server_state


async def test_async_image_edit_downloads_urls(tmp_path):
    path = tmp_path / "local.png"
    path.write_bytes(b"from-disk")
    async with _edit_server() as server:
        url = server.url + "/a.png"
        single = await server.handle.image_edit(url, "a cat")
        many = await server.handle.image_edit(
            [str(path), url, b"raw", io.BytesIO(b"file")], "a cat", mask=url
        )

    assert single["images"] == ["downloaded"]
    assert many["images[]"] == ["from-disk", "downloaded", "raw", "file"]
    assert many["mask"] == ["downloaded"]
    # The Xinference credentials go to Xinference only, never to the image host.
    assert single["authorization"] == many["authorization"] == ["Bearer secret"]
    assert server.downloads == [None, None, None]


async def test_async_image_edit_keeps_paths_bytes_and_file_objects(tmp_path):
    path = tmp_path / "local.png"
    path.write_bytes(b"from-disk")
    async with _edit_server() as server:
        single = await server.handle.image_edit(str(path), "a cat", mask=b"mask")
        many = await server.handle.image_edit(
            [str(path), b"raw", io.BytesIO(b"file")], "a cat", mask=str(path)
        )
        with pytest.raises(ValueError, match="Image list cannot be empty"):
            await server.handle.image_edit([], "a cat")

    assert single["images"] == ["from-disk"]
    assert single["mask"] == ["mask"]
    assert many["images[]"] == ["from-disk", "raw", "file"]
    assert many["mask"] == ["from-disk"]
    assert server.downloads == []
    assert len(server.edits) == 2


async def test_async_image_edit_failed_download_uploads_nothing():
    async with _edit_server() as server:
        with pytest.raises(RuntimeError, match="Failed to download an image") as error:
            await server.handle.image_edit(server.url + "/missing.png", "a cat")

    assert isinstance(error.value.__cause__, aiohttp.ClientResponseError)
    assert error.value.__cause__.status == 404
    assert server.edits == []


def _sync_edit_handle():
    """A sync handle whose session records downloads and uploads, with no network."""
    server_state = SimpleNamespace(downloads=[], uploads=[])
    handle = RESTfulImageModelHandle.__new__(RESTfulImageModelHandle)
    handle._model_uid = "sd"
    handle._base_url = "http://127.0.0.1:9997"
    handle.auth_headers = {"Authorization": "Bearer secret"}
    handle.session = MagicMock()

    def get(url, **kwargs):
        server_state.downloads.append((url, kwargs))
        response = MagicMock(status_code=200, content=b"downloaded")
        if url.endswith("/missing.png"):
            response.raise_for_status.side_effect = requests.HTTPError("404")
        return response

    def post(url, files=None, headers=None):
        # Read the parts now: the handle closes the files it opened afterwards.
        parts = [
            (name, part[1].read() if hasattr(part[1], "read") else part[1])
            for name, part in files
            if part[0] is not None
        ]
        server_state.uploads.append((url, headers, parts))
        return MagicMock(status_code=200, **{"json.return_value": {"data": []}})

    handle.session.get.side_effect = get
    handle.session.post.side_effect = post
    server_state.handle = handle
    return server_state


def test_sync_image_edit_downloads_urls(tmp_path):
    path = tmp_path / "local.png"
    path.write_bytes(b"from-disk")
    server = _sync_edit_handle()
    url = "https://example.com/a.png"

    server.handle.image_edit(url, "a cat")
    server.handle.image_edit(
        [str(path), url, b"raw", io.BytesIO(b"file")], "a cat", mask=url
    )

    (_, headers, single), (_, _, many) = server.uploads
    assert single == [("images", b"downloaded")]
    assert many == [
        ("images[]", b"from-disk"),
        ("images[]", b"downloaded"),
        ("images[]", b"raw"),
        ("images[]", b"file"),
        ("mask", b"downloaded"),
    ]
    # The Xinference credentials go to Xinference only, never to the image host.
    assert headers == {"Authorization": "Bearer secret"}
    assert [fetched for fetched, _ in server.downloads] == [url] * 3
    assert all("headers" not in kwargs for _, kwargs in server.downloads)


def test_sync_image_edit_keeps_paths_bytes_and_file_objects(tmp_path):
    path = tmp_path / "local.png"
    path.write_bytes(b"from-disk")
    server = _sync_edit_handle()

    server.handle.image_edit(str(path), "a cat", mask=b"mask")
    server.handle.image_edit(
        [str(path), b"raw", io.BytesIO(b"file")], "a cat", mask=str(path)
    )
    with pytest.raises(ValueError, match="Image list cannot be empty"):
        server.handle.image_edit([], "a cat")

    (_, _, single), (_, _, many) = server.uploads
    assert single == [("images", b"from-disk"), ("mask", b"mask")]
    assert many == [
        ("images[]", b"from-disk"),
        ("images[]", b"raw"),
        ("images[]", b"file"),
        ("mask", b"from-disk"),
    ]
    assert server.downloads == []


def test_sync_image_edit_failed_download_uploads_nothing():
    server = _sync_edit_handle()

    with pytest.raises(RuntimeError, match="Failed to download an image") as error:
        server.handle.image_edit("https://example.com/missing.png", "a cat")

    assert isinstance(error.value.__cause__, requests.HTTPError)
    assert server.uploads == []


_STALL_SECONDS = 4


@pytest.fixture
def host():
    """Loopback image host. /ok.png answers at once; /stall.png sends its headers
    and the first byte of the body, then stays quiet. Edit uploads are recorded."""
    state = SimpleNamespace(
        uploads=[], stalled=threading.Event(), release=threading.Event()
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body, content_type, length=None):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length or len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/stall.png":
                return self.reply(b"image", "image/png")
            self.reply(b"x", "image/png", length=1000)
            state.stalled.set()
            state.release.wait(_STALL_SECONDS)

        def do_POST(self):
            state.uploads.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.reply(b'{"data": []}', "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield state
    finally:
        state.release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def _shrink(timeout):
    """Cap a finite bound at 0.3 s so the stall tests take a fraction of a second.
    None (no bound) stays None, so a call without a bound still waits."""
    if isinstance(timeout, tuple):
        return tuple(_shrink(bound) for bound in timeout)
    return None if timeout is None else min(timeout, 0.3)


class _RecordingAdapter(requests.adapters.HTTPAdapter):
    """Records the timeout each request reaches the adapter with, then shrinks it."""

    def __init__(self, seen):
        super().__init__()
        self.seen = seen

    def send(self, request, stream=False, timeout=None, **kwargs):
        self.seen.append((request.method, timeout))
        return super().send(request, stream=stream, timeout=_shrink(timeout), **kwargs)


class _RecordingSession:
    """Wraps the handle's aiohttp session: records the timeout each request is
    sent with, then shrinks it."""

    def __init__(self, session, seen):
        self._session = session
        self._seen = seen

    def __getattr__(self, name):
        return getattr(self._session, name)

    def _send(self, method, url, kwargs):
        timeout = kwargs.get("timeout")
        self._seen.append((method, timeout))
        if timeout is not None:
            kwargs["timeout"] = aiohttp.ClientTimeout(
                total=timeout.total,
                sock_connect=_shrink(timeout.sock_connect),
                sock_read=_shrink(timeout.sock_read),
            )
        return getattr(self._session, method.lower())(url, **kwargs)

    def get(self, url, **kwargs):
        return self._send("GET", url, kwargs)

    def post(self, url, **kwargs):
        return self._send("POST", url, kwargs)


# Where the stalled URL sits in the call: the image, a list entry, or the mask.
_STALLS = {
    "image": lambda ok, stall: {"images": stall},
    "list_entry": lambda ok, stall: {"images": [ok, stall]},
    "mask": lambda ok, stall: {"images": ok, "mask": stall},
}


def test_sync_image_edit_bounds_each_download_but_not_the_edit_request(host):
    seen: list = []
    handle = RESTfulImageModelHandle("sd", host.url, {})
    handle.session.mount("http://", _RecordingAdapter(seen))
    try:
        handle.image_edit(host.url + "/ok.png", "a cat", mask=host.url + "/ok.png")
    finally:
        handle.close()

    assert [method for method, _ in seen] == ["GET", "GET", "POST"]
    for _, timeout in seen[:2]:  # the image, then the mask
        assert isinstance(timeout, tuple), f"download timeout: {timeout!r}"
        connect, read = timeout
        assert 0 < connect <= 60 and 0 < read <= 60
    # Generation can take minutes: the edit request keeps waiting for its answer.
    assert seen[2][1] is None


@pytest.mark.parametrize("where", _STALLS)
def test_sync_image_edit_stalled_download_fails_without_uploading(host, where):
    handle = RESTfulImageModelHandle("sd", host.url, {})
    handle.session.mount("http://", _RecordingAdapter([]))
    kwargs = _STALLS[where](host.url + "/ok.png", host.url + "/stall.png")
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="Failed to download an image") as error:
            handle.image_edit(prompt="a cat", **kwargs)
        assert isinstance(error.value.__cause__, requests.RequestException)
    finally:
        handle.close()
    elapsed = time.monotonic() - started

    assert host.stalled.is_set()  # the client got as far as the stalled body
    assert elapsed < _STALL_SECONDS / 2, f"waited {elapsed:.1f}s on a stalled download"
    assert host.uploads == []


async def test_async_image_edit_bounds_each_download_but_not_the_edit_request(host):
    seen: list = []
    handle = AsyncRESTfulImageModelHandle("sd", host.url, {})
    handle.session = _RecordingSession(handle.session, seen)
    try:
        await handle.image_edit(
            host.url + "/ok.png", "a cat", mask=host.url + "/ok.png"
        )
    finally:
        await handle.close()

    assert [method for method, _ in seen] == ["GET", "GET", "POST"]
    for _, timeout in seen[:2]:  # the image, then the mask
        assert timeout is not None, "download carries no timeout of its own"
        assert 0 < timeout.sock_connect <= 60 and 0 < timeout.sock_read <= 60
        # The session's overall bound stays in force next to the new ones.
        assert timeout.total == handle.timeout.total
    # Generation can take minutes: the edit request keeps waiting for its answer.
    assert seen[2][1] is None


@pytest.mark.parametrize("where", _STALLS)
async def test_async_image_edit_stalled_download_fails_without_uploading(host, where):
    handle = AsyncRESTfulImageModelHandle("sd", host.url, {})
    handle.session = _RecordingSession(handle.session, [])
    kwargs = _STALLS[where](host.url + "/ok.png", host.url + "/stall.png")
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="Failed to download an image") as error:
            await handle.image_edit(prompt="a cat", **kwargs)
        assert isinstance(error.value.__cause__, asyncio.TimeoutError)
    finally:
        await handle.close()
    elapsed = time.monotonic() - started

    assert host.stalled.is_set()  # the client got as far as the stalled body
    assert elapsed < _STALL_SECONDS / 2, f"waited {elapsed:.1f}s on a stalled download"
    assert host.uploads == []
