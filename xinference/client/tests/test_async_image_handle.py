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

import io
import json
from contextlib import asynccontextmanager
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
        with pytest.raises(aiohttp.ClientResponseError) as error:
            await server.handle.image_edit(server.url + "/missing.png", "a cat")

    assert error.value.status == 404
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

    with pytest.raises(requests.HTTPError):
        server.handle.image_edit("https://example.com/missing.png", "a cat")

    assert server.uploads == []
