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

from unittest.mock import MagicMock

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from xinference.client.restful.async_restful_client import AsyncRESTfulVideoModelHandle
from xinference.client.restful.restful_client import RESTfulVideoModelHandle


async def _echo_form(request: web.Request) -> web.Response:
    """Echo multipart fields; file parts carry their content."""
    fields: dict = {}
    for key, value in (await request.post()).items():
        if isinstance(value, web.FileField):
            value = value.file.read().decode()
        fields.setdefault(key, []).append(value)
    return web.json_response(fields)


async def test_async_flf_to_video_sends_multipart_form():
    app = web.Application()
    app.router.add_post("/v1/video/generations/flf", _echo_form)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        handle = AsyncRESTfulVideoModelHandle.__new__(AsyncRESTfulVideoModelHandle)
        handle._model_uid = "wan"
        handle._base_url = str(server.make_url("")).rstrip("/")
        handle.auth_headers = {}
        handle.session = session
        fields = await handle.flf_to_video(b"first", b"last", "a cat", n=2)

    assert fields["model"] == ["wan"]
    assert fields["prompt"] == ["a cat"]
    assert fields["n"] == ["2"]
    assert fields["first_frame"] == ["first"]
    assert fields["last_frame"] == ["last"]


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


async def _async_handle(server, session):
    handle = AsyncRESTfulVideoModelHandle.__new__(AsyncRESTfulVideoModelHandle)
    handle._model_uid = "wan"
    handle._base_url = str(server.make_url("")).rstrip("/")
    handle.auth_headers = {}
    handle.session = session
    return handle


async def test_async_video_handles_read_image_paths(tmp_path):
    first = _write(tmp_path, "first.png", b"first")
    last = _write(tmp_path, "last.png", b"last")
    app = web.Application()
    app.router.add_post("/v1/video/generations/image", _echo_form)
    app.router.add_post("/v1/video/generations/flf", _echo_form)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        handle = await _async_handle(server, session)
        i2v = await handle.image_to_video(first, "a cat")
        flf = await handle.flf_to_video(first, last, "a cat")

    assert i2v["image"] == ["first"]
    assert flf["first_frame"] == ["first"]
    assert flf["last_frame"] == ["last"]


def _sync_handle():
    handle = RESTfulVideoModelHandle.__new__(RESTfulVideoModelHandle)
    handle._model_uid = "wan"
    handle._base_url = "http://127.0.0.1:9997"
    handle.auth_headers = {}
    handle.session = MagicMock()
    handle.session.post.return_value = MagicMock(status_code=200)
    return handle


def _uploads(handle):
    _, kwargs = handle.session.post.call_args
    return {key: value[1] for key, value in kwargs["files"] if value[0] is not None}


def test_sync_video_handles_read_image_paths(tmp_path):
    first = _write(tmp_path, "first.png", b"first")
    last = _write(tmp_path, "last.png", b"last")
    handle = _sync_handle()

    handle.image_to_video(first, "a cat")
    assert _uploads(handle) == {"image": b"first"}

    handle.flf_to_video(first, last, "a cat")
    assert _uploads(handle) == {"first_frame": b"first", "last_frame": b"last"}

    handle.image_to_video(b"raw", "a cat")
    assert _uploads(handle) == {"image": b"raw"}
