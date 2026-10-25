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

from contextlib import asynccontextmanager

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from xinference.client.restful.async_restful_client import AsyncRESTfulImageModelHandle


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
