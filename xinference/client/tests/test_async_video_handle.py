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

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from xinference.client.restful.async_restful_client import AsyncRESTfulVideoModelHandle


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
