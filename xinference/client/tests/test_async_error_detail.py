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

from xinference.client.restful.async_restful_client import _get_error_string


async def _get(handler) -> str:
    app = web.Application()
    app.router.add_get("/", handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        response = await session.get(server.make_url("/"))
        return await _get_error_string(response)


async def test_async_error_string_prefers_server_detail():
    async def detail(request: web.Request) -> web.Response:
        return web.json_response({"detail": "Test error"}, status=500)

    assert await _get(detail) == "Test error"


async def test_async_error_string_falls_back_to_status():
    async def bare(request: web.Request) -> web.Response:
        return web.Response(status=502, reason="Bad Gateway")

    assert (await _get(bare)).startswith("502, message='Bad Gateway'")
