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

from xinference.client.restful.async_restful_client import AsyncClient


async def test_async_launch_model_without_waiting():
    async def launch(request: web.Request) -> web.Response:
        return web.json_response({"model_uid": request.query.get("wait_ready")})

    app = web.Application()
    app.router.add_post("/v1/models", launch)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        client = AsyncClient.__new__(AsyncClient)
        client.base_url = str(server.make_url("")).rstrip("/")
        client._headers = {}
        client.session = session

        assert await client.launch_model("qwen", wait_ready=False) == "false"
        assert await client.launch_model("qwen") is None
