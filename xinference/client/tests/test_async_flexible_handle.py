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

from xinference.client.restful.async_restful_client import (
    AsyncRESTfulFlexibleModelHandle,
)


async def _flexible_infer(request: web.Request) -> web.Response:
    # Same contract as RESTfulAPI.create_flexible_infer: infer(*args, **kwargs).
    payload = await request.json()
    args = payload.get("args")
    kwargs = {k: v for k, v in payload.items() if k not in {"model", "args"}}
    try:
        result = {"args": list(args), "kwargs": kwargs}
    except TypeError as e:
        return web.json_response({"detail": str(e)}, status=500)
    return web.json_response(result)


async def test_async_flexible_infer_sends_args_and_returns_json():
    app = web.Application()
    app.router.add_post("/v1/flexible/infers", _flexible_infer)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        handle = AsyncRESTfulFlexibleModelHandle.__new__(
            AsyncRESTfulFlexibleModelHandle
        )
        handle._model_uid = "flex"
        handle._base_url = str(server.make_url("")).rstrip("/")
        handle.auth_headers = {}
        handle.session = session

        assert await handle.infer(x=3) == {"args": [], "kwargs": {"x": 3}}
        assert await handle.infer(1, "a", x=3) == {
            "args": [1, "a"],
            "kwargs": {"x": 3},
        }
