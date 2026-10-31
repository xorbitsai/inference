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

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from xinference.client.restful.async_restful_client import (
    AsyncClient,
    _get_error_string,
)


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


@pytest.mark.parametrize(
    "body, detail",
    [
        pytest.param(b"<html>Bad Gateway</html>", None, id="html"),
        pytest.param(b"", None, id="empty"),
        pytest.param(b'{"error": "Bad Gateway"}', None, id="missing-detail"),
        pytest.param(b'{"detail": "Test error"}', "Test error", id="detail"),
    ],
)
@pytest.mark.parametrize("operation", ["vllm_models", "login"])
async def test_async_client_error_responses(body, detail, operation):
    response = Mock(spec=aiohttp.ClientResponse)
    response.status = 502
    response.content = Mock()
    response.json = AsyncMock(side_effect=lambda: json.loads(body))
    request_info = SimpleNamespace(real_url="http://localhost:9997/test")
    if body.startswith(b"<html>"):
        response.json.side_effect = aiohttp.ContentTypeError(
            request_info=request_info,
            history=(),
            status=502,
            message="Attempt to decode JSON with unexpected mimetype: text/html",
        )
    response.raise_for_status.side_effect = aiohttp.ClientResponseError(
        request_info=request_info,
        history=(),
        status=502,
        message="Bad Gateway",
    )
    response.wait_for_close = AsyncMock()
    client = AsyncClient.__new__(AsyncClient)
    client.base_url = "http://localhost:9997"
    client._headers = {}
    client._cluster_authed = True
    session = Mock(spec=aiohttp.ClientSession)
    session.get = AsyncMock(return_value=response)
    session.post = AsyncMock(return_value=response)
    client.session = session
    args = ("user", "password") if operation == "login" else ()

    try:
        with pytest.raises(RuntimeError) as exc:
            await getattr(client, operation)(*args)
        if detail is None:
            detail = "502, message='Bad Gateway', url='http://localhost:9997/test'"
        assert str(exc.value) == "Failed to get cluster information, detail: " + detail
        response.release.assert_called_once_with()
        response.wait_for_close.assert_awaited_once_with()
    finally:
        await client.close()
