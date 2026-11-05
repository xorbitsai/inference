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
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from ..restful.async_restful_client import (
    AsyncRESTfulChatModelHandle,
    AsyncRESTfulGenerateModelHandle,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handle_type", [AsyncRESTfulGenerateModelHandle, AsyncRESTfulChatModelHandle]
)
@pytest.mark.parametrize("finish", ["partial", "done", "error"])
async def test_async_stream_releases_response_once(handle_type, finish):
    async def lines():
        if finish == "error":
            yield b'data: {"error": "stream failed"}\n'
        else:
            yield b'data: {"choices": [{"text": "first"}]}\n'
            yield b"data: [DONE]\n"

    response = MagicMock(status=200)
    response.content = lines()
    response.wait_for_close = AsyncMock()
    handle = handle_type.__new__(handle_type)
    handle._model_uid = "model"
    handle._base_url = "http://localhost"
    handle.auth_headers = {}
    handle.session = MagicMock()
    handle.session.post = AsyncMock(return_value=response)
    try:
        if handle_type is AsyncRESTfulChatModelHandle:
            stream = await handle.chat(
                [{"role": "user", "content": "hello"}], generate_config={"stream": True}
            )
        else:
            stream = await handle.generate("hello", generate_config={"stream": True})
        if finish == "partial":
            assert (await anext(stream))["choices"][0]["text"] == "first"
            await stream.aclose()
        elif finish == "done":
            assert len([chunk async for chunk in stream]) == 1
        else:
            with pytest.raises(Exception, match="stream failed"):
                await anext(stream)
        response.release.assert_called_once()
        response.wait_for_close.assert_awaited_once()
    finally:
        handle.session = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handle_type", [AsyncRESTfulGenerateModelHandle, AsyncRESTfulChatModelHandle]
)
async def test_close_before_read_releases_async_connection(handle_type):
    finish = asyncio.Event()

    async def stream_response(request):
        await request.read()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices": [{"text": "first"}]}\n\n')
        await finish.wait()
        try:
            await response.write_eof()
        except ConnectionResetError:
            pass
        return response

    app = web.Application()
    app.router.add_post("/{path:.*}", stream_response)
    async with TestServer(app) as server:
        url = str(server.make_url("")).rstrip("/")
        handle = handle_type("model", url, {})
        await handle.session.close()
        responses = []
        trace = aiohttp.TraceConfig()

        async def record_response(_session, _context, params):
            responses.append(params.response)

        trace.on_request_end.append(record_response)
        handle.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=1), trace_configs=[trace]
        )
        try:
            if handle_type is AsyncRESTfulChatModelHandle:
                stream = await handle.chat(
                    [{"role": "user", "content": "hello"}],
                    generate_config={"stream": True},
                )
            else:
                stream = await handle.generate(
                    "hello", generate_config={"stream": True}
                )
            first_response = responses[0]
            assert not first_response.closed
            await stream.aclose()

            async def next_request():
                response = await handle.session.post(url, json={})
                response.release()
                await response.wait_for_close()

            task = asyncio.create_task(next_request())
            done, _ = await asyncio.wait({task}, timeout=1)
            if not done:
                first_response.release()
            await asyncio.wait_for(task, timeout=5)
            assert done, "the closed stream kept the only connection checked out"
            assert first_response.closed
        finally:
            finish.set()
            await handle.close()
