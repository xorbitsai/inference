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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from .. import restful_api


class _Request:
    client = "test-client"

    def __init__(self, body):
        self._body = body
        self.state = SimpleNamespace()

    async def json(self):
        return self._body


def _new_api():
    api = object.__new__(restful_api.RESTfulAPI)
    api._set_trace_model = MagicMock()
    api._set_trace_model_type = MagicMock()
    api._check_model_access = MagicMock()
    api._report_error_event = AsyncMock()
    api.handle_request_limit_error = MagicMock()
    return api


async def _create_streaming_response(monkeypatch, kind, source):
    method = "generate" if kind == "completion" else "chat"
    model = SimpleNamespace(
        uid=f"{kind}-model".encode(),
        abort_request=AsyncMock(return_value="DONE"),
        decrease_serve_count=AsyncMock(),
        is_vllm_backend=AsyncMock(return_value=True),
        **{method: AsyncMock(return_value=source)},
    )

    async def require_model(*_args, **_kwargs):
        return model

    monkeypatch.setattr(restful_api, "require_model", require_model)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    api = _new_api()
    if kind == "completion":
        api._get_supervisor_ref = AsyncMock()
        body = {
            "model": "completion-model",
            "prompt": "hello",
            "stream": True,
            "request_id": "completion-request",
        }
        response = await api.create_completion(_Request(body))
    else:
        supervisor = SimpleNamespace(
            describe_model=AsyncMock(return_value={"model_family": "test-family"})
        )
        api._get_supervisor_ref = AsyncMock(return_value=supervisor)
        body = {
            "model": "chat-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "request_id": "chat-request",
        }
        response = await api.create_chat_completion(_Request(body))
    return response, model


async def _run_asgi_disconnect(response, phase, next_item_started):
    first_body_started = asyncio.Event()

    async def receive():
        await first_body_started.wait()
        if phase == "next_item":
            await next_item_started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_body_started.set()
            if phase == "send":
                await asyncio.Event().wait()

    await asyncio.wait_for(
        response({"type": "http", "asgi": {"version": "3.0"}}, receive, send),
        timeout=5,
    )


def _tracked_stream(*, wait_after_first):
    closed = asyncio.Event()
    next_item_started = asyncio.Event()

    async def generate():
        try:
            yield b'{"choices": []}'
            if wait_after_first:
                next_item_started.set()
                await asyncio.Event().wait()
        finally:
            closed.set()

    return generate(), closed, next_item_started


@pytest.mark.asyncio
async def test_completion_disconnect_aborts_propagated_request_id(monkeypatch):
    async def disconnected_stream():
        yield b'{"choices": []}'
        raise asyncio.CancelledError

    model = SimpleNamespace(
        uid=b"completion-model",
        generate=AsyncMock(return_value=disconnected_stream()),
        abort_request=AsyncMock(return_value="DONE"),
        decrease_serve_count=AsyncMock(),
        is_vllm_backend=AsyncMock(return_value=True),
    )

    async def require_model(*_args, **_kwargs):
        return model

    monkeypatch.setattr(restful_api, "require_model", require_model)
    api = _new_api()
    api._get_supervisor_ref = AsyncMock()
    response = await api.create_completion(
        _Request(
            {
                "model": "completion-model",
                "prompt": "hello",
                "stream": True,
                "request_id": "completion-request",
            }
        )
    )

    assert await anext(response.body_iterator) == b'{"choices": []}'
    with pytest.raises(StopAsyncIteration):
        await anext(response.body_iterator)

    model.generate.assert_awaited_once()
    call = model.generate.await_args
    assert call.kwargs["request_id"] == "completion-request"
    assert "request_id" not in call.args[1]
    assert "request_id" not in call.kwargs["raw_params"]
    model.abort_request.assert_awaited_once_with("completion-request")
    model.decrease_serve_count.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
@pytest.mark.parametrize("phase", ["send", "next_item"])
async def test_real_sse_disconnect_aborts_and_cleans_stream(monkeypatch, kind, phase):
    source, closed, next_item_started = _tracked_stream(wait_after_first=True)
    response, model = await _create_streaming_response(monkeypatch, kind, source)

    await _run_asgi_disconnect(response, phase, next_item_started)

    model.abort_request.assert_awaited_once_with(f"{kind}-request")
    model.decrease_serve_count.assert_awaited_once()
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
async def test_explicit_stream_close_aborts_and_cleans_stream(monkeypatch, kind):
    source, closed, _ = _tracked_stream(wait_after_first=True)
    response, model = await _create_streaming_response(monkeypatch, kind, source)

    assert await anext(response.body_iterator) == b'{"choices": []}'
    await response.body_iterator.aclose()

    model.abort_request.assert_awaited_once_with(f"{kind}-request")
    model.decrease_serve_count.assert_awaited_once()
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
async def test_normal_stream_completion_does_not_abort(monkeypatch, kind):
    source, closed, _ = _tracked_stream(wait_after_first=False)
    response, model = await _create_streaming_response(monkeypatch, kind, source)

    chunks = [chunk async for chunk in response.body_iterator]

    assert chunks[0] == b'{"choices": []}'
    if kind == "chat":
        assert chunks[-1] == "[DONE]"
    model.abort_request.assert_not_awaited()
    model.decrease_serve_count.assert_awaited_once()
    assert closed.is_set()


@pytest.mark.asyncio
async def test_chat_passes_request_id_outside_generation_config(monkeypatch):
    model = SimpleNamespace(
        uid=b"chat-model",
        chat=AsyncMock(return_value=b'{"choices": []}'),
        is_vllm_backend=AsyncMock(return_value=True),
    )
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_family": "test-family"})
    )

    async def require_model(*_args, **_kwargs):
        return model

    monkeypatch.setattr(restful_api, "require_model", require_model)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    api = _new_api()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    response = await api.create_chat_completion(
        _Request(
            {
                "model": "chat-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "request_id": "chat-request",
            }
        )
    )

    assert response.body == b'{"choices": []}'
    model.chat.assert_awaited_once()
    call = model.chat.await_args
    assert call.kwargs["request_id"] == "chat-request"
    assert "request_id" not in call.args[1]
    assert "request_id" not in call.kwargs["raw_params"]


@pytest.mark.asyncio
async def test_non_vllm_chat_keeps_request_id_in_generation_config(monkeypatch):
    model = SimpleNamespace(
        uid=b"chat-model",
        chat=AsyncMock(return_value=b'{"choices": []}'),
        is_vllm_backend=AsyncMock(return_value=False),
    )
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_family": "test-family"})
    )

    async def require_model(*_args, **_kwargs):
        return model

    monkeypatch.setattr(restful_api, "require_model", require_model)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    api = _new_api()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    response = await api.create_chat_completion(
        _Request(
            {
                "model": "chat-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "request_id": "chat-request",
            }
        )
    )

    assert response.body == b'{"choices": []}'
    call = model.chat.await_args
    assert call.args[1]["request_id"] == "chat-request"
    assert "request_id" not in call.kwargs
