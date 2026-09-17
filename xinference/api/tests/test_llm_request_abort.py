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
async def test_chat_passes_request_id_outside_generation_config(monkeypatch):
    model = SimpleNamespace(
        uid=b"chat-model",
        chat=AsyncMock(return_value=b'{"choices": []}'),
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
