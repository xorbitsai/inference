# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from ..._compat import ValidationError
from ...request_priority import PRIORITY_ERROR_MESSAGES
from ...types import CreateChatCompletion, CreateCompletion
from .. import restful_api


@pytest.mark.parametrize("schema", [CreateChatCompletion, CreateCompletion])
@pytest.mark.parametrize("priority", [True, False, 1.5, "1", None, [], {}])
def test_priority_requires_an_integer(schema, priority):
    with pytest.raises(ValidationError, match="priority must be an integer"):
        schema.parse_obj(
            dict(model="model", prompt="hello", messages=[], priority=priority)
        )


@pytest.mark.parametrize("schema", [CreateChatCompletion, CreateCompletion])
@pytest.mark.parametrize("priority", [-10001, 10001])
def test_priority_range_is_validated_without_clamping(schema, priority):
    with pytest.raises(ValidationError, match="between -10000 and 10000"):
        schema.parse_obj(
            dict(model="model", prompt="hello", messages=[], priority=priority)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["chat", "completion"])
@pytest.mark.parametrize("priority", [True, 1.5, "1", None])
async def test_invalid_priority_returns_a_client_error(endpoint, priority):
    api = object.__new__(restful_api.RESTfulAPI)
    request = SimpleNamespace(
        json=AsyncMock(
            return_value=dict(
                model="model", prompt="hello", messages=[], priority=priority
            )
        )
    )
    handler = (
        api.create_chat_completion if endpoint == "chat" else api.create_completion
    )
    with pytest.raises(HTTPException) as caught:
        await handler(request)
    assert caught.value.status_code == 422
    assert caught.value.detail[0]["loc"] == ("priority",)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["chat", "completion"])
async def test_missing_model_returns_a_validation_error(endpoint):
    api = object.__new__(restful_api.RESTfulAPI)
    request = SimpleNamespace(
        json=AsyncMock(return_value={"prompt": "hello", "messages": []})
    )
    handler = (
        api.create_chat_completion if endpoint == "chat" else api.create_completion
    )
    with pytest.raises(HTTPException) as caught:
        await handler(request)
    assert caught.value.status_code == 422
    assert any(error["loc"] == ("model",) for error in caught.value.detail)


@pytest.mark.parametrize(
    "engine", ["vLLM", "SGLang", "Transformers", "MLX", "llama.cpp", "TRT-LLM"]
)
@pytest.mark.parametrize("endpoint", ["chat", "completion"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("priority", [-10, 0, 10, "omitted"])
@pytest.mark.asyncio
async def test_api_forwards_priority(monkeypatch, engine, endpoint, stream, priority):
    async def chunks():
        yield {"data": '{"choices": []}'}

    model = SimpleNamespace(
        uid=b"priority-model-0",
        chat=AsyncMock(return_value=chunks() if stream else b'{"choices": []}'),
        generate=AsyncMock(return_value=chunks() if stream else b'{"choices": []}'),
        is_vllm_backend=AsyncMock(return_value=engine == "vLLM"),
        is_transformers_batching_enabled=AsyncMock(return_value=False),
    )
    supervisor = SimpleNamespace(
        get_model=AsyncMock(return_value=model),
        describe_model=AsyncMock(
            return_value={"model_family": "test-family", "model_engine": engine}
        ),
    )
    api = object.__new__(restful_api.RESTfulAPI)
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    api._set_trace_model = api._set_trace_model_type = MagicMock()
    api._check_model_access = MagicMock()
    api._report_error_event = AsyncMock()
    api._get_model_last_error = AsyncMock(side_effect=lambda uid, error: error)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    body = dict(
        model="priority-model",
        messages=[{"role": "user", "content": "hello"}],
        prompt="hello",
        stream=stream,
        priority=priority,
    )
    if priority == "omitted":
        body.pop("priority")
    request = SimpleNamespace(
        json=AsyncMock(return_value=body), headers={}, state=SimpleNamespace()
    )
    handler = (
        api.create_chat_completion if endpoint == "chat" else api.create_completion
    )
    response = await handler(request)
    if stream:
        assert [item async for item in response.body_iterator]
    call = model.chat if endpoint == "chat" else model.generate
    if priority == "omitted":
        assert "priority" not in call.await_args.args[1]
    else:
        assert call.await_args.args[1]["priority"] == priority
    api._report_error_event.assert_not_awaited()


@pytest.mark.parametrize("message", PRIORITY_ERROR_MESSAGES)
@pytest.mark.parametrize("sanitize", [False, True])
def test_priority_configuration_errors_are_bad_requests(message, sanitize):
    error = ValueError(f"[address=internal-worker, pid=123] {message}")
    with pytest.raises(HTTPException) as caught:
        restful_api.RESTfulAPI.handle_request_limit_error(error, sanitize=sanitize)
    assert caught.value.status_code == 400
    assert caught.value.detail == message


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["chat", "completion"])
@pytest.mark.parametrize("message", PRIORITY_ERROR_MESSAGES)
async def test_streaming_priority_configuration_errors_use_sse(
    monkeypatch, endpoint, message
):
    model = SimpleNamespace(
        uid=b"priority-model-0",
        chat=AsyncMock(side_effect=ValueError(message)),
        generate=AsyncMock(side_effect=ValueError(message)),
        is_vllm_backend=AsyncMock(return_value=False),
        abort_request=AsyncMock(),
    )
    supervisor = SimpleNamespace(
        get_model=AsyncMock(return_value=model),
        describe_model=AsyncMock(return_value={"model_family": "test-family"}),
    )
    api = object.__new__(restful_api.RESTfulAPI)
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    api._set_trace_model = api._set_trace_model_type = MagicMock()
    api._check_model_access = MagicMock()
    api._report_error_event = AsyncMock()
    api._get_model_last_error = AsyncMock(side_effect=lambda uid, error: error)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    request = SimpleNamespace(
        json=AsyncMock(
            return_value=dict(
                model="priority-model",
                messages=[{"role": "user", "content": "hello"}],
                prompt="hello",
                stream=True,
                priority=-10,
            )
        ),
        headers={},
        state=SimpleNamespace(),
    )
    handler = (
        api.create_chat_completion if endpoint == "chat" else api.create_completion
    )
    response = await handler(request)
    assert response.status_code == 200
    chunks = [chunk async for chunk in response.body_iterator]
    assert len(chunks) == 1
    assert json.loads(chunks[0]["data"]) == {"error": message}
