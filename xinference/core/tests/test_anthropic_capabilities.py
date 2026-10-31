# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import json
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from starlette.requests import Request

from ...api.protocols.anthropic_capabilities import (
    AnthropicBackendCapabilities,
    SupportLevel,
)
from ...model.llm.context_length import ContextLengthExceeded
from ..model import ModelActor
from ..pd_model import PDModelActor


def _model_actor():
    actor = ModelActor.__new__(ModelActor)
    actor._replica_model_uid = "model-1-0"
    actor._model_state = "ready"
    actor._lock = None
    actor._model = SimpleNamespace(
        get_anthropic_capabilities=lambda: AnthropicBackendCapabilities(
            engine="vllm",
            vision=SupportLevel.NATIVE,
        )
    )
    return actor


def _request(path, payload):
    body = json.dumps(payload).encode()
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "root_path": "",
            "headers": [(b"anthropic-version", b"2023-06-01")],
            "query_string": b"",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("127.0.0.1", 1234),
        },
        receive,
    )


def _anthropic_api_with_prepare_error(monkeypatch, error):
    from ...api.restful_api import RESTfulAPI

    monkeypatch.setenv("XINFERENCE_AUTH_ADVANCED", "false")
    api = RESTfulAPI("localhost", "localhost", 9997)
    supervisor = AsyncMock()
    supervisor.resolve_token_router_runtime = AsyncMock(return_value=None)
    model = AsyncMock()
    model.uid = "test-model"
    model.is_vllm_backend = AsyncMock(return_value=True)
    model.prepare_anthropic_request = AsyncMock(side_effect=error)
    model.chat = AsyncMock(return_value="{}")
    supervisor.get_model = AsyncMock(return_value=model)
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    return api, model


def _operation_error(kind, operation="anthropic_prepare"):
    return json.dumps(
        {
            "_xinference_error": {
                "version": 1,
                "operation": operation,
                "kind": kind,
                "message": "internal message must not be exposed",
            }
        }
    ).encode()


@pytest.mark.asyncio
async def test_model_actor_capabilities_are_json_serializable():
    payload = await _model_actor().get_anthropic_capabilities()

    description = json.loads(payload)
    assert description["engine"] == "vllm"
    assert description["vision"] == "native"
    assert description["supported_context_edits"] == ["clear_thinking_20251015"]


def test_model_actor_safe_error_drops_optional_engine_exception_object():
    error_type = type(
        "VLLMValidationError",
        (ValueError,),
        {"__module__": "vllm.exceptions"},
    )
    error = error_type(
        "The decoder prompt (length 5000) is longer than the maximum "
        "model length of 4096."
    )

    encoded = ModelActor._safe_model_operation_error(error, "multimodal_preflight")
    round_tripped = pickle.loads(pickle.dumps(encoded))
    payload = json.loads(round_tripped)

    assert payload == {
        "_xinference_error": {
            "version": 1,
            "operation": "multimodal_preflight",
            "engine_error_type": "VLLMValidationError",
            "kind": "context_length_exceeded",
            "message": "Request exceeds the model's configured context length.",
            "input_tokens": 5000,
            "max_model_len": 4096,
            "stage": "multimodal_process",
            "diagnostic_code": "VLLM_PROMPT_LENGTH_REJECTED",
        }
    }


def test_model_actor_safe_error_preserves_context_diagnostics():
    class RuntimeContextLength(int):
        pass

    error = ContextLengthExceeded(
        "Multimodal input uses 263175 tokens, leaving no output capacity "
        "within max_model_len=262144",
        input_tokens=RuntimeContextLength(263175),
        max_model_len=RuntimeContextLength(262144),
        stage="base_prompt_budget",
        diagnostic_code="BASE_PROMPT_EXHAUSTS_CONTEXT",
    )

    payload = json.loads(
        ModelActor._safe_model_operation_error(error, "multimodal_preflight")
    )

    assert payload["_xinference_error"] == {
        "version": 1,
        "operation": "multimodal_preflight",
        "engine_error_type": "ContextLengthExceeded",
        "kind": "context_length_exceeded",
        "message": "Request exceeds the model's configured context length.",
        "input_tokens": 263175,
        "max_model_len": 262144,
        "stage": "base_prompt_budget",
        "diagnostic_code": "BASE_PROMPT_EXHAUSTS_CONTEXT",
    }
    assert type(payload["_xinference_error"]["input_tokens"]) is int
    assert type(payload["_xinference_error"]["max_model_len"]) is int


def test_model_actor_safe_error_preserves_multimodal_limit_detail():
    encoded = ModelActor._safe_model_operation_error(
        ValueError("At most 2 image(s) may be provided in one prompt; received 3."),
        "multimodal_preflight",
    )

    assert json.loads(encoded)["_xinference_error"] == {
        "version": 1,
        "operation": "multimodal_preflight",
        "engine_error_type": "ValueError",
        "kind": "multimodal_limit_exceeded",
        "message": "At most 2 image(s) may be provided in one prompt; received 3.",
    }


@pytest.mark.parametrize(
    "operation", ["multimodal_preflight", "anthropic_prepare", "token_count"]
)
def test_model_actor_safe_error_classifies_damaged_image(operation):
    error_type = type(
        "UnidentifiedImageError",
        (OSError,),
        {"__module__": "PIL"},
    )
    error = error_type("cannot identify image file '/tmp/private-image.png'")

    payload = json.loads(ModelActor._safe_model_operation_error(error, operation))

    assert payload["_xinference_error"] == {
        "version": 1,
        "operation": operation,
        "engine_error_type": "UnidentifiedImageError",
        "kind": "invalid_multimodal_input",
        "message": "Invalid multimodal input or processing parameters.",
        "stage": "multimodal_process",
        "diagnostic_code": "VLLM_MM_PROCESS_REJECTED",
    }
    assert "private-image" not in json.dumps(payload)


def test_model_actor_safe_error_classifies_wrapped_damaged_image():
    error_type = type(
        "UnidentifiedImageError",
        (OSError,),
        {"__module__": "PIL"},
    )
    media_error = error_type("private media detail")
    try:
        raise media_error
    except OSError as exc:
        try:
            raise RuntimeError("processor wrapper detail") from exc
        except RuntimeError as wrapped:
            payload = json.loads(
                ModelActor._safe_model_operation_error(wrapped, "multimodal_preflight")
            )

    error = payload["_xinference_error"]
    assert error["kind"] == "invalid_multimodal_input"
    assert error["engine_error_type"] == "UnidentifiedImageError"
    assert error["diagnostic_code"] == "VLLM_MM_PROCESS_REJECTED"
    assert "private" not in json.dumps(payload)
    assert "wrapper" not in json.dumps(payload)


def test_model_actor_unknown_preflight_error_fails_closed():
    payload = json.loads(
        ModelActor._safe_model_operation_error(
            RuntimeError("private processor failure"), "multimodal_preflight"
        )
    )

    assert payload["_xinference_error"] == {
        "version": 1,
        "operation": "multimodal_preflight",
        "engine_error_type": "RuntimeError",
        "kind": "internal_error",
        "message": "The requested model operation failed.",
        "stage": "multimodal_process",
        "diagnostic_code": "MM_PREFLIGHT_INTERNAL_ERROR",
    }
    assert "private processor failure" not in json.dumps(payload)


def test_model_actor_safe_error_classifier_failure_stays_serializable():
    class BrokenError(Exception):
        def __str__(self):
            raise RuntimeError("broken string conversion")

    encoded = ModelActor._safe_model_operation_error(
        BrokenError(), "multimodal_preflight"
    )

    assert json.loads(encoded) == {
        "_xinference_error": {
            "version": 1,
            "operation": "multimodal_preflight",
            "kind": "internal_error",
            "message": "The requested model operation failed.",
        }
    }


@pytest.mark.asyncio
async def test_model_actor_anthropic_preparation_returns_safe_error_envelope():
    error_type = type(
        "VLLMValidationError",
        (Exception,),
        {"__module__": "vllm.exceptions"},
    )
    actor = _model_actor()
    actor._model.async_prepare_anthropic_request = AsyncMock()
    actor._model.prepare_anthropic_request = Mock()
    actor._call_wrapper_json = AsyncMock(
        side_effect=error_type("engine-specific validation failed")
    )
    actor._serve_count = 0
    actor._request_limits = 1
    actor._metrics_labels = {}
    actor.record_metrics = AsyncMock()

    result = await actor.prepare_anthropic_request([], {})
    payload = json.loads(result)

    assert payload["_xinference_error"]["kind"] == "invalid_multimodal_input"
    assert payload["_xinference_error"]["operation"] == "anthropic_prepare"
    assert payload["_xinference_error"]["stage"] == "multimodal_process"
    assert payload["_xinference_error"]["diagnostic_code"] == "VLLM_MM_PROCESS_REJECTED"
    assert "engine-specific validation failed" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_pd_model_forwards_serialized_decode_capabilities():
    payload = await _model_actor().get_anthropic_capabilities()
    decode = SimpleNamespace(get_anthropic_capabilities=AsyncMock(return_value=payload))
    actor = PDModelActor("pd")
    actor._decode_replicas = {"decode": decode}

    forwarded = await actor.get_anthropic_capabilities()

    assert json.loads(forwarded)["supported_context_edits"] == [
        "clear_thinking_20251015"
    ]
    decode.get_anthropic_capabilities.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_pd_multimodal_preflight_preserves_consistent_explicit_limit():
    successful_preflight = json.dumps(
        {
            "mode": "adaptive",
            "limits": None,
            "input_tokens": 10,
            "max_model_len": 100,
            "effective_max_tokens": 90,
        }
    ).encode()
    decode1 = SimpleNamespace(
        preflight_multimodal_request=AsyncMock(return_value=successful_preflight),
        get_explicit_limit_mm_per_prompt=AsyncMock(return_value={"image": 4}),
    )
    decode2 = SimpleNamespace(
        preflight_multimodal_request=AsyncMock(return_value=successful_preflight),
        get_explicit_limit_mm_per_prompt=AsyncMock(return_value={"image": 4}),
    )
    actor = PDModelActor("pd")
    actor._decode_replicas = {"decode-1": decode1, "decode-2": decode2}

    result = await actor.preflight_multimodal_request([], {})

    assert result["mode"] == "explicit_limit"
    assert result["limits"] == {"image": 4, "video": 2, "audio": 2}
    assert result["effective_max_tokens"] is None
    decode1.preflight_multimodal_request.assert_awaited_once_with([], {})
    decode2.preflight_multimodal_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_pd_multimodal_preflight_is_conservative_for_mixed_replicas():
    successful_preflight = json.dumps(
        {
            "mode": "adaptive",
            "limits": None,
            "input_tokens": 10,
            "max_model_len": 100,
            "effective_max_tokens": 90,
        }
    ).encode()
    explicit = SimpleNamespace(
        preflight_multimodal_request=AsyncMock(return_value=successful_preflight),
        get_explicit_limit_mm_per_prompt=AsyncMock(return_value={"image": 4}),
    )
    adaptive = SimpleNamespace(
        preflight_multimodal_request=AsyncMock(return_value=successful_preflight),
        get_explicit_limit_mm_per_prompt=AsyncMock(return_value=None),
    )
    actor = PDModelActor("pd")
    actor._decode_replicas = {"decode-1": explicit, "decode-2": adaptive}

    result = await actor.preflight_multimodal_request([], {})

    assert result["mode"] == "compat_fallback"
    assert result["limits"]["image"] == 2
    assert result["effective_max_tokens"] is None


@pytest.mark.asyncio
async def test_pd_multimodal_preflight_propagates_representative_error():
    operation_error = _operation_error(
        "invalid_multimodal_input", "multimodal_preflight"
    )
    decode = SimpleNamespace(
        preflight_multimodal_request=AsyncMock(return_value=operation_error),
        get_explicit_limit_mm_per_prompt=AsyncMock(return_value={"image": 4}),
    )
    actor = PDModelActor("pd")
    actor._decode_replicas = {"decode": decode}

    result = await actor.preflight_multimodal_request([], {})

    assert result is operation_error
    decode.get_explicit_limit_mm_per_prompt.assert_not_awaited()


@pytest.mark.asyncio
async def test_pd_multimodal_preflight_without_decode_replica_fails_closed():
    from ..model import ModelNotReadyError

    actor = PDModelActor("pd")
    actor._decode_replicas = {}

    with pytest.raises(ModelNotReadyError, match="No decode replicas available"):
        await actor.preflight_multimodal_request([], {})


@pytest.mark.asyncio
async def test_pd_multimodal_limit_rpc_failure_fails_closed():
    successful_preflight = json.dumps(
        {
            "mode": "adaptive",
            "limits": None,
            "input_tokens": 10,
            "max_model_len": 100,
            "effective_max_tokens": 90,
        }
    ).encode()
    decode = SimpleNamespace(
        preflight_multimodal_request=AsyncMock(return_value=successful_preflight),
        get_explicit_limit_mm_per_prompt=AsyncMock(
            side_effect=RuntimeError("private replica failure")
        ),
    )
    actor = PDModelActor("pd")
    actor._decode_replicas = {"decode": decode}

    with pytest.raises(RuntimeError, match="private replica failure"):
        await actor.preflight_multimodal_request([], {})


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        ContextLengthExceeded(
            "Multimodal input uses 5000 tokens, leaving no output capacity "
            "within max_model_len=4096"
        ),
        RuntimeError(
            "[address=worker:1234, pid=5678] The decoder prompt "
            "(length 5000) is longer than the maximum model length of 4096."
        ),
    ],
)
async def test_anthropic_messages_rejects_context_before_stream(
    monkeypatch, stream, error
):
    api, model = _anthropic_api_with_prepare_error(monkeypatch, error)
    response = await api.create_message(
        _request(
            "/v1/messages",
            {
                "model": "test-model",
                "max_tokens": 8,
                "stream": stream,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    )

    assert response.status_code == 400
    assert response.media_type == "application/json"
    payload = json.loads(response.body)
    assert payload["type"] == "error"
    assert payload["error"] == {
        "type": "invalid_request_error",
        "message": "Request exceeds the model's configured context length.",
    }
    assert "vllm" not in payload["error"]["message"]
    assert "address" not in payload["error"]["message"]
    model.chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ContextLengthExceeded(
            "Multimodal input uses 5000 tokens, leaving no output capacity "
            "within max_model_len=4096"
        ),
        RuntimeError(
            "[address=worker:1234, pid=5678] The encoder prompt "
            "(length 5000) is longer than the maximum model length of 4096."
        ),
    ],
)
async def test_anthropic_count_tokens_rejects_context(monkeypatch, error):
    api, model = _anthropic_api_with_prepare_error(monkeypatch, error)
    response = await api.count_anthropic_tokens(
        _request(
            "/v1/messages/count_tokens",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    )

    assert response.status_code == 400
    payload = json.loads(response.body)
    assert payload["error"] == {
        "type": "invalid_request_error",
        "message": "Request exceeds the model's configured context length.",
    }
    model.chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_anthropic_messages_rejects_structured_context_before_stream(
    monkeypatch, stream
):
    api, model = _anthropic_api_with_prepare_error(monkeypatch, RuntimeError("unused"))
    model.prepare_anthropic_request.side_effect = None
    model.prepare_anthropic_request.return_value = _operation_error(
        "context_length_exceeded"
    )

    response = await api.create_message(
        _request(
            "/v1/messages",
            {
                "model": "test-model",
                "max_tokens": 8,
                "stream": stream,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    )

    assert response.status_code == 400
    assert response.media_type == "application/json"
    payload = json.loads(response.body)
    assert payload["error"] == {
        "type": "invalid_request_error",
        "message": "Request exceeds the model's configured context length.",
    }
    assert "internal message" not in json.dumps(payload)
    model.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_anthropic_count_tokens_rejects_structured_context(monkeypatch):
    api, model = _anthropic_api_with_prepare_error(monkeypatch, RuntimeError("unused"))
    model.prepare_anthropic_request.side_effect = None
    model.prepare_anthropic_request.return_value = _operation_error(
        "context_length_exceeded", "token_count"
    )

    response = await api.count_anthropic_tokens(
        _request(
            "/v1/messages/count_tokens",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    )

    assert response.status_code == 400
    payload = json.loads(response.body)
    assert payload["error"] == {
        "type": "invalid_request_error",
        "message": "Request exceeds the model's configured context length.",
    }
    assert "internal message" not in json.dumps(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_anthropic_messages_rejects_structured_invalid_media_before_stream(
    monkeypatch, stream
):
    api, model = _anthropic_api_with_prepare_error(monkeypatch, RuntimeError("unused"))
    model.prepare_anthropic_request.side_effect = None
    model.prepare_anthropic_request.return_value = _operation_error(
        "invalid_multimodal_input"
    )

    response = await api.create_message(
        _request(
            "/v1/messages",
            {
                "model": "test-model",
                "max_tokens": 8,
                "stream": stream,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    )

    assert response.status_code == 400
    assert response.media_type == "application/json"
    payload = json.loads(response.body)
    assert payload["error"] == {
        "type": "invalid_request_error",
        "message": "Invalid multimodal input or processing parameters.",
    }
    assert "internal message" not in json.dumps(payload)
    model.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_anthropic_count_tokens_rejects_structured_invalid_media(monkeypatch):
    api, model = _anthropic_api_with_prepare_error(monkeypatch, RuntimeError("unused"))
    model.prepare_anthropic_request.side_effect = None
    model.prepare_anthropic_request.return_value = _operation_error(
        "invalid_multimodal_input", "token_count"
    )

    response = await api.count_anthropic_tokens(
        _request(
            "/v1/messages/count_tokens",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    )

    assert response.status_code == 400
    payload = json.loads(response.body)
    assert payload["error"] == {
        "type": "invalid_request_error",
        "message": "Invalid multimodal input or processing parameters.",
    }
    assert "internal message" not in json.dumps(payload)
    model.chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_anthropic_messages_preflight_unavailable_returns_503(
    monkeypatch, stream
):
    api, model = _anthropic_api_with_prepare_error(monkeypatch, RuntimeError("unused"))
    model.prepare_anthropic_request.side_effect = None
    model.prepare_anthropic_request.return_value = _operation_error(
        "preflight_unavailable"
    )

    response = await api.create_message(
        _request(
            "/v1/messages",
            {
                "model": "test-model",
                "max_tokens": 8,
                "stream": stream,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    )

    assert response.status_code == 503
    assert response.media_type == "application/json"
    payload = json.loads(response.body)
    assert payload["error"] == {
        "type": "api_error",
        "message": "Multimodal request preflight is temporarily unavailable.",
    }
    assert "internal message" not in json.dumps(payload)
    model.chat.assert_not_awaited()
