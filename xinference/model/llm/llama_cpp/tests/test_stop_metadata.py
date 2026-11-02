# CI: linux

import json

import pytest

from xinference.api.protocols import anthropic_stream_events, openai_to_anthropic
from xinference.model.llm.llama_cpp.core import XllamaCppModel, _annotate_stop_sequence


class _InlineExecutor:
    def submit(self, fn):
        fn()


class _FakeLlamaCppServer:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def handle_chat_completions(self, data, callback):
        self.requests.append(data.copy())
        for response in self.responses:
            callback(response)

    def handle_completions(self, data, callback):
        self.requests.append(data.copy())
        for response in self.responses:
            callback(response)


def _new_fake_llamacpp_model(responses):
    model = XllamaCppModel.__new__(XllamaCppModel)
    model.model_uid = "test-model"
    model.reasoning_parser = None
    model.tool_parser = None
    model._executor = _InlineExecutor()
    model._llm = _FakeLlamaCppServer(responses)
    return model


def test_annotate_stop_sequence_preserves_backend_marker():
    response = {"choices": [{"finish_reason": "stop", "stop": "HALT"}]}

    assert _annotate_stop_sequence(response, ["HALT"]) == "HALT"
    assert response["choices"][0]["stop_sequence"] == "HALT"


def test_annotate_stop_sequence_uses_matching_text_as_fallback():
    response = {"choices": [{"finish_reason": "stop", "text": "answer HALT"}]}

    assert _annotate_stop_sequence(response, ["HALT"]) == "HALT"
    assert response["choices"][0]["stop_sequence"] == "HALT"


def test_chat_maps_non_streaming_xllamacpp_verbose_stop_marker():
    response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 123,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "1, 2, 3, 4, "},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5},
        "__verbose": {"stopping_word": "5"},
    }
    model = _new_fake_llamacpp_model([response])

    openai_response = model.chat(
        [{"role": "user", "content": "count to 10"}], {"stop": ["5"]}
    )

    assert model._llm.requests[0]["verbose"] is True
    assert "__verbose" not in openai_response
    assert openai_response["choices"][0]["stop_sequence"] == "5"
    anthropic_response = openai_to_anthropic(
        openai_response, "test-model", stop_sequences=["5"]
    )
    assert anthropic_response["stop_reason"] == "stop_sequence"
    assert anthropic_response["stop_sequence"] == "5"


def test_generate_exposes_non_streaming_xllamacpp_verbose_stop_marker():
    response = {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 123,
        "model": "test-model",
        "choices": [{"text": "1, 2, 3, 4, ", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5},
        "__verbose": {"stopping_word": "5"},
    }
    model = _new_fake_llamacpp_model([response])

    completion = model.generate("count to 10", {"stop": ["5"]})

    assert model._llm.requests[0]["verbose"] is True
    assert "__verbose" not in completion
    assert completion["choices"][0]["stop_sequence"] == "5"


def test_generate_carries_stream_marker_to_empty_final_chunk():
    responses = [
        {
            "id": "cmpl-test",
            "object": "text_completion.chunk",
            "created": 123,
            "model": "test-model",
            "choices": [{"text": "1, 2, 3, 4, ", "finish_reason": None}],
            "__verbose": {"stopping_word": "5"},
        },
        {
            "id": "cmpl-test",
            "object": "text_completion.chunk",
            "created": 123,
            "model": "test-model",
            "choices": [{"text": "", "finish_reason": "stop"}],
        },
    ]
    model = _new_fake_llamacpp_model(responses)
    chunks = list(model.generate("count to 10", {"stream": True, "stop": ["5"]}))

    assert model._llm.requests[0]["verbose"] is True
    assert "__verbose" not in chunks[0]
    assert chunks[-1]["choices"][0]["stop_sequence"] == "5"


@pytest.mark.asyncio
async def test_chat_carries_stream_marker_to_empty_final_chunk():
    responses = [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "1, 2, 3, 4, "},
                    "finish_reason": None,
                }
            ],
            "__verbose": {"stopping_word": "5"},
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    model = _new_fake_llamacpp_model(responses)
    chunks = list(
        model.chat(
            [{"role": "user", "content": "count to 10"}],
            {"stream": True, "stop": ["5"]},
        )
    )

    assert model._llm.requests[0]["verbose"] is True
    assert "__verbose" not in chunks[0]
    assert chunks[-1]["choices"][0]["stop_sequence"] == "5"

    async def stream_chunks():
        for chunk in chunks:
            yield chunk

    events = [
        json.loads(event["data"])
        async for event in anthropic_stream_events(
            stream_chunks(), "test-model", "req-test", stop_sequences=["5"]
        )
    ]
    message_delta = next(event for event in events if event["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "stop_sequence"
    assert message_delta["delta"]["stop_sequence"] == "5"
