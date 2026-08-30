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

import pytest

from ..core import chat_context_var
from ..reasoning_parser import ReasoningParser
from ..tool_parsers.llama3_tool_parser import Llama3ToolParser
from ..tool_parsers.qwen_tool_parser import QwenToolParser
from ..utils import ChatModelMixin


@pytest.fixture(autouse=True)
def reset_chat_context_var():
    token = chat_context_var.set({})
    try:
        yield
    finally:
        chat_context_var.reset(token)


def test_is_valid_model_name():
    from ...utils import is_valid_model_name

    assert is_valid_model_name("foo")
    assert is_valid_model_name("foo-bar")
    assert is_valid_model_name("foo_bar")
    assert is_valid_model_name("123")
    assert is_valid_model_name("foo@bar")
    assert is_valid_model_name("_foo")
    assert is_valid_model_name("-foo")
    assert not is_valid_model_name("foo bar")
    assert not is_valid_model_name("foo/bar")
    assert not is_valid_model_name("   ")
    assert not is_valid_model_name("")


def filter_ids_and_created(data):
    if isinstance(data, list):
        return [filter_ids_and_created(item) for item in data]
    elif isinstance(data, dict):
        return {
            k: filter_ids_and_created(v)
            for k, v in data.items()
            if k not in ["id", "created"]
        }
    return data


def test_to_chat_completion_chunks_usage_only_chunk_without_metadata():
    chunks = [
        {
            "id": "cmpl-test",
            "object": "text_completion",
            "created": 123,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "text": "hello",
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        },
        {
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            }
        },
    ]

    results = list(ChatModelMixin._to_chat_completion_chunks(iter(chunks)))

    assert results[-1] == {
        "id": "chatcmpl-test",
        "model": "test-model",
        "created": 123,
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
    }


def test_to_chat_completion_propagates_logprobs():
    # Regression for #1911 / #3553: /v1/chat/completions silently returned
    # logprobs: null even when the engine produced them. The chat builder must
    # propagate the source Completion choice's logprobs onto the chat choice.
    from ....types import (
        Completion,
        CompletionChoice,
        CompletionLogprobs,
        CompletionUsage,
    )

    logprobs = CompletionLogprobs(
        text_offset=[0, 5],
        token_logprobs=[None, -0.1],
        tokens=["Hello", " world"],
        top_logprobs=[None, {" world": -0.2}],
    )
    completion = Completion(
        id="cmpl-1",
        object="text_completion",
        created=123,
        model="test-model",
        choices=[
            CompletionChoice(
                text="Hello world",
                index=0,
                logprobs=logprobs,
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    chat = ChatModelMixin._to_chat_completion(completion)

    assert chat["choices"][0]["logprobs"] is not None
    # Chat completions carry the chat ``content[]`` logprobs shape (token/bytes/
    # logprob/top_logprobs), not the legacy parallel-list shape, so openai-python
    # parses ``choice.logprobs.content`` instead of dropping it as extra fields.
    assert chat["choices"][0]["logprobs"] == {
        # The first token's legacy logprob is None (not computed), so it is
        # omitted from content[] rather than emitted with a null logprob —
        # openai-python rejects null (see test_chat_logprobs_parse_openai).
        "content": [
            {
                "token": " world",
                "bytes": [32, 119, 111, 114, 108, 100],
                "logprob": -0.1,
                "top_logprobs": [
                    {
                        "token": " world",
                        "bytes": [32, 119, 111, 114, 108, 100],
                        "logprob": -0.2,
                    }
                ],
            },
        ]
    }


def test_to_chat_completion_chunks_propagates_logprobs():
    # Streaming counterpart: the chat chunk choice must carry the source
    # CompletionChunk choice's logprobs instead of dropping them.
    from ....types import CompletionChoice, CompletionChunk, CompletionLogprobs

    logprobs = CompletionLogprobs(
        text_offset=[0, 5],
        token_logprobs=[None, -0.1],
        tokens=["Hello", " world"],
        top_logprobs=[None, {" world": -0.2}],
    )
    chunk = CompletionChunk(
        id="cmpl-2",
        object="text_completion",
        created=123,
        model="test-model",
        choices=[
            CompletionChoice(
                text="Hello", index=0, logprobs=logprobs, finish_reason=None
            )
        ],
    )

    results = list(ChatModelMixin._to_chat_completion_chunks(iter([chunk])))

    assert results, "expected at least one chat chunk"
    assert results[0]["choices"][0]["logprobs"] is not None
    # Streaming chat chunks carry the chat ``content[]`` logprobs shape, converted
    # from the legacy parallel-list shape the engine emits.
    assert results[0]["choices"][0]["logprobs"] == {
        # The first token's legacy logprob is None -> omitted from content[]
        # (see test_chat_logprobs_parse_openai); only the known-logprob token
        # survives, in the chat content[] shape.
        "content": [
            {
                "token": " world",
                "bytes": [32, 119, 111, 114, 108, 100],
                "logprob": -0.1,
                "top_logprobs": [
                    {
                        "token": " world",
                        "bytes": [32, 119, 111, 114, 108, 100],
                        "logprob": -0.2,
                    }
                ],
            },
        ]
    }


def test_chat_logprobs_parse_openai():
    # Regression for the openai-python validation failure qinxuye reproduced with
    # 1.99.9: ``ChatCompletionTokenLogprob.logprob`` is non-nullable, so a chat
    # logprobs ``content[]`` entry must never carry ``null``. The converter now
    # skips tokens whose legacy ``token_logprobs`` entry is ``None`` instead of
    # emitting a null logprob; assert every emitted entry parses through the
    # OpenAI client model.
    pytest.importorskip("openai")
    from openai.types.chat.chat_completion_token_logprob import (
        ChatCompletionTokenLogprob,
    )

    from ....types import (
        Completion,
        CompletionChoice,
        CompletionLogprobs,
        CompletionUsage,
    )

    # token_logprobs[0] is None (the legacy first-token case). The converter must
    # drop that token rather than emit ``logprob: None``.
    logprobs = CompletionLogprobs(
        text_offset=[0, 5],
        token_logprobs=[None, -0.1],
        tokens=["Hello", " world"],
        top_logprobs=[None, {" world": -0.2}],
    )
    completion = Completion(
        id="cmpl-1",
        object="text_completion",
        created=123,
        model="test-model",
        choices=[
            CompletionChoice(
                text="Hello world",
                index=0,
                logprobs=logprobs,
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    chat = ChatModelMixin._to_chat_completion(completion)
    content = chat["choices"][0]["logprobs"]["content"]
    assert content, "content[] must not be empty when a known-logprob token exists"
    # No entry carries a null logprob, and the None-logprob first token is dropped.
    assert [e["token"] for e in content] == [" world"]
    for entry in content:
        # Must parse through the OpenAI client model without raising — the exact
        # gate that was failing ("logprob: Input should be a valid number").
        ChatCompletionTokenLogprob(**entry)


class _NoOpToolParser:
    """Minimal tool-parser stub: reports no tool calls so the tool post-processors
    take the content-preserving branch while still exercising the logprobs path
    (the real parsers are heavy to construct in a unit test)."""

    def extract_tool_calls(self, text):  # non-streaming
        return []

    def extract_tool_calls_streaming(
        self, previous_texts, current_text, delta_text
    ):  # streaming; returns (content, func, args)
        return (delta_text or "", None, None)


class _MultiToolParser:
    def extract_tool_calls_streaming(self, previous_texts, current_text, delta_text):
        return [
            (None, "get_weather", {"city": "Beijing"}),
            (None, "get_time", {"timezone": "UTC+8"}),
            (" tail", None, None),
        ]


class _IndexedToolParser:
    def extract_tool_calls_streaming(self, previous_texts, current_text, delta_text):
        if not delta_text:
            return None
        if delta_text == " gap":
            return (delta_text, None, None)
        if current_text == "first":
            return (None, "first", {}, 0)
        return (None, "second", {}, 1)


class _IncrementalToolParser:
    def extract_tool_calls_streaming(self, previous_texts, current_text, delta_text):
        if current_text == "start":
            return (None, "get_weather", None, 0)
        return (None, "get_weather", {"city": "Beijing"}, 0)


class _InterleavedIncrementalToolParser:
    def extract_tool_calls_streaming(self, previous_texts, current_text, delta_text):
        events = {
            "call0-start": (None, "get_weather", None, 0),
            "call1-start": (None, "get_time", None, 1),
            "call0-complete": (None, "get_weather", {"city": "Beijing"}, 0),
            "call1-complete": (None, "get_time", {"timezone": "UTC+8"}, 1),
        }
        return events[delta_text]


def test_post_process_completion_chunk_supports_multiple_tool_calls():
    mixin = ChatModelMixin()
    mixin.tool_parser = _MultiToolParser()
    result = mixin._post_process_completion_chunk(
        "test-family",
        "test-model",
        {
            "choices": [
                {
                    "delta": {"content": "tool output"},
                    "finish_reason": None,
                    "logprobs": None,
                }
            ]
        },
        previous_texts=[""],
    )

    assert result is not None
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["delta"]["content"] == " tail"
    tool_calls = choice["delta"]["tool_calls"]
    assert [call["index"] for call in tool_calls] == [0, 1]
    assert all(call["id"].startswith("call_") for call in tool_calls)
    assert all(call["type"] == "function" for call in tool_calls)
    assert [
        (call["function"]["name"], call["function"]["arguments"]) for call in tool_calls
    ] == [
        ("get_weather", '{"city": "Beijing"}'),
        ("get_time", '{"timezone": "UTC+8"}'),
    ]


def test_post_process_completion_chunk_preserves_absolute_tool_call_index():
    mixin = ChatModelMixin()
    mixin.tool_parser = _IndexedToolParser()
    previous_texts = [""]
    tool_call_state = {"seen": False}

    first = mixin._post_process_completion_chunk(
        "test-family",
        "test-model",
        {
            "choices": [
                {"delta": {"content": "first"}, "finish_reason": None, "logprobs": None}
            ]
        },
        previous_texts=previous_texts,
        tool_call_state=tool_call_state,
    )
    gap = mixin._post_process_completion_chunk(
        "test-family",
        "test-model",
        {
            "choices": [
                {"delta": {"content": " gap"}, "finish_reason": None, "logprobs": None}
            ]
        },
        previous_texts=previous_texts,
        tool_call_state=tool_call_state,
    )
    second = mixin._post_process_completion_chunk(
        "test-family",
        "test-model",
        {
            "choices": [
                {
                    "delta": {"content": " second"},
                    "finish_reason": None,
                    "logprobs": None,
                }
            ]
        },
        previous_texts=previous_texts,
        tool_call_state=tool_call_state,
    )
    final = mixin._post_process_completion_chunk(
        "test-family",
        "test-model",
        {
            "choices": [
                {"delta": {"content": ""}, "finish_reason": "stop", "logprobs": None}
            ]
        },
        previous_texts=previous_texts,
        tool_call_state=tool_call_state,
    )

    assert first is not None
    assert gap is not None
    assert second is not None
    assert final is not None
    assert first["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert gap["choices"][0]["delta"]["tool_calls"] == []
    assert gap["choices"][0]["delta"]["content"] == " gap"
    assert second["choices"][0]["delta"]["tool_calls"][0]["index"] == 1
    assert first["choices"][0]["finish_reason"] is None
    assert gap["choices"][0]["finish_reason"] is None
    assert second["choices"][0]["finish_reason"] is None
    assert final["choices"][0]["finish_reason"] == "tool_calls"


def test_post_process_completion_chunk_emits_incremental_metadata_once():
    mixin = ChatModelMixin()
    mixin.tool_parser = _IncrementalToolParser()
    previous_texts = [""]
    tool_call_state = {"seen": False}

    first = mixin._post_process_completion_chunk(
        "test-family",
        "test-model",
        {
            "choices": [
                {"delta": {"content": "start"}, "finish_reason": None, "logprobs": None}
            ]
        },
        previous_texts=previous_texts,
        tool_call_state=tool_call_state,
    )
    complete = mixin._post_process_completion_chunk(
        "test-family",
        "test-model",
        {
            "choices": [
                {
                    "delta": {"content": " complete"},
                    "finish_reason": None,
                    "logprobs": None,
                }
            ]
        },
        previous_texts=previous_texts,
        tool_call_state=tool_call_state,
    )

    assert first is not None
    assert complete is not None
    first_call = first["choices"][0]["delta"]["tool_calls"][0]
    complete_call = complete["choices"][0]["delta"]["tool_calls"][0]
    assert first_call["id"] == tool_call_state["call_ids"][0]
    assert first_call["type"] == "function"
    assert first_call["function"] == {"name": "get_weather", "arguments": ""}
    assert "id" not in complete_call
    assert "type" not in complete_call
    assert complete_call["function"] == {"arguments": '{"city": "Beijing"}'}
    assert tool_call_state["sent_metadata"] == {0}


def test_post_process_completion_chunk_tracks_metadata_per_tool_call_index():
    mixin = ChatModelMixin()
    mixin.tool_parser = _InterleavedIncrementalToolParser()
    previous_texts = [""]
    tool_call_state = {"seen": False}

    def process(delta_text):
        result = mixin._post_process_completion_chunk(
            "test-family",
            "test-model",
            {
                "choices": [
                    {
                        "delta": {"content": delta_text},
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ]
            },
            previous_texts=previous_texts,
            tool_call_state=tool_call_state,
        )
        assert result is not None
        return result["choices"][0]["delta"]["tool_calls"][0]

    first_call0 = process("call0-start")
    first_call1 = process("call1-start")
    complete_call0 = process("call0-complete")
    complete_call1 = process("call1-complete")

    assert first_call0["index"] == 0
    assert first_call1["index"] == 1
    assert first_call0["id"] != first_call1["id"]
    assert first_call0["type"] == first_call1["type"] == "function"
    assert first_call0["function"] == {"name": "get_weather", "arguments": ""}
    assert first_call1["function"] == {"name": "get_time", "arguments": ""}

    assert complete_call0 == {
        "index": 0,
        "function": {"arguments": '{"city": "Beijing"}'},
    }
    assert complete_call1 == {
        "index": 1,
        "function": {"arguments": '{"timezone": "UTC+8"}'},
    }
    assert tool_call_state["call_ids"] == {
        0: first_call0["id"],
        1: first_call1["id"],
    }
    assert tool_call_state["sent_names"] == {0, 1}
    assert tool_call_state["sent_metadata"] == {0, 1}


def test_post_process_completion_chunk_preserves_length_finish_reason():
    mixin = ChatModelMixin()
    mixin.tool_parser = _IndexedToolParser()

    result = mixin._post_process_completion_chunk(
        "test-family",
        "test-model",
        {
            "choices": [
                {
                    "delta": {"content": ""},
                    "finish_reason": "length",
                    "logprobs": None,
                }
            ]
        },
        previous_texts=["first"],
        tool_call_state={"seen": True},
    )

    assert result is not None
    assert result["choices"][0]["finish_reason"] == "length"


def test_post_process_completion_preserves_chat_logprobs():
    # Non-streaming tools path: _post_process_completion previously omitted the
    # logprobs field entirely, so tool-enabled requests lost them. It must now
    # carry the converted chat-shape logprobs from the source Completion choice.
    from ....types import (
        Completion,
        CompletionChoice,
        CompletionLogprobs,
        CompletionUsage,
    )

    logprobs = CompletionLogprobs(
        text_offset=[0, 5],
        token_logprobs=[None, -0.1],
        tokens=["Hello", " world"],
        top_logprobs=[None, {" world": -0.2}],
    )
    completion = Completion(
        id="cmpl-1",
        object="text_completion",
        created=123,
        model="test-model",
        choices=[
            CompletionChoice(
                text="Hello world",
                index=0,
                logprobs=logprobs,
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    mixin = ChatModelMixin()
    mixin.tool_parser = _NoOpToolParser()
    mixin.reasoning_parser = None

    result = mixin._post_process_completion("test-family", "test-model", completion)

    assert result["choices"][0]["logprobs"] is not None
    assert result["choices"][0]["logprobs"] == {
        "content": [
            {
                "token": " world",
                "bytes": [32, 119, 111, 114, 108, 100],
                "logprob": -0.1,
                "top_logprobs": [
                    {
                        "token": " world",
                        "bytes": [32, 119, 111, 114, 108, 100],
                        "logprob": -0.2,
                    }
                ],
            },
        ]
    }


def test_post_process_completion_chunk_preserves_chat_logprobs():
    # Streaming tools path: `_async_to_tool_completion_chunks` first converts each
    # legacy CompletionChunk to a chat chunk via `_to_chat_completion_chunk` (which
    # already turns the legacy parallel-list logprobs into the chat ``content[]``
    # shape), then hands the chat chunk to `_post_process_completion_chunk`. That
    # post-processor must pass the already-converted logprobs through unchanged;
    # re-running the converter would find no ``tokens`` key and collapse a real
    # one-token logprob to ``{"content": []}`` (regression flagged on #5252).
    from ....types import CompletionChoice, CompletionChunk, CompletionLogprobs

    logprobs = CompletionLogprobs(
        text_offset=[0, 5],
        token_logprobs=[None, -0.1],
        tokens=["Hello", " world"],
        top_logprobs=[None, {" world": -0.2}],
    )
    raw_chunk = CompletionChunk(
        id="cmpl-2",
        object="text_completion",
        created=123,
        model="test-model",
        choices=[
            CompletionChoice(
                text="Hello",
                index=0,
                logprobs=logprobs,
                finish_reason=None,
            )
        ],
    )

    # Production order: convert to a chat chunk first, then post-process it.
    chat_chunk = ChatModelMixin._to_chat_completion_chunk(raw_chunk)
    expected_logprobs = chat_chunk["choices"][0]["logprobs"]
    assert expected_logprobs is not None  # converted once to the chat shape

    mixin = ChatModelMixin()
    mixin.tool_parser = _NoOpToolParser()
    mixin.reasoning_parser = None

    result = mixin._post_process_completion_chunk(
        "test-family", "test-model", chat_chunk, chunk_id="cmpl-2"
    )

    assert result is not None
    # Passed through unchanged -- not double-converted to {"content": []}.
    assert result["choices"][0]["logprobs"] == expected_logprobs
    assert result["choices"][0]["logprobs"] == {
        "content": [
            {
                "token": " world",
                "bytes": [32, 119, 111, 114, 108, 100],
                "logprob": -0.1,
                "top_logprobs": [
                    {
                        "token": " world",
                        "bytes": [32, 119, 111, 114, 108, 100],
                        "logprob": -0.2,
                    }
                ],
            },
        ]
    }


def test_async_to_chat_completion_chunks_preserves_usage_only_chunk():
    async def _chunks():
        yield {
            "id": "cmpl-test",
            "object": "text_completion",
            "created": 123,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
        }
        yield {
            "choices": [],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }

    async def _collect():
        return [
            chunk
            async for chunk in ChatModelMixin._async_to_chat_completion_chunks(
                _chunks()
            )
        ]

    results = asyncio.run(_collect())

    assert results[-1] == {
        "id": "chatcmpl-test",
        "model": "test-model",
        "created": 123,
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
    }


def test_transform_messages_preserves_tool_call_fields():
    mixin = ChatModelMixin()
    messages = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_bed4c5f1",
                    "function": {
                        "arguments": '{"file_path": "README*"}',
                        "name": "view_file_in_detail",
                    },
                    "type": "function",
                }
            ],
        },
        {
            "role": "tool",
            "content": "Tool execute results: file not found: README*",
            "tool_call_id": "call_bed4c5f1",
        },
    ]

    transformed = mixin._transform_messages(messages)

    assert transformed[0] == {
        "role": "user",
        "content": [{"type": "text", "text": "Hi"}],
    }
    assert transformed[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_bed4c5f1",
                "function": {
                    "arguments": {"file_path": "README*"},
                    "name": "view_file_in_detail",
                },
                "type": "function",
            }
        ],
    }
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == (
        '{"file_path": "README*"}'
    )
    assert transformed[2] == {
        "role": "tool",
        "content": [
            {"type": "text", "text": "Tool execute results: file not found: README*"}
        ],
        "tool_call_id": "call_bed4c5f1",
    }


def test_transform_messages_materializes_tool_call_iterators():
    mixin = ChatModelMixin()
    messages = [
        {
            "role": "assistant",
            "tool_calls": iter(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": iter([("path", "README.md")]),
                        },
                    }
                ]
            ),
        }
    ]

    transformed = mixin._transform_messages(messages)

    assert transformed == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": {"path": "README.md"},
                    },
                }
            ],
        }
    ]


def test_transform_messages_rejects_invalid_tool_call_arguments_json():
    mixin = ChatModelMixin()
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"path":',
                    },
                }
            ],
        }
    ]

    with pytest.raises(
        ValueError, match="Tool call arguments must be a valid JSON object"
    ):
        mixin._transform_messages(messages)


def test_deepseekv4_get_full_context_attaches_tools(tmp_path):
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text(
        "def encode_messages(messages, thinking_mode):\n    return messages\n",
        encoding="utf-8",
    )
    mixin = ChatModelMixin()
    mixin.model_family = SimpleNamespace(
        model_name="DeepSeek-V4-Flash",
        model_ability=["chat", "hybrid", "tools"],
        is_builtin=True,
    )
    mixin.model_path = str(tmp_path)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    messages = mixin.get_full_context(
        [{"role": "user", "content": "How is the weather?"}],
        "",
        tokenizer=object(),
        tools=tools,
    )

    assert messages[0] == {"role": "system", "content": "", "tools": tools}
    assert messages[1] == {"role": "user", "content": "How is the weather?"}


def test_deepseekv4_get_full_context_blocks_custom_remote_code(tmp_path):
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text(
        "def encode_messages(messages, thinking_mode):\n    return messages\n",
        encoding="utf-8",
    )
    mixin = ChatModelMixin()
    mixin.model_family = SimpleNamespace(
        model_name="DeepSeek-V4-Flash",
        model_ability=["chat", "hybrid", "tools"],
    )
    mixin.model_path = str(tmp_path)

    with pytest.raises(ValueError, match="XINFERENCE_TRUST_REMOTE_CODE=1"):
        mixin.get_full_context(
            [{"role": "user", "content": "How is the weather?"}],
            "",
            tokenizer=object(),
        )


def test_chat_template_kwargs_inherit_model_thinking_default():
    reasoning_parser = ReasoningParser(
        reasoning_content=False,
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        enable_thinking=False,
    )

    kwargs = ChatModelMixin._get_chat_template_kwargs_from_generate_config(
        {"chat_template_kwargs": {"add_vision_id": True}},
        reasoning_parser,
    )

    assert kwargs == {"add_vision_id": True, "enable_thinking": False}


def test_chat_template_kwargs_keep_request_thinking_override():
    reasoning_parser = ReasoningParser(
        reasoning_content=False,
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        enable_thinking=False,
    )

    kwargs = ChatModelMixin._get_chat_template_kwargs_from_generate_config(
        {"chat_template_kwargs": {"thinking": True}},
        reasoning_parser,
    )

    assert kwargs == {"thinking": True, "enable_thinking": True}


def test_chat_template_kwargs_string_normalizes_thinking():
    reasoning_parser = ReasoningParser(
        reasoning_content=False,
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        enable_thinking=False,
    )

    kwargs = ChatModelMixin._get_chat_template_kwargs_from_generate_config(
        {"chat_template_kwargs": '{"thinking": true}'},
        reasoning_parser,
    )

    assert kwargs == {"thinking": True, "enable_thinking": True}


def test_post_process_completion_chunk_without_thinking():
    mixin = ChatModelMixin()
    mixin.tool_parser = QwenToolParser()

    test_cases = [
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "<tool_call>"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": "\n"}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": '{"'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "name"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": '":'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": ' "'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": "get"}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "_current"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "_weather"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": '",'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": ' "'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "arguments"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": '":'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": ' {"'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "location"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": '":'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": ' "'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "\u4e0a\u6d77"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": '"}}\n'},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "</tool_call>"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
        ),
    ]
    expected_results = [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {
            "id": "chatcmpl-9a5150cb-5475-4a1b-9535-61599de39ad8",
            "model": "qwen",
            "object": "chat.completion.chunk",
            "created": 1756546331,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_9a5150cb-5475-4a1b-9535-61599de39ad8",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location": "上海"}',
                                },
                            }
                        ],
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": None,
        },
        None,
        {
            "id": "chatcmpl-99f6e3ba-bbc4-4a6d-93ad-a86e2f7705d6",
            "model": "qwen",
            "object": "chat.completion.chunk",
            "created": 1756646416,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "", "tool_calls": []},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 159,
                "completion_tokens": 91,
                "total_tokens": 250,
            },
        },
    ]
    previous_texts = [""]

    for i, (model_uid, chunk_data) in enumerate(test_cases):
        result = mixin._post_process_completion_chunk(
            None, model_uid=model_uid, c=chunk_data, previous_texts=previous_texts
        )

        expected = expected_results[i]

        if expected is None:
            assert result is None, f"Expected None but got {result}"
        else:
            result_filtered = filter_ids_and_created(result)
            expected_filtered = filter_ids_and_created(expected)
            assert (
                result_filtered == expected_filtered
            ), f"Mismatch at case {i}: expected {expected_filtered}, got {result_filtered}"


def test_post_process_completion_chunk_with_thinking():
    mixin = ChatModelMixin()
    mixin.tool_parser = QwenToolParser()
    test_cases = [
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "<think>"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": "\n"}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "\u597d\u7684"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "</think>"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "\n\n"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "<tool_call>"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": "\n"}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": '{"'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "name"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": '":'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": ' "'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": "get"}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "_current"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "_weather"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": '",'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": ' "'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "arguments"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": '":'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": ' {"'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "location"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": '":'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": ' "'}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "\u4e0a\u6d77"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": '"}}\n'},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "</tool_call>"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": None}
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
        ),
    ]
    expected_results = [
        {
            "id": "chatcmpl-7a377eb3-76a6-4503-b86f-1f8d4945bd76",
            "model": "qwen",
            "object": "chat.completion.chunk",
            "created": 1756546331,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "<think>",
                        "tool_calls": [],
                    },
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
            "usage": None,
        },
        {
            "id": "chatcmpl-83dc7d9a-3672-4d2b-a053-cd42a187cef3",
            "model": "qwen",
            "object": "chat.completion.chunk",
            "created": 1756546331,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "\n",
                        "tool_calls": [],
                    },
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
            "usage": None,
        },
        {
            "id": "chatcmpl-2755f3b2-c020-4127-aa12-9998dfcda26e",
            "model": "qwen",
            "object": "chat.completion.chunk",
            "created": 1756546331,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "好的",
                        "tool_calls": [],
                    },
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
            "usage": None,
        },
        {
            "id": "chatcmpl-4850ba39-7a0f-4d84-89c6-f1344d229616",
            "model": "qwen",
            "object": "chat.completion.chunk",
            "created": 1756546331,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "</think>",
                        "tool_calls": [],
                    },
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
            "usage": None,
        },
        {
            "id": "chatcmpl-2f2a2673-2242-442e-9a70-9efc83e7a2e7",
            "model": "qwen",
            "object": "chat.completion.chunk",
            "created": 1756546331,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "\n\n",
                        "tool_calls": [],
                    },
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
            "usage": None,
        },
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {
            "id": "chatcmpl-9a5150cb-5475-4a1b-9535-61599de39ad8",
            "model": "qwen",
            "object": "chat.completion.chunk",
            "created": 1756546331,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_9a5150cb-5475-4a1b-9535-61599de39ad8",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location": "上海"}',
                                },
                            }
                        ],
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": None,
        },
        None,
        {
            "id": "chatcmpl-882865ef-6c91-4cc0-9aec-1dff0f7cb21d",
            "model": "qwen",
            "object": "chat.completion.chunk",
            "created": 1756646459,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "", "tool_calls": []},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 159,
                "completion_tokens": 91,
                "total_tokens": 250,
            },
        },
    ]
    previous_texts = [""]

    for i, (model_uid, chunk_data) in enumerate(test_cases):
        result = mixin._post_process_completion_chunk(
            None, model_uid=model_uid, c=chunk_data, previous_texts=previous_texts
        )

        expected = expected_results[i]

        if expected is None:
            assert result is None, f"Expected None but got {result}"
        else:
            result_filtered = filter_ids_and_created(result)
            expected_filtered = filter_ids_and_created(expected)
            assert (
                result_filtered == expected_filtered
            ), f"Mismatch at case {i}: expected {expected_filtered}, got {result_filtered}"


def test_post_process_completion_chunk_with_parser():
    mixin = ChatModelMixin()
    mixin.tool_parser = QwenToolParser()
    mixin.reasoning_parser = ReasoningParser(
        reasoning_content=True,
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        enable_thinking=True,
    )
    test_cases = [
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": "\n\n"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": None,
                            "content": "<tool_call>",
                        },
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": "\n"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": '{"'},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": "name"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": '":'},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": ' "'},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": "get"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": "_current"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": "_weather"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": '",'},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": ' "'},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": None,
                            "content": "arguments",
                        },
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": '":'},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": ' {"'},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": "location"},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": '":'},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": ' "'},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": None,
                            "content": "\u4e0a\u6d77",
                        },
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": '"}}\n'},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": None,
                            "content": "</tool_call>",
                        },
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": ""},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        (
            "qwen",
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "chat.completion.chunk",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": None, "content": ""},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
        ),
    ]
    expected_results = [
        {
            "id": "chatcmpl-2f2a2673-2242-442e-9a70-9efc83e7a2e7",
            "model": "qwen",
            "object": "chat.completion.chunk",
            "created": 1756546331,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "\n\n",
                        "tool_calls": [],
                    },
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
            "usage": None,
        },
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {
            "id": "chatcmpl-9a5150cb-5475-4a1b-9535-61599de39ad8",
            "model": "qwen",
            "object": "chat.completion.chunk",
            "created": 1756546331,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_9a5150cb-5475-4a1b-9535-61599de39ad8",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location": "上海"}',
                                },
                            }
                        ],
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": None,
        },
        None,
        {
            "id": "chatcmpl-816b37c0-4e73-4a80-811c-8f7a7d9fd285",
            "model": "qwen",
            "object": "chat.completion.chunk",
            "created": 1756646644,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "", "tool_calls": []},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 159,
                "completion_tokens": 91,
                "total_tokens": 250,
            },
        },
    ]
    previous_texts = [""]

    for i, (model_uid, chunk_data) in enumerate(test_cases):
        result = mixin._post_process_completion_chunk(
            None, model_uid=model_uid, c=chunk_data, previous_texts=previous_texts
        )

        expected = expected_results[i]

        if expected is None:
            assert result is None, f"Expected None but got {result}"
        else:
            result_filtered = filter_ids_and_created(result)
            expected_filtered = filter_ids_and_created(expected)
            assert (
                result_filtered == expected_filtered
            ), f"Mismatch at case {i}: expected {expected_filtered}, got {result_filtered}"


def test_post_process_completion_without_thinking():
    mixin = ChatModelMixin()
    mixin.tool_parser = QwenToolParser()

    test_case = {
        "id": "255e6054-8686-11f0-a993-bc2411fe6c28",
        "object": "text_completion",
        "created": 1756657116,
        "model": "qwen3",
        "choices": [
            {
                "text": '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>',
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 163, "completion_tokens": 21, "total_tokens": 184},
    }

    expected_results = {
        "id": "chatcmpl-5d039f33-dcec-436f-b055-517c2ee928f9",
        "model": "qwen3",
        "object": "chat.completion",
        "created": 1756657638,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_5d039f33-dcec-436f-b055-517c2ee928f9",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{"location": "上海"}',
                            },
                        }
                    ],
                },
                "logprobs": None,
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 163, "completion_tokens": 21, "total_tokens": 184},
    }

    result = mixin._post_process_completion(None, model_uid="qwen3", c=test_case)

    result_filtered = filter_ids_and_created(result)
    expected_filtered = filter_ids_and_created(expected_results)

    assert (
        result_filtered == expected_filtered
    ), f"Mismatch: expected {expected_filtered}, got {result_filtered}"


def test_post_process_completion_with_thinking():
    mixin = ChatModelMixin()
    mixin.tool_parser = QwenToolParser()
    test_case = {
        "id": "7381cc1a-8688-11f0-a604-bc2411fe6c28",
        "object": "text_completion",
        "created": 1756658106,
        "model": "qwen3",
        "choices": [
            {
                "text": '<think>\n好的，用户问的是上海当前的天气。我需要调用get_current_weather这个工具，参数是location，也就是上海。首先确认一下工具的参数是否正确，location是必填的，所以没问题。然后生成对应的JSON请求，确保参数正确无误。最后返回工具调用的XML标签。\n</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>',
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 159, "completion_tokens": 91, "total_tokens": 250},
    }
    expected_results = {
        "id": "chatcmpl-47473344-4d79-4c64-b237-60622d52560c",
        "model": "qwen3",
        "object": "chat.completion",
        "created": 1756658106,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>\n好的，用户问的是上海当前的天气。我需要调用get_current_weather这个工具，参数是location，也就是上海。首先确认一下工具的参数是否正确，location是必填的，所以没问题。然后生成对应的JSON请求，确保参数正确无误。最后返回工具调用的XML标签。\n</think>\n\n",
                    "tool_calls": [
                        {
                            "id": "call_47473344-4d79-4c64-b237-60622d52560c",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{"location": "上海"}',
                            },
                        }
                    ],
                },
                "logprobs": None,
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 159, "completion_tokens": 91, "total_tokens": 250},
    }

    result = mixin._post_process_completion(None, model_uid="qwen3", c=test_case)

    result_filtered = filter_ids_and_created(result)
    expected_filtered = filter_ids_and_created(expected_results)

    assert (
        result_filtered == expected_filtered
    ), f"Mismatch: expected {expected_filtered}, got {result_filtered}"


def test_post_process_completion_with_parser():
    mixin = ChatModelMixin()
    mixin.tool_parser = QwenToolParser()
    mixin.reasoning_parser = ReasoningParser(
        reasoning_content=True,
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        enable_thinking=True,
    )
    test_case = {
        "id": "0bd473e2-868d-11f0-88a0-bc2411fe6c28",
        "object": "text_completion",
        "created": 1756660080,
        "model": "qwen3",
        "choices": [
            {
                "text": '<think>\n好的，用户问的是上海当前的天气。我需要调用get_current_weather这个工具来获取数据。首先，确认工具的参数是location，必须填写城市名称。用户提到的是上海，所以参数应该是"location": "上海"。然后，生成对应的JSON格式，确保正确无误。检查一下有没有其他必填项，这里只有location，所以没问题。最后，用工具调用的格式返回结果。\n</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>',
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 159, "completion_tokens": 115, "total_tokens": 274},
    }
    expected_results = {
        "id": "chatcmpl-67b060a7-674a-4288-ba73-a4089f1c3c26",
        "model": "qwen3",
        "object": "chat.completion",
        "created": 1756660090,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\n",
                    "tool_calls": [
                        {
                            "id": "call_67b060a7-674a-4288-ba73-a4089f1c3c26",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{"location": "上海"}',
                            },
                        }
                    ],
                    "reasoning_content": '\n好的，用户问的是上海当前的天气。我需要调用get_current_weather这个工具来获取数据。首先，确认工具的参数是location，必须填写城市名称。用户提到的是上海，所以参数应该是"location": "上海"。然后，生成对应的JSON格式，确保正确无误。检查一下有没有其他必填项，这里只有location，所以没问题。最后，用工具调用的格式返回结果。\n',
                },
                "logprobs": None,
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 159, "completion_tokens": 115, "total_tokens": 274},
    }
    result = mixin._post_process_completion(None, model_uid="qwen3", c=test_case)

    result_filtered = filter_ids_and_created(result)
    expected_filtered = filter_ids_and_created(expected_results)

    assert (
        result_filtered == expected_filtered
    ), f"Mismatch: expected {expected_filtered}, got {result_filtered}"


# ── Security tests for Llama3ToolParser ────────────────────


class TestEvalLlama3ChatArgumentsSecurity:
    """Verify Llama3ToolParser uses safe parsing (no eval() RCE)."""

    def test_valid_json_tool_call(self):
        text = '{"name": "get_weather", "parameters": {"location": "Tokyo"}}'
        result = Llama3ToolParser().extract_tool_calls(text)
        assert result == [(None, "get_weather", {"location": "Tokyo"})]

    def test_python_literal_tool_call(self):
        text = "{'name': 'toggle', 'parameters': {'enabled': True}}"
        result = Llama3ToolParser().extract_tool_calls(text)
        assert result == [(None, "toggle", {"enabled": True})]

    def test_reject_os_import_rce(self):
        text = "__import__('os').system('echo PWNED')"
        result = Llama3ToolParser().extract_tool_calls(text)
        assert result == [(text, None, None)]

    def test_reject_class_exploit(self):
        text = "().__class__.__bases__[0].__subclasses__()"
        result = Llama3ToolParser().extract_tool_calls(text)
        assert result == [(text, None, None)]

    def test_reject_exec(self):
        text = "exec('import os')"
        result = Llama3ToolParser().extract_tool_calls(text)
        assert result == [(text, None, None)]

    def test_malformed_json(self):
        text = '{"name": "func", "parameters": {'
        result = Llama3ToolParser().extract_tool_calls(text)
        assert result == [(text, None, None)]

    def test_missing_keys(self):
        text = '{"function": "test", "args": {}}'
        result = Llama3ToolParser().extract_tool_calls(text)
        assert result == [(text, None, None)]


def test_normalize_tool_call_arguments_to_dict():
    # Pure-function unit test for ChatModelMixin._normalize_tool_call_arguments_to_dict.
    # Contract: string arguments are parsed to dict; dict passthrough; malformed
    # JSON and empty strings are left as-is so downstream surfaces a clear error;
    # None / missing arguments are skipped; non-assistant messages are skipped.
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"北京"}',  # OpenAI spec: JSON string
                    },
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {
                        "name": "dict_passthrough",
                        "arguments": {"already": "dict"},  # already dict, untouched
                    },
                },
                {
                    "id": "c3",
                    "type": "function",
                    "function": {
                        "name": "malformed",
                        "arguments": "{not json",  # malformed JSON, left as-is
                    },
                },
                {
                    "id": "c4",
                    "type": "function",
                    "function": {
                        "name": "empty_str",
                        "arguments": "",  # empty string, left as-is
                    },
                },
                {
                    "id": "c5",
                    "type": "function",
                    "function": {
                        "name": "null_args",
                        "arguments": None,  # null, left as-is
                    },
                },
                {
                    "id": "c6",
                    "type": "function",
                    "function": {
                        "name": "list_args",
                        # valid JSON but not an object — left as-is so
                        # the template raises a clear error
                        "arguments": "[1, 2, 3]",
                    },
                },
                {
                    "id": "c7",
                    "type": "function",
                    "function": {
                        "name": "null_json",
                        "arguments": "null",  # valid JSON null, left as-is
                    },
                },
                {
                    "id": "c8",
                    "type": "function",
                    "function": {
                        "name": "number_args",
                        "arguments": "42",  # valid JSON number, left as-is
                    },
                },
            ],
        },
        {"role": "user", "content": "ignored"},  # non-assistant, skipped
        "not-a-dict",  # malformed message, skipped
    ]

    result = ChatModelMixin._normalize_tool_call_arguments_to_dict(
        [m for m in messages if m != "not-a-dict"] + ["not-a-dict"]
    )

    assistant_tc = result[0]["tool_calls"]
    assert assistant_tc[0]["function"]["arguments"] == {"city": "北京"}  # string parsed
    assert assistant_tc[1]["function"]["arguments"] == {
        "already": "dict"
    }  # dict passthrough
    assert (
        assistant_tc[2]["function"]["arguments"] == "{not json"
    )  # malformed preserved
    assert assistant_tc[3]["function"]["arguments"] == ""  # empty preserved
    assert assistant_tc[4]["function"]["arguments"] is None  # null preserved
    # Valid JSON but not an object — must be left as-is so the template
    # raises a clear error rather than silently producing garbage.
    assert assistant_tc[5]["function"]["arguments"] == "[1, 2, 3]"  # list
    assert assistant_tc[6]["function"]["arguments"] == "null"  # JSON null
    assert assistant_tc[7]["function"]["arguments"] == "42"  # JSON number

    # Non-mutating contract: input messages must be unchanged.
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"city":"北京"}'
    ), "_normalize_tool_call_arguments_to_dict mutated input"
    assert messages[0]["tool_calls"][1]["function"]["arguments"] == {"already": "dict"}
    assert messages[0]["tool_calls"][2]["function"]["arguments"] == "{not json"
    assert messages[0]["tool_calls"][3]["function"]["arguments"] == ""
    assert messages[0]["tool_calls"][4]["function"]["arguments"] is None
    assert messages[0]["tool_calls"][5]["function"]["arguments"] == "[1, 2, 3]"
    assert messages[0]["tool_calls"][6]["function"]["arguments"] == "null"
    assert messages[0]["tool_calls"][7]["function"]["arguments"] == "42"
    # The returned list is a new list (callers can mutate it independently).
    assert result is not messages


def test_qwen3_family_get_full_context_handles_string_arguments():
    # Regression for the OpenAI-spec string tool_calls.function.arguments crash.
    # Pre-fix: builtin Qwen3 templates raised
    # "Can only get item pairs from a mapping" because their templates iterate
    # `tool_call.arguments|items` while OpenAI sends arguments as a JSON-encoded
    # string.
    from .. import BUILTIN_LLM_FAMILIES

    targets = {
        "Qwen3-Coder",
        "qwen3.5",
        "qwen3.6",
        "qwen3.8",
        "qwen3.8-max",
    }
    families = {
        f.model_name: f for f in BUILTIN_LLM_FAMILIES if f.model_name in targets
    }
    assert set(families) == targets, f"missing: {targets - set(families)}"

    base_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "北京天气？"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"北京"}',  # OpenAI spec: string
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": '{"temp":25,"condition":"晴"}',
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]

    for name in targets:
        fam = families[name]
        mixin = ChatModelMixin()
        mixin.model_family = SimpleNamespace(
            model_name=fam.model_name,
            model_ability=getattr(fam, "model_ability", ["chat", "tools"]),
            chat_template=fam.chat_template,
        )
        # tokenizer=None forces _build_from_raw_template, which uses the same
        # ImmutableSandboxedEnvironment as production (and does NOT register
        # from_json — the constraint that rules out in-template fixes).
        prompt = mixin.get_full_context(
            base_messages,
            chat_template=fam.chat_template,
            tokenizer=None,
            tools=tools,
        )
        assert "<parameter=city>" in prompt, f"{name}: parameter block missing"
        assert "北京" in prompt, f"{name}: argument value missing"
        # Non-mutating contract: the input messages must be unchanged.
        assert base_messages[2]["tool_calls"][0]["function"]["arguments"] == (
            '{"city":"北京"}'
        ), f"{name}: _normalize_tool_call_arguments_to_dict mutated input"
