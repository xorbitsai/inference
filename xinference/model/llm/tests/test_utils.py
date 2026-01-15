# Copyright 2022-2026 XProbe Inc.
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

from ..reasoning_parser import ReasoningParser
from ..tool_parsers.qwen_tool_parser import QwenToolParser
from ..utils import ChatModelMixin


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
