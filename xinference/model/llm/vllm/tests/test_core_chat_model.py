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

from unittest.mock import MagicMock

import pytest

from ...tool_parsers.qwen_tool_parser import QwenToolParser


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


class TestVLLMChatModel:

    @pytest.fixture
    def real_vllm_chat_model(self):
        from ..core import VLLMChatModel

        model = object.__new__(VLLMChatModel)

        model.model_family = MagicMock()
        model.model_family.model_family = "qwen"
        model.model_family.reasoning_start_tag = "<think>"
        model.model_family.reasoning_end_tag = "</think>"
        model.model_uid = "test-model-0"
        model.reasoning_parser = None
        model.tool_parser = QwenToolParser()

        return model

    async def create_mock_chunks(self, chunks_data):
        for chunk in chunks_data:
            yield chunk

    @pytest.mark.asyncio
    async def test_async_to_tool_completion_chunks_without_thinking(
        self, real_vllm_chat_model
    ):
        test_chunks = [
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "<tool_call>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 71,
                    "total_tokens": 230,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "\n", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 72,
                    "total_tokens": 231,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '{"', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 73,
                    "total_tokens": 232,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "name",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 74,
                    "total_tokens": 233,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 75,
                    "total_tokens": 234,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 76,
                    "total_tokens": 235,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "get", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 77,
                    "total_tokens": 236,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "_current",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 78,
                    "total_tokens": 237,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "_weather",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 79,
                    "total_tokens": 238,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '",', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 80,
                    "total_tokens": 239,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 81,
                    "total_tokens": 240,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "arguments",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 82,
                    "total_tokens": 241,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 83,
                    "total_tokens": 242,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' {"', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 84,
                    "total_tokens": 243,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "location",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 85,
                    "total_tokens": 244,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 86,
                    "total_tokens": 245,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 87,
                    "total_tokens": 246,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "上海",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 88,
                    "total_tokens": 247,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": '"}}\n',
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 89,
                    "total_tokens": 248,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "</tool_call>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 90,
                    "total_tokens": 249,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
        ]

        chunks_generator = self.create_mock_chunks(test_chunks)
        result_chunks = []
        expected_chunks = [
            {
                "id": "chatcmpl-7fcac134-7380-4a19-b665-d93ffaacfbca",
                "model": "test-model-0",
                "object": "chat.completion.chunk",
                "created": 1756644905,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_7fcac134-7380-4a19-b665-d93ffaacfbca",
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
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1,
                },
            },
            {
                "id": "chatcmpl-06a03091-f455-4dfe-a348-2163cf285811",
                "model": "test-model-0",
                "object": "chat.completion.chunk",
                "created": 1756644905,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "", "tool_calls": []},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "completion_tokens": 91,
                    "prompt_tokens": 159,
                    "total_tokens": 250,
                },
            },
        ]

        i = 0
        async for chunk in real_vllm_chat_model._async_to_tool_completion_chunks(
            chunks_generator
        ):
            result = filter_ids_and_created(chunk)
            expected_result = filter_ids_and_created(expected_chunks[i])
            assert result == expected_result
            result_chunks.append(chunk)
            i += 1

    @pytest.mark.asyncio
    async def test_async_to_tool_completion_chunks_with_thinking(
        self, real_vllm_chat_model
    ):
        test_chunks = [
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "<think>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 1,
                    "total_tokens": 160,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {"text": "\n", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 2,
                    "total_tokens": 161,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "好的",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 3,
                    "total_tokens": 162,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "</think>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 69,
                    "total_tokens": 228,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "\n\n",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 70,
                    "total_tokens": 229,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "<tool_call>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 71,
                    "total_tokens": 230,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "\n", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 72,
                    "total_tokens": 231,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '{"', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 73,
                    "total_tokens": 232,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "name",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 74,
                    "total_tokens": 233,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 75,
                    "total_tokens": 234,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 76,
                    "total_tokens": 235,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "get", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 77,
                    "total_tokens": 236,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "_current",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 78,
                    "total_tokens": 237,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "_weather",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 79,
                    "total_tokens": 238,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '",', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 80,
                    "total_tokens": 239,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 81,
                    "total_tokens": 240,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "arguments",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 82,
                    "total_tokens": 241,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 83,
                    "total_tokens": 242,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' {"', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 84,
                    "total_tokens": 243,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "location",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 85,
                    "total_tokens": 244,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 86,
                    "total_tokens": 245,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 87,
                    "total_tokens": 246,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "上海",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 88,
                    "total_tokens": 247,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": '"}}\n',
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 89,
                    "total_tokens": 248,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "</tool_call>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 90,
                    "total_tokens": 249,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
        ]

        chunks_generator = self.create_mock_chunks(test_chunks)
        result_chunks = []

        gen = real_vllm_chat_model._async_to_tool_completion_chunks(chunks_generator)

        async for chunk in gen:
            result_chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_async_to_tool_completion_chunks_with_parser(
        self, real_vllm_chat_model
    ):
        test_chunks = [
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "<think>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 1,
                    "total_tokens": 160,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {"text": "\n", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 2,
                    "total_tokens": 161,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "好的",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 3,
                    "total_tokens": 162,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "</think>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 69,
                    "total_tokens": 228,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "\n\n",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 70,
                    "total_tokens": 229,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "<tool_call>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 71,
                    "total_tokens": 230,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "\n", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 72,
                    "total_tokens": 231,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '{"', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 73,
                    "total_tokens": 232,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "name",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 74,
                    "total_tokens": 233,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 75,
                    "total_tokens": 234,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 76,
                    "total_tokens": 235,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "get", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 77,
                    "total_tokens": 236,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "_current",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 78,
                    "total_tokens": 237,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "_weather",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 79,
                    "total_tokens": 238,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '",', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 80,
                    "total_tokens": 239,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 81,
                    "total_tokens": 240,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "arguments",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 82,
                    "total_tokens": 241,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 83,
                    "total_tokens": 242,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' {"', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 84,
                    "total_tokens": 243,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "location",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 85,
                    "total_tokens": 244,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 86,
                    "total_tokens": 245,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 87,
                    "total_tokens": 246,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "上海",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 88,
                    "total_tokens": 247,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": '"}}\n',
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 89,
                    "total_tokens": 248,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "</tool_call>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 90,
                    "total_tokens": 249,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
        ]

        chunks_generator = self.create_mock_chunks(test_chunks)
        result_chunks = []
        expected_chunks = [
            {
                "id": "chatcd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "model": "qwen3",
                "created": 1756451239,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "", "content": None},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "model": "qwen3",
                "created": 1756451239,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "\n", "content": None},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "model": "qwen3",
                "created": 1756451239,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "好的", "content": None},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "model": "qwen3",
                "created": 1756451240,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "", "content": None},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-e3ec64af-ed8f-4706-9544-4f8d7b42c85b",
                "model": "test-model-0",
                "object": "chat.completion.chunk",
                "created": 1756646208,
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
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1,
                },
            },
            {
                "id": "chatcmpl-490011af-9e50-4dea-969b-f10828d5a5ea",
                "model": "test-model-0",
                "object": "chat.completion.chunk",
                "created": 1756646208,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_490011af-9e50-4dea-969b-f10828d5a5ea",
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
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1,
                },
            },
            {
                "id": "chatcmpl-b5a05647-d043-43bb-a7e6-58907e7f4288",
                "model": "test-model-0",
                "object": "chat.completion.chunk",
                "created": 1756646208,
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
        real_vllm_chat_model.prepare_parse_reasoning_content(True, enable_thinking=True)

        i = 0
        async for chunk in real_vllm_chat_model._async_to_tool_completion_chunks(
            chunks_generator
        ):
            result_chunks.append(chunk)
            result = filter_ids_and_created(chunk)
            expected_result = filter_ids_and_created(expected_chunks[i])
            assert result == expected_result
            i = i + 1
