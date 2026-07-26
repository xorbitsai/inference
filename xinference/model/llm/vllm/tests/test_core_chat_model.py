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

from unittest.mock import MagicMock

import pytest
from packaging import version

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


class TestVLLMSanitizeGenerateConfig:
    """Guided-decoding extraction from response_format=json_schema.

    The OpenAI-compatible request body serializes the JSONSchema model without
    by_alias, so the real key reaching the engine is the field name ``schema_``
    (``schema`` is a reserved Pydantic name, aliased in _compat.JSONSchema).
    _sanitize_generate_config must read ``schema_`` (with a ``schema`` fallback
    for raw passthrough), matching sglang/core.py and llm/utils.py.
    """

    _SCHEMA = {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
    }

    def _sanitize(self, response_format):
        from ..core import VLLMChatModel

        return VLLMChatModel._sanitize_generate_config(
            {"response_format": response_format}
        )

    def test_json_schema_serialized_key(self):
        # Real serialized shape (field name, not alias).
        sanitized = self._sanitize(
            {
                "type": "json_schema",
                "json_schema": {"name": "judge", "schema_": self._SCHEMA},
            }
        )
        assert sanitized["guided_json"] == self._SCHEMA

    def test_json_schema_alias_fallback(self):
        # Raw passthrough shape using the alias.
        sanitized = self._sanitize(
            {
                "type": "json_schema",
                "json_schema": {"name": "judge", "schema": self._SCHEMA},
            }
        )
        assert sanitized["guided_json"] == self._SCHEMA

    def test_json_object_unaffected(self):
        sanitized = self._sanitize({"type": "json_object"})
        assert sanitized["guided_json"] is None
        assert sanitized["guided_json_object"] is True

    def test_json_schema_empty_schema_preserved(self):
        # An empty dict is a valid schema; `is None` (not truthiness) must keep
        # it instead of falling through to the absent `schema` key. The
        # downstream guard in async_generate likewise uses `is not None`, so a
        # preserved {} actually reaches vLLM guided decoding rather than being
        # dropped. This asserts the intermediate value; the async_generate path
        # depends on vLLM and is not unit-tested here.
        sanitized = self._sanitize(
            {
                "type": "json_schema",
                "json_schema": {"name": "judge", "schema_": {}},
            }
        )
        assert sanitized["guided_json"] == {}

    def test_no_response_format(self):
        from ..core import VLLMChatModel

        sanitized = VLLMChatModel._sanitize_generate_config({})
        assert sanitized["guided_json"] is None


class TestVLLMModelLogprobs:
    """Tests for vLLM logprobs wiring in the completion response path.

    These exercise the function under fix (``_build_logprobs`` and the
    ``_convert_request_output_to_completion`` site that used to hardcode
    ``logprobs=None``) directly, with a mocked vLLM request output. On
    master the response ``logprobs`` is always ``None``; on the branch it is
    populated when the engine produced them.
    """

    @staticmethod
    def _make_logprob(logprob: float, decoded_token: str):
        lp = MagicMock()
        lp.logprob = logprob
        lp.decoded_token = decoded_token
        return lp

    @classmethod
    def _make_request_output(
        cls, *, text, token_ids, logprobs, finish_reason="stop", prompt=""
    ):
        output = MagicMock()
        output.text = text
        output.index = 0
        output.token_ids = list(token_ids)
        output.logprobs = logprobs
        output.finish_reason = finish_reason
        request_output = MagicMock()
        request_output.outputs = [output]
        request_output.prompt = prompt
        request_output.prompt_token_ids = [1, 2, 3]
        return request_output

    def test_build_logprobs_none_when_not_requested(self):
        from ..core import VLLMModel

        request_output = self._make_request_output(
            text=" hello world", token_ids=[10, 11], logprobs=None
        )
        output = request_output.outputs[0]
        assert VLLMModel._build_logprobs(output) is None

    def test_build_logprobs_populated_when_present(self):
        from ..core import VLLMModel

        logprobs = [
            {
                10: self._make_logprob(-0.1, " hello"),
                99: self._make_logprob(-3.2, " hi"),
            },
            {11: self._make_logprob(-0.2, " world")},
        ]
        request_output = self._make_request_output(
            text=" hello world", token_ids=[10, 11], logprobs=logprobs
        )
        result = VLLMModel._build_logprobs(request_output.outputs[0], prompt_offset=4)
        assert result is not None
        assert result["tokens"] == [" hello", " world"]
        assert result["token_logprobs"] == [-0.1, -0.2]
        assert result["top_logprobs"][0][" hello"] == -0.1
        assert result["top_logprobs"][0][" hi"] == -3.2
        assert result["top_logprobs"][1][" world"] == -0.2
        # text_offset is relative to the full prompt + completion text.
        assert result["text_offset"] == [4, 10]

    def test_convert_request_output_to_completion_carries_logprobs(self):
        from ..core import VLLMModel

        logprobs = [{10: self._make_logprob(-0.5, "ab")}]
        request_output = self._make_request_output(
            text="ab",
            token_ids=[10],
            logprobs=logprobs,
            finish_reason="stop",
            prompt="say:",
        )
        completion = VLLMModel._convert_request_output_to_completion(
            request_id="req-1", model="m", request_output=request_output
        )
        choice = completion["choices"][0]
        assert choice["logprobs"] is not None
        assert choice["logprobs"]["tokens"] == ["ab"]
        assert choice["logprobs"]["token_logprobs"] == [-0.5]
        assert choice["logprobs"]["text_offset"] == [4]

    def test_convert_request_output_to_completion_logprobs_none_when_absent(self):
        from ..core import VLLMModel

        request_output = self._make_request_output(
            text="ab", token_ids=[10], logprobs=None
        )
        completion = VLLMModel._convert_request_output_to_completion(
            request_id="req-1", model="m", request_output=request_output
        )
        assert completion["choices"][0]["logprobs"] is None

    def test_convert_request_output_to_completion_chunk_carries_logprobs(self):
        from ..core import VLLMModel

        logprobs = [{10: self._make_logprob(-0.5, "ab")}]
        request_output = self._make_request_output(
            text="ab", token_ids=[10], logprobs=logprobs, finish_reason=None
        )
        chunk, finish_reason = VLLMModel._convert_request_output_to_completion_chunk(
            request_id="req-1", model="m", request_output=request_output
        )
        choice = chunk["choices"][0]
        assert choice["logprobs"] is not None
        assert choice["logprobs"]["tokens"] == ["ab"]
        assert choice["logprobs"]["token_logprobs"] == [-0.5]

    def test_convert_request_output_to_completion_chunk_logprobs_none_when_absent(self):
        from ..core import VLLMModel

        request_output = self._make_request_output(
            text="ab", token_ids=[10], logprobs=None, finish_reason=None
        )
        chunk, _ = VLLMModel._convert_request_output_to_completion_chunk(
            request_id="req-1", model="m", request_output=request_output
        )
        assert chunk["choices"][0]["logprobs"] is None

    def test_build_logprobs_skips_empty_decoded_token(self):
        """A malformed Logprob with no decoded token must not emit a ``str(tid)``
        key -- OpenAI ``top_logprobs`` keys are decoded token strings."""
        from ..core import VLLMModel

        lp_with_text = self._make_logprob(-0.1, "ab")
        lp_no_text = MagicMock()
        lp_no_text.logprob = -2.0
        lp_no_text.decoded_token = None
        logprobs = [{10: lp_with_text, 99: lp_no_text}]
        request_output = self._make_request_output(
            text="ab", token_ids=[10], logprobs=logprobs
        )
        result = VLLMModel._build_logprobs(request_output.outputs[0])
        assert result is not None
        # Malformed alternatives without decoded text are omitted.
        assert "99" not in result["top_logprobs"][0]
        assert "" not in result["top_logprobs"][0]
        assert "ab" in result["top_logprobs"][0]

    def test_slice_logprobs_returns_only_new_stream_tokens(self):
        from ..core import VLLMModel

        cumulative = VLLMModel._build_logprobs(
            self._make_request_output(
                text="ab",
                token_ids=[10, 11],
                logprobs=[
                    {10: self._make_logprob(-0.1, "a")},
                    {11: self._make_logprob(-0.2, "b")},
                ],
            ).outputs[0],
            prompt_offset=4,
        )
        assert cumulative is not None
        delta = VLLMModel._slice_logprobs(cumulative, 1)
        assert delta == {
            "text_offset": [5],
            "tokens": ["b"],
            "token_logprobs": [-0.2],
            "top_logprobs": [{"b": -0.2}],
        }

    def test_sanitize_translates_logprobs_request_to_vllm_int(self, monkeypatch):
        from .. import core

        monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.21.0"))

        # Legacy completions pass their integer logprobs value through.
        sanitized = core.VLLMModel._sanitize_generate_config({"logprobs": 5})
        assert sanitized["logprobs"] == 5
        sanitized = core.VLLMModel._sanitize_generate_config({"logprobs": 0})
        assert sanitized["logprobs"] == 0

        # Chat requests use a boolean plus top_logprobs.
        sanitized = core.VLLMModel._sanitize_generate_config(
            {"logprobs": True, "top_logprobs": 5}
        )
        assert sanitized["logprobs"] == 5
        sanitized = core.VLLMModel._sanitize_generate_config({"logprobs": True})
        assert sanitized["logprobs"] == 0

        # vLLM uses None, rather than 0, to disable logprobs.
        sanitized = core.VLLMModel._sanitize_generate_config({"logprobs": False})
        assert sanitized["logprobs"] is None
        sanitized = core.VLLMModel._sanitize_generate_config({})
        assert sanitized["logprobs"] is None

        # prompt_logprobs passes through
        sanitized = core.VLLMModel._sanitize_generate_config({"prompt_logprobs": 3})
        assert sanitized["prompt_logprobs"] == 3
