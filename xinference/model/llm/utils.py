# Copyright 2022-2023 XProbe Inc.
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

from typing import Iterator, List

from ...types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
)


class ChatModelDataProcessorMixin:
    def __init__(
        self,
        sep: str,
        user_name: str,
        assistant_name: str,
    ):
        self._sep: str = sep
        self._user_name: str = user_name
        self._assistant_name: str = assistant_name

    def _to_prompt(
        self,
        prompt: str,
        system_prompt: str,
        chat_history: List[ChatCompletionMessage],
    ):
        ret = system_prompt
        for message in chat_history:
            role = message["role"]
            content = message["content"]
            ret += f"{self._sep}{role}: {content}"
        ret += f"{self._sep}{self._user_name}: {prompt}"
        ret += f"{self._sep}{self._assistant_name}:"
        return ret

    @staticmethod
    def _convert_chat_completion_chunks_to_chat(
        chunks: Iterator[CompletionChunk],
    ) -> Iterator[ChatCompletionChunk]:
        for i, chunk in enumerate(chunks):
            if i == 0:
                yield {
                    "id": "chat" + chunk["id"],
                    "model": chunk["model"],
                    "created": chunk["created"],
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            yield {
                "id": "chat" + chunk["id"],
                "model": chunk["model"],
                "created": chunk["created"],
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk["choices"][0]["text"],
                        },
                        "finish_reason": chunk["choices"][0]["finish_reason"],
                    }
                ],
            }

    @staticmethod
    def _convert_text_completion_to_chat(completion: Completion) -> ChatCompletion:
        return {
            "id": "chat" + completion["id"],
            "object": "chat.completion",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion["choices"][0]["text"],
                    },
                    "finish_reason": completion["choices"][0]["finish_reason"],
                }
            ],
            "usage": completion["usage"],
        }
