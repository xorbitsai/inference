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

import logging
import time
import uuid
from typing import Iterator, List, Optional

from .core import ChatglmCppGenerateConfig, Model
from .types import ChatCompletionChunk, ChatCompletionMessage

logger = logging.getLogger(__name__)


class ChatglmCppChatModel(Model):
    def __init__(
        self,
        model_path: str,
        chatglmcpp_generate_config: Optional[ChatglmCppGenerateConfig] = None,
    ):
        super().__init__()
        self._llm = None
        self._model_path = model_path
        self._model_name = "-".join(self._model_path.split("/")[-2].split("-")[:-3])
        self._chatglmcpp_generate_config: ChatglmCppGenerateConfig = (
            self._sanitize_generate_config(chatglmcpp_generate_config)
        )

    @classmethod
    def _sanitize_generate_config(
        cls,
        chatglmcpp_generate_config: Optional[ChatglmCppGenerateConfig],
    ) -> ChatglmCppGenerateConfig:
        if chatglmcpp_generate_config is None:
            chatglmcpp_generate_config = ChatglmCppGenerateConfig()
        return chatglmcpp_generate_config

    def load(self):
        import chatglm_cpp

        self._llm = chatglm_cpp.Pipeline(self._model_path)

    @staticmethod
    def _convert_raw_text_chunks_to_chat(
        tokens: Iterator[str], model_name: str
    ) -> Iterator[ChatCompletionChunk]:
        yield {
            "id": "chat" + f"cmpl-{str(uuid.uuid4())}",
            "model": model_name,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
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
        for token in enumerate(tokens):
            yield {
                "id": "chat" + f"cmpl-{str(uuid.uuid4())}",
                "model": model_name,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": token[1],
                        },
                        "finish_reason": None,
                    }
                ],
            }

    def chat(
        self,
        prompt: str,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[ChatglmCppGenerateConfig] = None,
    ) -> Iterator[ChatCompletionChunk]:
        if chat_history is not None:
            chat_history_list = [message["content"] for message in chat_history]
        else:
            chat_history_list = []

        chat_history_list.append(prompt)
        logger.debug("Full conversation history:\n%s", str(chat_history_list))
        generate_config = generate_config or {}

        assert self._llm is not None
        it = self._llm.stream_chat(
            chat_history_list,
            max_context_length=1000,
            max_length=generate_config["max_tokens"],
            temperature=generate_config["temperature"],
            top_p=generate_config["top_p"],
        )
        return self._convert_raw_text_chunks_to_chat(it, self._model_name)
