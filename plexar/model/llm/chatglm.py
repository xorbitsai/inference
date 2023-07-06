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
from pathlib import Path
from typing import Iterator, List, Optional, Union

from .core import ChatglmCppGenerateConfig, Model
from .types import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage

logger = logging.getLogger(__name__)


class ChatglmCppChatModel(Model):
    def __init__(
        self,
        model_path: str,
        model_config: Optional[ChatglmCppGenerateConfig] = None,
    ):
        super().__init__()
        self._llm = None
        self._model_path = model_path
        self._model_name = "-".join(self._model_path.split("/")[-2].split("-")[:-3])

    @classmethod
    def _sanitize_generate_config(
        cls,
        chatglmcpp_generate_config: Optional[ChatglmCppGenerateConfig],
    ) -> ChatglmCppGenerateConfig:
        if chatglmcpp_generate_config is None:
            chatglmcpp_generate_config = ChatglmCppGenerateConfig()
        chatglmcpp_generate_config.setdefault("max_tokens", 8192)
        chatglmcpp_generate_config.setdefault("temperature", 0.95)
        chatglmcpp_generate_config.setdefault("top_p", 0.8)
        chatglmcpp_generate_config.setdefault("stream", True)
        return chatglmcpp_generate_config

    def load(self):
        import chatglm_cpp

        self._llm = chatglm_cpp.Pipeline(Path(self._model_path))

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

    @staticmethod
    def _convert_raw_text_completion_to_chat(
        text: str, model_name: str
    ) -> ChatCompletion:
        return {
            "id": "chat" + f"cmpl-{str(uuid.uuid4())}",
            "model": model_name,
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": None,
                }
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }

    def chat(
        self,
        prompt: str,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[ChatglmCppGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        if chat_history is not None:
            chat_history_list = [message["content"] for message in chat_history]
        else:
            chat_history_list = []

        chat_history_list.append(prompt)
        logger.debug("Full conversation history:\n%s", str(chat_history_list))

        stream = False
        generate_config = self._sanitize_generate_config(generate_config)
        if "stream" not in generate_config:
            generate_config["stream"] = stream
        else:
            stream = generate_config["stream"]

        assert self._llm is not None

        if stream:
            it = self._llm.stream_chat(
                chat_history_list,
                max_context_length=8192,
                max_length=generate_config["max_tokens"],
                temperature=generate_config["temperature"],
                top_p=generate_config["top_p"],
            )
            assert not isinstance(it, str)
            return self._convert_raw_text_chunks_to_chat(it, self._model_name)
        else:
            c = self._llm.chat(
                chat_history_list,
                max_context_length=8192,
                max_length=generate_config["max_tokens"],
                temperature=generate_config["temperature"],
                top_p=generate_config["top_p"],
            )
            assert not isinstance(c, Iterator)
            print(self._convert_raw_text_completion_to_chat(c, self._model_name))
            return self._convert_raw_text_completion_to_chat(c, self._model_name)
