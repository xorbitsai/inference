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
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, TypedDict, Union

from ....types import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from .. import LLMFamilyV1, LLMSpecV1
from ..core import LLM

if TYPE_CHECKING:
    from chatglm_cpp import Pipeline


logger = logging.getLogger(__name__)


class ChatglmCppModelConfig(TypedDict, total=False):
    pass


class ChatglmCppGenerateConfig(TypedDict, total=False):
    max_tokens: int
    top_p: float
    temperature: float
    stream: bool


class ChatglmCppChatModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[ChatglmCppModelConfig] = None,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._llm: Optional["Pipeline"] = None

        # just a placeholder for now as the chatglm_cpp repo doesn't support model config.
        self._model_config = model_config

    @classmethod
    def _sanitize_generate_config(
        cls,
        chatglmcpp_generate_config: Optional[ChatglmCppGenerateConfig],
    ) -> ChatglmCppGenerateConfig:
        if chatglmcpp_generate_config is None:
            chatglmcpp_generate_config = ChatglmCppGenerateConfig()
        chatglmcpp_generate_config.setdefault("stream", False)
        return chatglmcpp_generate_config

    def load(self):
        try:
            import chatglm_cpp
        except ImportError:
            error_message = "Failed to import module 'chatglm_cpp'"
            installation_guide = [
                "Please make sure 'chatglm_cpp' is installed. ",
                "You can install it by running the following command in the terminal:\n",
                "pip install git+https://github.com/li-plus/chatglm.cpp.git@main\n\n",
                "Or visit the original git repo if the above command fails:\n",
                "https://github.com/li-plus/chatglm.cpp",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        model_file_path = os.path.join(
            self.model_path,
            self.model_spec.model_file_name_template.format(
                quantization=self.quantization
            ),
        )

        # handle legacy cache.
        legacy_model_file_path = os.path.join(self.model_path, "model.bin")
        if os.path.exists(legacy_model_file_path):
            model_file_path = legacy_model_file_path

        self._llm = chatglm_cpp.Pipeline(Path(model_file_path))

    @classmethod
    def match(cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1") -> bool:
        if llm_spec.model_format != "ggmlv3":
            return False
        if "chatglm" not in llm_family.model_name:
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True

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

        generate_config = self._sanitize_generate_config(generate_config)

        params = {
            "max_length": generate_config.get("max_tokens"),
            "max_context_length": generate_config.get("max_tokens"),
            "top_k": generate_config.get("top_k"),
            "top_p": generate_config.get("top_p"),
            "temperature": generate_config.get("temperature"),
        }

        # Remove None values to exclude missing keys from params
        params = {k: v for k, v in params.items() if v is not None}

        assert self._llm is not None

        if generate_config["stream"]:
            it = self._llm.stream_chat(
                chat_history_list,
                **params,
            )
            assert not isinstance(it, str)
            return self._convert_raw_text_chunks_to_chat(it, self.model_uid)
        else:
            c = self._llm.chat(
                chat_history_list,
                **params,
            )
            assert not isinstance(c, Iterator)
            return self._convert_raw_text_completion_to_chat(c, self.model_uid)
