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
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

from ....types import (
    SPECIAL_TOOL_PROMPT,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatglmCppGenerateConfig,
    ChatglmCppModelConfig,
    Completion,
    CompletionChunk,
)
from .. import LLMFamilyV1, LLMSpecV1
from ..core import LLM

if TYPE_CHECKING:
    from chatglm_cpp import Pipeline


logger = logging.getLogger(__name__)


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
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format != "ggmlv3":
            return False
        if "chatglm" not in llm_family.model_name:
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True

    @staticmethod
    def _convert_raw_text_chunks_to_chat(
        tokens: Iterator[Any], model_name: str
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
        for token in tokens:
            yield {
                "id": "chat" + f"cmpl-{str(uuid.uuid4())}",
                "model": model_name,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": (
                                token if isinstance(token, str) else token.content
                            ),
                        },
                        "finish_reason": None,
                    }
                ],
            }

    @classmethod
    def _convert_raw_text_completion_to_chat(
        cls, text: Any, model_name: str
    ) -> ChatCompletion:
        _id = str(uuid.uuid4())
        return {
            "id": "chat" + f"cmpl-{_id}",
            "model": model_name,
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": cls._message_to_json_string(_id, text),
                    "finish_reason": cls._finish_reason_from_msg(text),
                }
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }

    @staticmethod
    def _finish_reason_from_msg(msg):
        if isinstance(msg, str):
            return None
        else:
            return "tool_calls" if msg.tool_calls else "stop"

    @staticmethod
    def _eval_arguments(arguments):
        def tool_call(**kwargs):
            return kwargs

        try:
            return json.dumps(eval(arguments, dict(tool_call=tool_call)))
        except Exception:
            return f"Invalid arguments {arguments}"

    @classmethod
    def _message_to_json_string(cls, _id, msg) -> ChatCompletionMessage:
        if isinstance(msg, str):
            return {
                "role": "assistant",
                "content": msg,
            }
        else:
            return {
                "role": msg.role,
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": f"call_{_id}",
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": cls._eval_arguments(tc.function.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }

    @staticmethod
    def _handle_tools(generate_config) -> Optional[ChatCompletionMessage]:
        """Convert openai tools to ChatGLM tools."""
        if generate_config is None:
            return None
        tools = generate_config.pop("tools", None)
        if tools is None:
            return None
        chatglm_tools = []
        for elem in tools:
            if elem.get("type") != "function" or "function" not in elem:
                raise ValueError("ChatGLM tools only support function type.")
            chatglm_tools.append(elem["function"])
        return {
            "role": "system",
            "content": (
                f"Answer the following questions as best as you can. You have access to the following tools:\n"
                f"{json.dumps(chatglm_tools, indent=4, ensure_ascii=False)}"
            ),
        }

    @staticmethod
    def _to_chatglm_chat_messages(history_list: List[Any]):
        from chatglm_cpp import ChatMessage

        return [ChatMessage(role=v["role"], content=v["content"]) for v in history_list]

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[ChatglmCppGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        chat_history_list = []
        if system_prompt is not None:
            chat_history_list.append({"role": "system", "content": system_prompt})
        if chat_history is not None:
            chat_history_list.extend(chat_history)  # type: ignore

        tool_message = self._handle_tools(generate_config)
        if tool_message is not None:
            chat_history_list.insert(0, tool_message)  # type: ignore

        # We drop the message which contains tool calls to walkaround the issue:
        # https://github.com/li-plus/chatglm.cpp/issues/231
        chat_history_list = [m for m in chat_history_list if not m.get("tool_calls")]
        for idx, m in enumerate(chat_history_list):
            if m.get("role") == "tool":
                # Reconstruct a simple tool message.
                chat_history_list[idx] = {
                    "content": m["content"],
                    "role": "observation",
                }
                break

        if prompt != SPECIAL_TOOL_PROMPT:
            chat_history_list.append({"role": "user", "content": prompt})
        logger.debug("Full conversation history:\n%s", str(chat_history_list))

        generate_config = self._sanitize_generate_config(generate_config)

        params = {
            "max_length": generate_config.get("max_tokens"),
            "max_context_length": generate_config.get("max_tokens"),
            "top_k": generate_config.get("top_k"),
            "top_p": generate_config.get("top_p"),
            "temperature": generate_config.get("temperature"),
            "stream": generate_config.get("stream", False),
        }

        # Remove None values to exclude missing keys from params
        params = {k: v for k, v in params.items() if v is not None}

        assert self._llm is not None
        chat_history_messages = self._to_chatglm_chat_messages(chat_history_list)

        if generate_config["stream"]:
            it = self._llm.chat(
                chat_history_messages,
                **params,
            )
            assert not isinstance(it, str)
            return self._convert_raw_text_chunks_to_chat(it, self.model_uid)
        else:
            c = self._llm.chat(
                chat_history_messages,
                **params,
            )
            assert not isinstance(c, Iterator)
            return self._convert_raw_text_completion_to_chat(c, self.model_uid)

    @staticmethod
    def _convert_str_to_completion(data: str, model_name: str) -> Completion:
        return {
            "id": "generate" + f"-{str(uuid.uuid4())}",
            "model": model_name,
            "object": "text_completion",
            "created": int(time.time()),
            "choices": [
                {"index": 0, "text": data, "finish_reason": None, "logprobs": None}
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }

    @staticmethod
    def _convert_str_to_completion_chunk(
        tokens: Iterator[str], model_name: str
    ) -> Iterator[CompletionChunk]:
        for token in tokens:
            yield {
                "id": "generate" + f"-{str(uuid.uuid4())}",
                "model": model_name,
                "object": "text_completion",
                "created": int(time.time()),
                "choices": [
                    {"index": 0, "text": token, "finish_reason": None, "logprobs": None}
                ],
            }

    def generate(
        self,
        prompt: str,
        generate_config: Optional[ChatglmCppGenerateConfig] = None,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        logger.debug(f"Prompt for generate:\n{prompt}")

        generate_config = self._sanitize_generate_config(generate_config)

        params = {
            "max_length": generate_config.get("max_tokens"),
            "max_context_length": generate_config.get("max_tokens"),
            "top_k": generate_config.get("top_k"),
            "top_p": generate_config.get("top_p"),
            "temperature": generate_config.get("temperature"),
            "stream": generate_config.get("stream", False),
        }

        # Remove None values to exclude missing keys from params
        params = {k: v for k, v in params.items() if v is not None}

        assert self._llm is not None

        if generate_config["stream"]:
            it = self._llm.generate(
                prompt,
                **params,
            )
            assert not isinstance(it, str)
            return self._convert_str_to_completion_chunk(it, self.model_uid)
        else:
            c = self._llm.generate(
                prompt,
                **params,
            )
            assert not isinstance(c, Iterator)
            return self._convert_str_to_completion(c, self.model_uid)
