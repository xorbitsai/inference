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
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

from ....types import (
    SPECIAL_TOOL_PROMPT,
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionMessage,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
    PytorchGenerateConfig,
)
from ..llm_family import LLMFamilyV1, LLMSpecV1
from .core import PytorchChatModel, PytorchModelConfig


class ChatglmPytorchChatModel(PytorchChatModel):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_spec,
            quantization,
            model_path,
            pytorch_model_config=pytorch_model_config,
        )

    def _load_model(self, **kwargs):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            error_message = "Failed to import module 'transformers'"
            installation_guide = [
                "Please make sure 'transformers' is installed. ",
                "You can install it by `pip install transformers`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=kwargs["trust_remote_code"],
            encode_special_tokens=True,
            revision=kwargs["revision"],
        )
        model = AutoModel.from_pretrained(
            self.model_path,
            **kwargs,
        )
        return model, tokenizer

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format != "pytorch":
            return False
        if "chatglm" not in llm_family.model_name:
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True

    @staticmethod
    def _handle_tools(generate_config) -> Optional[dict]:
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
            "content": f"Answer the following questions as best as you can. You have access to the following tools:",
            "tools": chatglm_tools,
        }

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        tools = self._handle_tools(generate_config)
        kwargs: Dict[str, Any] = {}
        generate_config = generate_config or {}
        temperature = generate_config.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        top_p = generate_config.get("top_p")
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        max_new_tokens = generate_config.get("max_tokens")
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = int(max_new_tokens)
        # Tool calls only works for non stream, so we call chat directly.
        if prompt == SPECIAL_TOOL_PROMPT and chat_history:
            tool_message = chat_history.pop()
            content = tool_message.get("content")
            assert content is not None
            prompt = content
            kwargs["role"] = "observation"
            chat_history = [h for h in chat_history if not h.get("tool_calls")]
        if not chat_history:
            chat_history = []
        if tools:
            msg = self._model.chat(
                self._tokenizer, prompt, [tools] + chat_history, **kwargs
            )
            return self._tool_calls_completion(
                self.model_family, self.model_uid, msg, tools
            )
        else:
            stream = generate_config.get("stream", False)
            if stream:

                def _stream_generator():
                    last_chunk_text_length = 0
                    for chunk_text, _ in self._model.stream_chat(
                        self._tokenizer, prompt, chat_history, **kwargs
                    ):
                        chunk_text = chunk_text[last_chunk_text_length:]
                        last_chunk_text_length += len(chunk_text)
                        completion_choice = CompletionChoice(
                            text=chunk_text, index=0, logprobs=None, finish_reason=None
                        )
                        yield CompletionChunk(
                            id=str(uuid.uuid1()),
                            object="text_completion",
                            created=int(time.time()),
                            model=self.model_uid,
                            choices=[completion_choice],
                        )

                return self._to_chat_completion_chunks(_stream_generator())
            else:
                response, _ = self._model.chat(
                    self._tokenizer, prompt, chat_history, **kwargs
                )
                return ChatCompletion(
                    id="chat" + str(uuid.uuid1()),
                    object="chat.completion",
                    created=int(time.time()),
                    model=self.model_uid,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message={"role": "assistant", "content": response},
                            finish_reason="stop",
                        )
                    ],
                    usage=CompletionUsage(
                        prompt_tokens=-1, completion_tokens=-1, total_tokens=-1
                    ),
                )
