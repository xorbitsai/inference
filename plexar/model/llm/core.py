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

import abc
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, TypedDict, Union

from .types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
)

if TYPE_CHECKING:
    from llama_cpp import LogitsProcessorList, StoppingCriteriaList

logger = logging.getLogger(__name__)


class StrictTypedDict(TypedDict):
    def __setitem__(self, key: str, value: Any):  # type: ignore
        if key not in self.__annotations__:
            raise KeyError(f"Key '{key}' is not allowed in {self.__class__.__name__}")

        expected_type = self.__annotations__[key]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Value for key '{key}' must be of type '{expected_type.__name__}', "
                f"not '{type(value).__name__}'"
            )

        super().__setitem__(key, value)


class LlamaCppGenerateConfig(StrictTypedDict, total=False):
    suffix: Optional[str]
    max_tokens: int
    temperature: float
    top_p: float
    logprobs: Optional[int]
    echo: bool
    stop: Optional[Union[str, List[str]]]
    frequency_penalty: float
    presence_penalty: float
    repeat_penalty: float
    top_k: int
    stream: bool
    tfs_z: float
    mirostat_mode: int
    mirostat_tau: float
    mirostat_eta: float
    model: Optional[str]
    stopping_criteria: Optional["StoppingCriteriaList"]
    logits_processor: Optional["LogitsProcessorList"]


class LlamaCppModelConfig(StrictTypedDict, total=False):
    n_ctx: int
    n_parts: int
    n_gpu_layers: int
    seed: int
    f16_kv: bool
    logits_all: bool
    vocab_only: bool
    use_mmap: bool
    use_mlock: bool
    embedding: bool
    n_threads: Optional[int]
    n_batch: int
    last_n_tokens_size: int
    lora_base: Optional[str]
    lora_path: Optional[str]
    low_vram: bool
    verbose: bool


class Model(abc.ABC):
    name: str

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self):
        pass


class LlamaCppModel(Model):
    def __init__(
        self,
        model_path: str,
        llamacpp_model_config: Optional[LlamaCppModelConfig] = None,
    ):
        super().__init__()
        self._model_path = model_path
        self._llamacpp_model_config: LlamaCppModelConfig = self._sanitize_model_config(
            llamacpp_model_config
        )
        self._llm = None

    @classmethod
    def _sanitize_model_config(
        cls, llamacpp_model_config: Optional[LlamaCppModelConfig]
    ) -> LlamaCppModelConfig:
        if llamacpp_model_config is None:
            llamacpp_model_config = LlamaCppModelConfig()
        return llamacpp_model_config

    @classmethod
    def _sanitize_generate_config(
        cls,
        generate_config: Optional[LlamaCppGenerateConfig],
    ) -> LlamaCppGenerateConfig:
        if generate_config is None:
            generate_config = LlamaCppGenerateConfig()
        return generate_config

    def load(self):
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=self._model_path,
            verbose=False,
            **self._llamacpp_model_config,
        )

    def generate(
        self, prompt: str, generate_config: Optional[LlamaCppGenerateConfig] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        def generator_wrapper(
            _prompt: str, _generate_config: LlamaCppGenerateConfig
        ) -> Iterator[CompletionChunk]:
            assert self._llm is not None
            for _completion_chunk in self._llm(prompt=_prompt, **_generate_config):
                yield _completion_chunk

        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        generate_config = self._sanitize_generate_config(generate_config)

        assert self._llm is not None

        stream = True
        if not generate_config or "stream" not in generate_config:
            generate_config["stream"] = stream
        else:
            stream = generate_config["stream"]

        if not stream:
            completion = self._llm(prompt=prompt, **generate_config)

            return completion
        else:
            return generator_wrapper(prompt, generate_config)


class LlamaCppChatModel(LlamaCppModel):
    def __init__(
        self,
        model_path: str,
        system_prompt: str,
        sep: str,
        user_name: str,
        assistant_name: str,
        llamacpp_model_config: Optional[LlamaCppModelConfig] = None,
    ):
        super().__init__(model_path, llamacpp_model_config)
        self._system_prompt: str = system_prompt
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

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[LlamaCppGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        system_prompt = system_prompt or self._system_prompt
        chat_history = chat_history or []
        full_prompt = self._to_prompt(prompt, system_prompt, chat_history=chat_history)

        logger.debug("Full prompt:\n%s", full_prompt)

        stream = True
        generate_config = generate_config or {}
        if "stream" not in generate_config:
            generate_config["stream"] = stream
        else:
            stream = generate_config["stream"]

        if stream:
            it = self.generate(full_prompt, generate_config)
            assert isinstance(it, Iterator)
            return self._convert_chat_completion_chunks_to_chat(it)
        else:
            c = self.generate(full_prompt, generate_config)
            assert not isinstance(c, Iterator)
            return self._convert_text_completion_to_chat(c)
