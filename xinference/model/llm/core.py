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
import platform
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, TypedDict, Union

import torch

from ...constants import XINFERENCE_CACHE_DIR
from ...types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
)
from .pytorch.utils import generate_stream

if TYPE_CHECKING:
    from llama_cpp import LogitsProcessorList, StoppingCriteriaList

    from .. import ModelSpec

logger = logging.getLogger(__name__)

SIZE_TO_GPU_LAYERS = {
    3: 26,
    7: 32,
    13: 40,
    30: 60,
    65: 80,
}


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


class LlamaCppGenerateConfig(TypedDict, total=False):
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


class LlamaCppModelConfig(TypedDict, total=False):
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
    def __init__(self, model_uid: str, model_spec: "ModelSpec", *args, **kwargs):
        self.model_uid = model_uid
        self.model_spec = model_spec

    @abstractmethod
    def load(self):
        pass


class LlamaCppModel(Model):
    def __init__(
        self,
        model_uid: str,
        model_spec: "ModelSpec",
        model_path: str,
        llamacpp_model_config: Optional[LlamaCppModelConfig] = None,
    ):
        super().__init__(model_uid, model_spec)

        closest_size = min(
            SIZE_TO_GPU_LAYERS.keys(),
            key=lambda x: abs(x - model_spec.model_size_in_billions),
        )
        self._gpu_layers = SIZE_TO_GPU_LAYERS[closest_size]
        self._model_path = model_path
        self._llamacpp_model_config: LlamaCppModelConfig = self._sanitize_model_config(
            llamacpp_model_config
        )
        self._llm = None

    @staticmethod
    def _is_darwin_and_apple_silicon():
        return platform.system() == "Darwin" and platform.processor() == "arm"

    @staticmethod
    def _is_linux():
        return platform.system() == "Linux"

    def _can_apply_metal(self):
        return (
            self.model_spec.quantization == "q4_0"
            or self.model_spec.quantization == "q4_1"
        )

    def _can_apply_cublas(self):
        # TODO: figure out the quantizations supported.
        return True

    def _sanitize_model_config(
        self, llamacpp_model_config: Optional[LlamaCppModelConfig]
    ) -> LlamaCppModelConfig:
        if llamacpp_model_config is None:
            llamacpp_model_config = LlamaCppModelConfig()
        if platform.system() == "Windows":
            llamacpp_model_config.setdefault("n_ctx", 512)
        else:
            llamacpp_model_config.setdefault("n_ctx", 2048)

        if self._is_darwin_and_apple_silicon() and self._can_apply_metal():
            llamacpp_model_config.setdefault("n_gpu_layers", 1)
        elif self._is_linux() and self._can_apply_cublas():
            llamacpp_model_config.setdefault("n_gpu_layers", self._gpu_layers)

        return llamacpp_model_config

    def _sanitize_generate_config(
        self, generate_config: Optional[LlamaCppGenerateConfig]
    ) -> LlamaCppGenerateConfig:
        if generate_config is None:
            generate_config = LlamaCppGenerateConfig()
        generate_config["model"] = self.model_uid
        return generate_config

    def load(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            error_message = "Failed to import module 'llama_cpp'"
            installation_guide = [
                "Please make sure 'llama_cpp' is installed. ",
                "You can install it by visiting the installation section of the git repo:\n",
                "https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

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

        stream = generate_config.get("stream", False)
        if not stream:
            assert self._llm is not None
            completion = self._llm(prompt=prompt, **generate_config)

            return completion
        else:
            return generator_wrapper(prompt, generate_config)


class LlamaCppChatModel(LlamaCppModel):
    def __init__(
        self,
        model_uid: str,
        model_spec: "ModelSpec",
        model_path: str,
        system_prompt: str,
        sep: str,
        user_name: str,
        assistant_name: str,
        llamacpp_model_config: Optional[LlamaCppModelConfig] = None,
    ):
        super().__init__(model_uid, model_spec, model_path, llamacpp_model_config)
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

        generate_config = self._sanitize_generate_config(generate_config)

        stream = generate_config.get("stream", False)
        if stream:
            it = self.generate(full_prompt, generate_config)
            assert isinstance(it, Iterator)
            return self._convert_chat_completion_chunks_to_chat(it)
        else:
            c = self.generate(full_prompt, generate_config)
            assert not isinstance(c, Iterator)
            return self._convert_text_completion_to_chat(c)


class PytorchGenerateConfig(StrictTypedDict, total=False):
    temperature: float
    repetition_penalty: float
    top_p: float
    top_k: int
    stream: bool
    max_new_tokens: int
    echo: bool
    stop: Optional[Union[str, List[str]]]
    stop_token_ids: Optional[Union[int, List[int]]]
    stream_interval: int
    model: Optional[str]


class PytorchModelConfig(StrictTypedDict, total=False):
    revision: str
    device: str
    gpus: Optional[str]
    num_gpus: int
    max_gpu_memory: str
    load_8bit: bool
    cpu_offloading: bool
    gptq_ckpt: Optional[str]
    gptq_wbits: int
    gptq_groupsize: int
    gptq_act_order: bool


class PytorchModel(Model):
    def __init__(
        self,
        model_uid: str,
        model_spec: "ModelSpec",
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
    ):
        super().__init__(model_uid, model_spec)
        self._use_fast_tokenizer = True
        self._model_path = model_path
        self._pytorch_model_config: PytorchModelConfig = self._sanitize_model_config(
            pytorch_model_config
        )

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        if pytorch_model_config is None:
            pytorch_model_config = PytorchModelConfig()
        pytorch_model_config.setdefault("revision", "main")
        pytorch_model_config.setdefault("device", "cuda")
        pytorch_model_config.setdefault("gpus", None)
        pytorch_model_config.setdefault("num_gpus", 1)
        pytorch_model_config.setdefault("load_8bit", False)
        pytorch_model_config.setdefault("cpu_offloading", False)
        pytorch_model_config.setdefault("gptq_ckpt", None)
        pytorch_model_config.setdefault("gptq_wbits", 16)
        pytorch_model_config.setdefault("gptq_groupsize", -1)
        pytorch_model_config.setdefault("gptq_act_order", False)
        return pytorch_model_config

    def _sanitize_generate_config(
        self,
        pytorch_generate_config: Optional[PytorchGenerateConfig],
    ) -> PytorchGenerateConfig:
        if pytorch_generate_config is None:
            pytorch_generate_config = PytorchGenerateConfig()
        pytorch_generate_config.setdefault("temperature", 0.7)
        pytorch_generate_config.setdefault("repetition_penalty", 1.0)
        pytorch_generate_config.setdefault("max_new_tokens", 512)
        pytorch_generate_config.setdefault("stream_interval", 2)
        pytorch_generate_config["model"] = self.model_uid
        return pytorch_generate_config

    def load(self):
        device = self._pytorch_model_config.get("device", "cuda")
        num_gpus = self._pytorch_model_config.get("num_gpus", 1)
        cpu_offloading = self._pytorch_model_config.get("cpu_offloading", False)
        if device == "cpu":
            kwargs = {"torch_dtype": torch.float32}
        elif device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
        else:
            raise ValueError(f"Device {device} is not supported in temporary")
        kwargs["revision"] = self._pytorch_model_config.get("revision", "main")

        self._model, self._tokenizer = self.load_model(kwargs)

        if (
            device == "cuda" and num_gpus == 1 and not cpu_offloading
        ) or device == "mps":
            self._model.to(device)
        print(self._model)

    def load_model(self, kwargs: dict):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            error_message = "Failed to import module 'transformers'"
            installation_guide = [
                "Please make sure 'transformers' is installed. ",
                "You can install it by `pip install transformers`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            use_fast=self._use_fast_tokenizer,
            revision=kwargs["revision"],
            cache_dir=XINFERENCE_CACHE_DIR,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            low_cpu_mem_usage=True,
            cache_dir=XINFERENCE_CACHE_DIR,
            **kwargs,
        )
        return model, tokenizer

    def generate(
        self, prompt: str, generate_config: Optional[PytorchGenerateConfig] = None
    ) -> Union[Completion, Iterator[Completion]]:
        def generator_wrapper(prompt, generate_config):
            device = self._pytorch_model_config.get("device", "cuda")
            for completion_chunk in generate_stream(
                self._model, self._tokenizer, prompt, device, generate_config
            ):
                yield completion_chunk

        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        generate_config = self._sanitize_generate_config(generate_config)

        assert self._model is not None
        assert self._tokenizer is not None

        stream = generate_config.get("stream", False)

        if not stream:
            for completion in generator_wrapper(prompt, generate_config):
                pass
            return completion
        else:
            return generator_wrapper(prompt, generate_config)


class PytorchChatModel(PytorchModel):
    def __init__(
        self,
        model_uid: str,
        model_spec: "ModelSpec",
        model_path: str,
        system_prompt: str,
        sep: str,
        user_name: str,
        assistant_name: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
    ):
        super().__init__(model_uid, model_spec, model_path, pytorch_model_config)
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
        chunks: Iterator[Completion],
    ) -> Iterator[ChatCompletion]:
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
                    "usage": chunk["usage"],
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
                "usage": chunk["usage"],
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
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletion]]:
        system_prompt = system_prompt or self._system_prompt
        chat_history = chat_history or []
        full_prompt = self._to_prompt(prompt, system_prompt, chat_history=chat_history)

        generate_config = self._sanitize_generate_config(generate_config)

        stream = generate_config.get("stream", False)
        if stream:
            it = self.generate(full_prompt, generate_config)
            assert isinstance(it, Iterator)
            return self._convert_chat_completion_chunks_to_chat(it)
        else:
            c = self.generate(full_prompt, generate_config)
            assert not isinstance(c, Iterator)
            return self._convert_text_completion_to_chat(c)
