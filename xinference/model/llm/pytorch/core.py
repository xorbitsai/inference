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
from typing import TYPE_CHECKING, Iterator, List, Optional, TypedDict, Union

import torch

from ....constants import XINFERENCE_CACHE_DIR
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
    Embedding,
)
from ..core import Model
from ..utils import ChatModelDataProcessorMixin
from .utils import generate_stream

if TYPE_CHECKING:
    from ... import ModelSpec

logger = logging.getLogger(__name__)


class PytorchGenerateConfig(TypedDict, total=False):
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


class PytorchModelConfig(TypedDict, total=False):
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
        pytorch_model_config.setdefault("gpus", None)
        pytorch_model_config.setdefault("num_gpus", 1)
        pytorch_model_config.setdefault("load_8bit", False)
        pytorch_model_config.setdefault("cpu_offloading", False)
        pytorch_model_config.setdefault("gptq_ckpt", None)
        pytorch_model_config.setdefault("gptq_wbits", 16)
        pytorch_model_config.setdefault("gptq_groupsize", -1)
        pytorch_model_config.setdefault("gptq_act_order", False)
        if self._is_darwin_and_apple_silicon():
            pytorch_model_config.setdefault("device", "mps")
        else:
            pytorch_model_config.setdefault("device", "cuda")
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

    def _load_model(self, kwargs: dict):
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

    def load(self):
        num_gpus = self._pytorch_model_config.get("num_gpus", 1)
        cpu_offloading = self._pytorch_model_config.get("cpu_offloading", False)
        if self._is_darwin_and_apple_silicon():
            device = self._pytorch_model_config.get("device", "mps")
        else:
            device = self._pytorch_model_config.get("device", "cuda")

        if device == "cpu":
            kwargs = {"torch_dtype": torch.float32}
        elif device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
            if cpu_offloading:
                kwargs["device_map"] = "auto"
        elif device == "mps":
            kwargs = {"torch_dtype": torch.float16}
        else:
            raise ValueError(f"Device {device} is not supported in temporary")
        kwargs["revision"] = self._pytorch_model_config.get("revision", "main")

        self._model, self._tokenizer = self._load_model(kwargs)

        quantization = self.model_spec.quantization
        if quantization == "int4":
            self._model = self._model.quantize(4)
        elif quantization == "int8":
            self._model = self._model.quantize(8)

        if (
            device == "cuda" and num_gpus == 1 and not cpu_offloading
        ) or device == "mps":
            self._model.to(device)
        print(self._model)

    def generate(
        self, prompt: str, generate_config: Optional[PytorchGenerateConfig] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        def generator_wrapper(
            prompt: str, device: str, generate_config: PytorchGenerateConfig
        ) -> Iterator[CompletionChunk]:
            for completion_chunk, _ in generate_stream(
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
        if self._is_darwin_and_apple_silicon():
            device = self._pytorch_model_config.get("device", "mps")
        else:
            device = self._pytorch_model_config.get("device", "cuda")
        if not stream:
            for completion_chunk, completion_usage in generate_stream(
                self._model, self._tokenizer, prompt, device, generate_config
            ):
                pass
            completion = Completion(
                id=completion_chunk["id"],
                object=completion_chunk["object"],
                created=completion_chunk["created"],
                model=completion_chunk["model"],
                choices=completion_chunk["choices"],
                usage=completion_usage,
            )
            return completion
        else:
            return generator_wrapper(prompt, device, generate_config)

    def create_embedding(self, input: Union[str, List[str]]) -> Embedding:
        raise NotImplementedError


class PytorchChatModel(PytorchModel, ChatModelDataProcessorMixin):
    def __init__(
        self,
        model_uid: str,
        model_spec: "ModelSpec",
        model_path: str,
        system_prompt: str,
        sep: str,
        user_name: str,
        assistant_name: str,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[Union[int, List[int]]] = None,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
    ):
        super().__init__(model_uid, model_spec, model_path, pytorch_model_config)
        self._system_prompt: str = system_prompt
        self._sep: str = sep
        self._user_name: str = user_name
        self._assistant_name: str = assistant_name
        self._stop: Optional[Union[str, List[str]]] = stop
        self._stop_token_ids: Optional[Union[int, List[int]]] = stop_token_ids

    def _sanitize_generate_config(
        self,
        pytorch_generate_config: Optional[PytorchGenerateConfig],
    ) -> PytorchGenerateConfig:
        pytorch_generate_config = super()._sanitize_generate_config(
            pytorch_generate_config
        )
        if "stop" not in pytorch_generate_config and self._stop is not None:
            pytorch_generate_config["stop"] = self._stop
        if (
            "stop_token_ids" not in pytorch_generate_config
            and self._stop_token_ids is not None
        ):
            pytorch_generate_config["stop_token_ids"] = self._stop_token_ids

        return pytorch_generate_config

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
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
