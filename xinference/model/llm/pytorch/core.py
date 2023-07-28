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
from typing import Iterator, List, Optional, TypedDict, Union

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
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import ChatModelMixin
from .compression import load_compress_model
from .utils import generate_stream

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
    gptq_ckpt: Optional[str]
    gptq_wbits: int
    gptq_groupsize: int
    gptq_act_order: bool


class PytorchModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._use_fast_tokenizer = True
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
            self.model_path,
            use_fast=self._use_fast_tokenizer,
            revision=kwargs["revision"],
            cache_dir=XINFERENCE_CACHE_DIR,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            cache_dir=XINFERENCE_CACHE_DIR,
            **kwargs,
        )
        return model, tokenizer

    def load(self):
        quantization = self.quantization
        num_gpus = self._pytorch_model_config.get("num_gpus", 1)
        if self._is_darwin_and_apple_silicon():
            device = self._pytorch_model_config.get("device", "mps")
        else:
            device = self._pytorch_model_config.get("device", "cuda")

        if device == "cpu":
            kwargs = {"torch_dtype": torch.float32}
        elif device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
        elif device == "mps":
            kwargs = {"torch_dtype": torch.float16}
        else:
            raise ValueError(f"Device {device} is not supported in temporary")
        kwargs["revision"] = self._pytorch_model_config.get("revision", "main")

        if quantization != "none":
            if device == "cuda" and self._is_linux():
                kwargs["device_map"] = "auto"
                if quantization == "4-bit":
                    kwargs["load_in_4bit"] = True
                elif quantization == "8-bit":
                    kwargs["load_in_8bit"] = True
                else:
                    raise ValueError(
                        f"Quantization {quantization} is not supported in temporary"
                    )
            else:
                if num_gpus != 1:
                    raise ValueError(f"Quantization is not supported for multi-gpu")
                elif quantization != "8-bit":
                    raise ValueError(
                        f"Only 8-bit quantization is supported if it is not linux system or cuda device"
                    )
                else:
                    self._model, self._tokenizer = load_compress_model(
                        model_path=self.model_path,
                        device=device,
                        torch_dtype=kwargs["torch_dtype"],
                        use_fast=self._use_fast_tokenizer,
                        revision=kwargs["revision"],
                    )
                    logger.debug(f"Model Memory: {self._model.get_memory_footprint()}")
                    return

        self._model, self._tokenizer = self._load_model(kwargs)

        if (
            device == "cuda" and num_gpus == 1 and quantization == "none"
        ) or device == "mps":
            self._model.to(device)
        logger.debug(f"Model Memory: {self._model.get_memory_footprint()}")

    @classmethod
    def match(cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1") -> bool:
        if llm_spec.model_format != "pytorch":
            return False
        if "baichuan" in llm_family.model_name:
            return False
        if "generate" not in llm_family.model_ability:
            return False
        return True

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


class PytorchChatModel(PytorchModel, ChatModelMixin):
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
            pytorch_model_config,
        )

    def _sanitize_generate_config(
        self,
        pytorch_generate_config: Optional[PytorchGenerateConfig],
    ) -> PytorchGenerateConfig:
        pytorch_generate_config = super()._sanitize_generate_config(
            pytorch_generate_config
        )
        if (
            "stop" not in pytorch_generate_config
            and self.model_family.prompt_style
            and self.model_family.prompt_style.stop
        ):
            pytorch_generate_config["stop"] = self.model_family.prompt_style.stop
        if (
            "stop_token_ids" not in pytorch_generate_config
            and self.model_family.prompt_style
            and self.model_family.prompt_style.stop_token_ids
        ):
            pytorch_generate_config[
                "stop_token_ids"
            ] = self.model_family.prompt_style.stop_token_ids

        return pytorch_generate_config

    @classmethod
    def match(cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1") -> bool:
        if llm_spec.model_format != "pytorch":
            return False
        if "baichuan" in llm_family.model_name:
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        assert self.model_family.prompt_style is not None
        prompt_style = self.model_family.prompt_style.copy()
        if system_prompt:
            prompt_style.system_prompt = system_prompt
        chat_history = chat_history or []
        full_prompt = self.get_prompt(prompt, chat_history, prompt_style)

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
