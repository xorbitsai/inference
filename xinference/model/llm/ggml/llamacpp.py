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
import platform
from typing import TYPE_CHECKING, Iterator, List, Optional, TypedDict, Union

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
from .ctransformers import CTRANSFORMERS_SUPPORTED_MODEL

if TYPE_CHECKING:
    from llama_cpp import LogitsProcessorList, StoppingCriteriaList

logger = logging.getLogger(__name__)


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
    repetition_penalty: float
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
    n_gqa: Optional[int]  # (TEMPORARY) must be 8 for llama2 70b
    rms_norm_eps: Optional[float]  # (TEMPORARY)
    verbose: bool


SIZE_TO_GPU_LAYERS = {
    3: 26,
    7: 32,
    13: 40,
    30: 60,
    65: 80,
}


class LlamaCppModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        llamacpp_model_config: Optional[LlamaCppModelConfig] = None,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)

        closest_size = min(
            SIZE_TO_GPU_LAYERS.keys(),
            key=lambda x: abs(x - model_spec.model_size_in_billions),
        )
        self._gpu_layers = SIZE_TO_GPU_LAYERS[closest_size]
        self._llamacpp_model_config: LlamaCppModelConfig = self._sanitize_model_config(
            llamacpp_model_config
        )
        self._llm = None

    def _can_apply_metal(self):
        return self.quantization in ["q4_0", "q4_1"]

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

        llamacpp_model_config.setdefault("embedding", True)
        llamacpp_model_config.setdefault("use_mmap", False)
        llamacpp_model_config.setdefault("use_mlock", True)

        if (
            "llama-2" in self.model_family.model_name
            and self.model_spec.model_size_in_billions == 70
        ):
            llamacpp_model_config["use_mlock"] = False
            llamacpp_model_config["n_gqa"] = 8

        if self._is_darwin_and_apple_silicon() and self._can_apply_metal():
            # TODO: platform.processor() is not safe, need to be replaced to other method.
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

        # handle legacy cache.
        model_path = os.path.join(
            self.model_path,
            self.model_spec.model_file_name_template.format(
                quantization=self.quantization
            ),
        )
        legacy_model_file_path = os.path.join(self.model_path, "model.bin")
        if os.path.exists(legacy_model_file_path):
            model_path = legacy_model_file_path

        self._llm = Llama(
            model_path=model_path,
            verbose=False,
            **self._llamacpp_model_config,
        )

    @classmethod
    def match(cls, llm_family: LLMFamilyV1, llm_spec: LLMSpecV1) -> bool:
        if llm_spec.model_format != "ggmlv3":
            return False
        if (
            "chatglm" in llm_family.model_name
            or llm_family.model_name in CTRANSFORMERS_SUPPORTED_MODEL
        ):
            return False
        if "generate" not in llm_family.model_ability:
            return False
        return True

    def generate(
        self, prompt: str, generate_config: Optional[LlamaCppGenerateConfig] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        def generator_wrapper(
            _prompt: str,
            repeat_penalty: float,
            _generate_config: LlamaCppGenerateConfig,
        ) -> Iterator[CompletionChunk]:
            assert self._llm is not None
            for _completion_chunk in self._llm(
                prompt=_prompt, repeat_penalty=repeat_penalty, **_generate_config
            ):
                yield _completion_chunk

        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        generate_config = self._sanitize_generate_config(generate_config)

        repeat_penalty = 1.1
        if "repetition_penalty" in generate_config:
            repeat_penalty = generate_config["repetition_penalty"]
            generate_config.pop("repetition_penalty")

        stream = generate_config.get("stream", False)
        if not stream:
            assert self._llm is not None
            completion = self._llm(
                prompt=prompt, repeat_penalty=repeat_penalty, **generate_config
            )

            return completion
        else:
            return generator_wrapper(prompt, repeat_penalty, generate_config)

    def create_embedding(self, input: Union[str, List[str]]) -> Embedding:
        assert self._llm is not None
        embedding = self._llm.create_embedding(input)
        return embedding


class LlamaCppChatModel(LlamaCppModel, ChatModelMixin):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        llamacpp_model_config: Optional[LlamaCppModelConfig] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_spec,
            quantization,
            model_path,
            llamacpp_model_config,
        )

    @classmethod
    def match(cls, llm_family: LLMFamilyV1, llm_spec: LLMSpecV1) -> bool:
        if llm_spec.model_format != "ggmlv3":
            return False
        if (
            "chatglm" in llm_family.model_name
            or llm_family.model_name in CTRANSFORMERS_SUPPORTED_MODEL
        ):
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True

    def _sanitize_generate_config(
        self, generate_config: Optional[LlamaCppGenerateConfig]
    ) -> LlamaCppGenerateConfig:
        generate_config = super()._sanitize_generate_config(generate_config)
        if self.model_family.prompt_style and self.model_family.prompt_style.stop:
            generate_config["stop"] = self.model_family.prompt_style.stop
        return generate_config

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[LlamaCppGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        assert self.model_family.prompt_style is not None
        prompt_style = self.model_family.prompt_style.copy()
        if system_prompt:
            prompt_style.system_prompt = system_prompt

        chat_history = chat_history or []
        assert prompt_style is not None
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
