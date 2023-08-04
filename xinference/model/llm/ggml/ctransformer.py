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
from typing import Iterator, Optional, Sequence, TypedDict, Union

from ctransformers import AutoConfig

from xinference.model.llm.ggml.ctransformers_util import generate_stream
from xinference.types import Completion, CompletionChunk

from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from .llamacpp import SIZE_TO_GPU_LAYERS

logger = logging.getLogger(__name__)


# class AutoConfig(TypedDict, total=False):
#     top_k: int
#     top_p: float
#     temperature: float
#     repetition_penalty: float
#     last_n_tokens: float
#     seed: int
#     max_new_tokens: int
#     stop: List[str]
#     stream: bool
#     reset: bool
#     batch_size: int
#     threads: int
#     context_length: int
#     gpu_layers: int

# all supported models for Ctransformers with their model type.
model_type_for_ctransformer = {
    "GPT-2": "gpt2",
    "GPT-J": "gptj",
    "GPT4All-J": "gptj",
    "GPT-NeoX": "gpt_neox",
    "StableLM": "gpt_neox",
    "LLaMA": "llama",
    "LLaMA-2": "llama",
    "MPT": "mpt",
    "Dolly-V2": "dolly-v2",
    "Replit": "replit",
    "StarCoder": "starcoder",
    "StarChat": "starcoder",
    "Falcon": "falcon",
}


class CtransformerGenerateConfig(TypedDict, total=False):
    max_new_tokens: Optional[int]
    top_k: Optional[int]
    top_p: Optional[float]
    temperature: Optional[float]
    repetition_penalty: Optional[float]
    last_n_tokens: Optional[int]
    seed: Optional[int]
    batch_size: Optional[int]
    threads: Optional[int]
    stop: Optional[Sequence[str]]
    stream: Optional[bool]
    reset: Optional[bool]


class CtransformerModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        ctransformerModelConfig: Optional[AutoConfig] = None,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)

        self._model_type = None
        closest_size = min(
            SIZE_TO_GPU_LAYERS.keys(),
            key=lambda x: abs(x - model_spec.model_size_in_billions),
        )
        self._gpu_layers = SIZE_TO_GPU_LAYERS[closest_size]
        self._ctransformer_model_config: AutoConfig = self._sanitize_model_config(
            model_path, ctransformerModelConfig
        )
        self._model_family = model_family
        self._llm = None

    def _sanitize_model_config(
        self, model_path, ctransformerModelConfig: Optional[AutoConfig]
    ) -> AutoConfig:
        if ctransformerModelConfig is None:
            ctransformerModelConfig = AutoConfig.from_pretrained(
                model_path,
                local_files_only=False,
            )

        return ctransformerModelConfig

    def _sanitize_generate_config(
        self,
        ctransformerGenerateConfig: Optional[CtransformerGenerateConfig],
    ) -> CtransformerGenerateConfig:
        if ctransformerGenerateConfig is None:
            ctransformerGenerateConfig = CtransformerGenerateConfig()
        ctransformerGenerateConfig.setdefault("top_k", 40)
        ctransformerGenerateConfig.setdefault("top_p", 0.95)
        ctransformerGenerateConfig.setdefault("temperature", 0.8)
        ctransformerGenerateConfig.setdefault("repetition_penalty", 1.1)
        ctransformerGenerateConfig.setdefault("last_n_tokens", 64)
        ctransformerGenerateConfig.setdefault("seed", -1)
        ctransformerGenerateConfig.setdefault("batch_size", 8)
        ctransformerGenerateConfig.setdefault("threads", -1)
        ctransformerGenerateConfig.setdefault("stop", None)
        ctransformerGenerateConfig.setdefault("stream", None)
        ctransformerGenerateConfig.setdefault("reset", True)

        return ctransformerGenerateConfig

    def load(self):
        try:
            from ctransformers import AutoModelForCausalLM
        except ImportError:
            error_message = "Failed to import module 'ctransformers'"
            if self._is_darwin_and_apple_silicon():
                system = "Metal"
            else:
                system = "CUDA"

            installation_guide = [
                f"Please make sure 'ctransformers' is installed and {system} accelerator is provided.",
                f"You can install it by checking out the repository for command for {system} platform:"
                f"https://github.com/marella/ctransformers",
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

        self._model_type = self._determine_model_type()
        self._llm = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=model_path,
            model_type=self._model_type,
            config=self._ctransformer_model_config,
        )

    @classmethod
    def match(cls, llm_family: LLMFamilyV1, llm_spec: LLMSpecV1) -> bool:
        if llm_spec.model_format != "ggmlv3":
            return False
        if "StarCoder" not in llm_family.model_name:
            return False
        if "generate" not in llm_family.model_ability:
            return False
        return True

    def _determine_model_type(self):
        if self._model_family.model_name not in model_type_for_ctransformer:
            raise ValueError(
                "The current model is not supported, check your model name. "
            )
        return model_type_for_ctransformer[self._model_family.model_name]

    def generate(
        self, prompt: str, generate_config: CtransformerGenerateConfig
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        def generator_wrapper(
            _prompt: str,
            _generate_config: CtransformerGenerateConfig,
        ) -> Iterator[CompletionChunk]:
            assert self._llm is not None
            for _completion_chunk, _ in generate_stream(
                model=self._llm, prompt=_prompt, **_generate_config
            ):
                yield _completion_chunk

        generate_config = self._sanitize_generate_config(generate_config)

        logger.error(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        stream_or_not = generate_config.get("stream", False)
        if stream_or_not:
            return generator_wrapper(_prompt=prompt, _generate_config=generate_config)
        else:
            assert self._llm is not None
            for completion_chunk, completion_usage in generate_stream(
                self._llm, prompt=prompt, **generate_config
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

            logger.error("Generated", completion, generate_config)

            return completion
