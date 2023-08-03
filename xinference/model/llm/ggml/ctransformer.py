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
from typing import TYPE_CHECKING, Iterator, Optional, Sequence, TypedDict, Union

from ctransformers import AutoConfig

from xinference.types import Completion, CompletionChunk

from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from .ctransformers_util import generate_stream

if TYPE_CHECKING:
    from ... import ModelSpec

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
        model_spec: "ModelSpec",
        model_path: str,
        model_type: str,
        model_file: Optional[str],
        ctransformerModelConfig: Optional[AutoConfig] = None,
    ):
        super().__init__(model_uid, model_spec)

        # closest_size = min(
        #     SIZE_TO_GPU_LAYERS.keys(),
        #     key=lambda x: abs(x - model_spec.model_size_in_billions),
        # )
        # self._gpu_layers = SIZE_TO_GPU_LAYERS[closest_size]
        self._model_path = model_path
        self._ctransformer_model_config: AutoConfig = self._sanitize_model_config(
            model_path, ctransformerModelConfig
        )
        self._model_type = model_type
        self._model_file = model_file
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

        self._llm = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=self._model_path,
            model_type=self._model_type,
            model_file=self._model_file,
            config=self._ctransformer_model_config,
        )

    @classmethod
    def match(cls, llm_family: LLMFamilyV1, llm_spec: LLMSpecV1) -> bool:
        if llm_spec.model_format != "ggmlv3":
            return False
        if llm_spec.model_id not in ["TheBloke/starcoder-GGML"]:
            return False
        if "chatglm" in llm_family.model_name:
            return False
        if "generate" not in llm_family.model_ability:
            return False
        return True

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

        stream_or_not = generate_config.get("stream", False)
        if stream_or_not:
            return generator_wrapper(_prompt=prompt, _generate_config=generate_config)
        else:
            for completion_chunk, completion_usage in generate_stream(
                self._model, prompt=prompt, **generate_config
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
