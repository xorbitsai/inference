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

from xinference.types import Completion, CompletionChunk

from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from .ctransformers_util import generate_stream
from .llamacpp import SIZE_TO_GPU_LAYERS

logger = logging.getLogger(__name__)

# all supported models for Ctransformers with their model type.
# Please Strictly follows this name format when inputting new model to model_family.
MODEL_TYPE_FOR_CTRANSFORMERS = {
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


class CtransformersModelConfig(TypedDict, total=False):
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    last_n_tokens: int
    seed: int
    batch_size: int
    threads: int
    max_new_tokens: int
    stop: Optional[Sequence[str]]
    stream: bool
    reset: bool
    context_length: int
    gpu_layers: int


class CtransformersGenerateConfig(TypedDict, total=False):
    max_tokens: Optional[int]
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


class CtransformersModel(LLM):
    try:
        from ctransformers import AutoConfig
    except ImportError:
        error_message = "Failed to import module 'ctransformers - AutoConfig'"
        installation_guide = [
            "Please make sure 'ctransformers' is installed, You can install it by checking out the repository for "
            "command: https://github.com/marella/ctransformers",
        ]

    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        ctransformers_Model_Config: Optional[CtransformersModelConfig],
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)

        self._model_type = None
        closest_size = min(
            SIZE_TO_GPU_LAYERS.keys(),
            key=lambda x: abs(x - model_spec.model_size_in_billions),
        )
        self._gpu_layers = SIZE_TO_GPU_LAYERS[closest_size]
        self._ctransformer_model_config = self._sanitize_model_config(
            model_path, ctransformers_Model_Config
        )
        self._model_family = model_family
        self._model_uid = model_uid
        self._llm = None

    def _sanitize_model_config(
        self, model_path, ctransformers_model_config: Optional[CtransformersModelConfig]
    ) -> "AutoConfig":
        try:
            from ctransformers import AutoConfig
        except ImportError:
            error_message = "Failed to import module 'ctransformers - AutoConfig'"
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

        if ctransformers_model_config is None:
            ctransformers_model_config = AutoConfig.from_pretrained(
                model_path,
                local_files_only=False,
            )

        return ctransformers_model_config

    def _sanitize_generate_config(
        self,
        ctransformers_generate_config: Optional[CtransformersGenerateConfig],
    ) -> CtransformersGenerateConfig:
        # if the input config is not None, we try to copy the selected attributes to the ctransformersGenerateConfig.
        if ctransformers_generate_config is None:
            ctransformers_generate_config = CtransformersGenerateConfig()

        ctransformers_generate_config.setdefault("top_k", 40)
        ctransformers_generate_config.setdefault("top_p", 0.95)
        ctransformers_generate_config.setdefault("temperature", 0.8)
        ctransformers_generate_config.setdefault("repetition_penalty", 1.1)
        ctransformers_generate_config.setdefault("last_n_tokens", 64)
        ctransformers_generate_config.setdefault("seed", -1)
        ctransformers_generate_config.setdefault("batch_size", 8)
        ctransformers_generate_config.setdefault("threads", -1)
        ctransformers_generate_config.setdefault("stop", None)
        ctransformers_generate_config.setdefault("stream", None)
        ctransformers_generate_config.setdefault("reset", True)

        return ctransformers_generate_config

    def load(self):
        try:
            from ctransformers import AutoModelForCausalLM
        except ImportError:
            error_message = "Failed to import module 'ctransformers'"

            installation_guide = [
                f"Please make sure 'ctransformers' is installed.",
                f"You can install it by checking out the repository for command."
                f"https://github.com/marella/ctransformers",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        model_path = os.path.join(
            self.model_path,
            self.model_spec.model_file_name_template.format(
                quantization=self.quantization
            ),
        )

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
        if self._model_family.model_name not in MODEL_TYPE_FOR_CTRANSFORMERS:
            raise ValueError(
                f"The current model {self._model_family.model_name} is not supported, check your model name. "
            )
        return MODEL_TYPE_FOR_CTRANSFORMERS[self._model_family.model_name]

    def generate(
        self, prompt: str, generate_config_raw: CtransformersGenerateConfig
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        def generator_wrapper(
            _prompt: str,
            _max_new_tokens: Union[int, None],
            _generate_config: CtransformersGenerateConfig,
        ) -> Iterator[CompletionChunk]:
            assert self._model_uid is not None
            for _completion_chunk, _ in generate_stream(
                model=self._model_uid,
                model_ref=self._llm,
                prompt=_prompt,
                max_new_tokens=_max_new_tokens,
                **_generate_config,
            ):
                yield _completion_chunk

        generate_config = self._sanitize_generate_config(generate_config_raw)
        max_new_tokens = generate_config.pop("max_tokens", None)

        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        stream_or_not = generate_config.get("stream", False)
        if stream_or_not:
            return generator_wrapper(
                _prompt=prompt,
                _max_new_tokens=max_new_tokens,
                _generate_config=generate_config,
            )
        else:
            assert self.model_uid is not None
            completion_chunk = None
            completion_usage = None
            for completion_chunk, completion_usage in generate_stream(
                model=self.model_uid,
                model_ref=self._llm,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                **generate_config,
            ):
                pass

            assert completion_chunk is not None
            assert completion_usage is not None

            completion = Completion(
                id=completion_chunk["id"],
                object=completion_chunk["object"],
                created=completion_chunk["created"],
                model=completion_chunk["model"],
                choices=completion_chunk["choices"],
                usage=completion_usage,
            )

            logger.debug(
                "Generated, completion: %s, generate config: %s",
                completion,
                generate_config,
            )

            return completion
