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
from typing import TYPE_CHECKING, Iterator, Optional, Sequence, TypedDict, Union

if TYPE_CHECKING:
    from ctransformers import AutoConfig

from ....types import Completion, CompletionChunk
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from .ctransformers_util import generate_stream

logger = logging.getLogger(__name__)

# all supported models for Ctransformers with their model type.
# Please Strictly follows this name format when inputting new model to model_family.
MODEL_TYPE_FOR_CTRANSFORMERS = {
    "gpt-2": "gpt2",
    "gpt-j": "gptj",
    "gpt4all-j": "gptj",
    "gpt-neox": "gpt_neox",
    "stablelm": "gpt_neox",
    "llama": "llama",
    "llama-2": "llama",
    "mpt": "mpt",
    "dolly-v2": "dolly-v2",
    "replit": "replit",
    "starcoder": "starcoder",
    "starchat": "starcoder",
    "falcon": "falcon",
}

# these two constants subjects to change for future development and ctransformers updates.
CTRANSFORMERS_SUPPORTED_MODEL = ["starcoder", "gpt-2"]

CTRANSFORMERS_GPU_SUPPORT = ["llama", "llama-2", "mpt", "falcon"]

SIZE_TO_GPU_LAYERS = {
    3: 26,
    7: 32,
    13: 40,
    30: 60,
    65: 80,
}


class CtransformersModelConfig(TypedDict, total=False):
    n_ctx: int
    n_gpu_layers: int


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


def _has_cuda_device():
    from xorbits._mars.resource import cuda_count

    return cuda_count() > 0


class CtransformersModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        ctransformers_model_config: Optional[CtransformersModelConfig],
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)

        self._model_type = None
        closest_size = min(
            SIZE_TO_GPU_LAYERS.keys(),
            key=lambda x: abs(x - model_spec.model_size_in_billions),
        )

        self._model_family = model_family
        self._model_uid = model_uid
        self._llm = None

        self._gpu_layers = SIZE_TO_GPU_LAYERS[closest_size]
        self._ctransformer_model_config = self._sanitize_model_config(
            model_path, ctransformers_model_config
        )

    def _sanitize_model_config(
        self, model_path, ctransformers_model_config: Optional[CtransformersModelConfig]
    ) -> "AutoConfig":
        try:
            from ctransformers import AutoConfig, Config
        except ImportError:
            error_message = (
                "Failed to import module 'ctransformers - AutoConfig and Config'"
            )

            installation_guide = [
                f"Please make sure 'ctransformers' is installed.",
                f"You can install it by checking out the repository for command:"
                f"https://github.com/marella/ctransformers",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        # if the model have customized config, we update it.
        model_config_ret = Config()
        potential_gpu_layers = None
        if ctransformers_model_config:
            potential_context_length = ctransformers_model_config.pop("n_ctx", None)
            potential_gpu_layers = ctransformers_model_config.pop("n_gpu_layers", None)

            model_config_ret.context_length = potential_context_length
            model_config_ret.gpu_layers = potential_gpu_layers

        # if user does not define gpu layers, we have to set it with our system if applicable.
        if potential_gpu_layers is None:
            if self._model_family.model_name not in CTRANSFORMERS_GPU_SUPPORT:
                model_config_ret.gpu_layers = -1
            elif self._is_darwin_and_apple_silicon():
                model_config_ret.gpu_layers = 1
            elif _has_cuda_device():
                model_config_ret.gpu_layers = self._gpu_layers

        return AutoConfig(model_config_ret)

    def _sanitize_generate_config(
        self,
        ctransformers_generate_config: Optional[CtransformersGenerateConfig],
    ) -> CtransformersGenerateConfig:
        # if the input config is not None, we try to copy the selected attributes to the ctransformersGenerateConfig.
        if ctransformers_generate_config is None:
            ctransformers_generate_config = CtransformersGenerateConfig()

        # for our system, the threads will have to be set to 4
        # all other parameters, if not specified, will be set to default when generate.
        ctransformers_generate_config.setdefault("threads", 4)

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
        if llm_family.model_name not in CTRANSFORMERS_SUPPORTED_MODEL:
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

        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        max_new_tokens = generate_config.pop("max_tokens", None)

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
