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
from typing import TYPE_CHECKING, List, Optional, Sequence, TypedDict

from xinference.model.llm.core import Model

if TYPE_CHECKING:
    from ... import ModelSpec

logger = logging.getLogger(__name__)


class CtransformerModelConfig(TypedDict, total=False):
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    last_n_tokens: float
    seed: int
    max_new_tokens: int
    stop: List[str]
    stream: bool
    reset: bool
    batch_size: int
    threads: int
    context_length: int
    gpu_layers: int


class CtransformerGenerateConfig(TypedDict, total=False):
    tokens: Sequence[int]
    top_k: Optional[int]
    top_p: Optional[float]
    temperature: Optional[float]
    repetition_penalty: Optional[float]
    last_n_tokens: Optional[int]
    seed: Optional[int]
    batch_size: Optional[int]
    threads: Optional[int]
    reset: Optional[bool]


class CtransformerModel(Model):
    def __init__(
        self,
        model_uid: str,
        model_spec: "ModelSpec",
        model_path: str,
        ctransformerModelConfig: Optional[CtransformerModelConfig] = None,
    ):
        super().__init__(model_uid, model_spec)

        # closest_size = min(
        #     SIZE_TO_GPU_LAYERS.keys(),
        #     key=lambda x: abs(x - model_spec.model_size_in_billions),
        # )
        # self._gpu_layers = SIZE_TO_GPU_LAYERS[closest_size]
        self._model_path = model_path
        self._ctransformer_model_config: CtransformerModelConfig = (
            self._sanitize_model_config(ctransformerModelConfig)
        )
        self._llm = None

    def _sanitize_model_config(
        self, ctransformerModelConfig: Optional[CtransformerModelConfig]
    ) -> CtransformerModelConfig:
        if ctransformerModelConfig is None:
            ctransformerModelConfig = CtransformerModelConfig()
        ctransformerModelConfig.setdefault("top_k", 40)
        ctransformerModelConfig.setdefault("top_p", 0.95)
        ctransformerModelConfig.setdefault("temperature", 0.8)
        ctransformerModelConfig.setdefault("repetition_penalty", 1.1)
        ctransformerModelConfig.setdefault("last_n_tokens", 64)
        ctransformerModelConfig.setdefault("seed", -1)
        ctransformerModelConfig.setdefault("max_new_tokens", 256)
        # ctransformerModelConfig.setdefault("stop", None)
        ctransformerModelConfig.setdefault("stream", False)
        ctransformerModelConfig.setdefault("reset", True)
        ctransformerModelConfig.setdefault("batch_size", 8)
        ctransformerModelConfig.setdefault("threads", -1)
        ctransformerModelConfig.setdefault("context_length", -1)
        ctransformerModelConfig.setdefault("gpu_layers", 0)

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
        ctransformerGenerateConfig.setdefault("reset", True)

        return ctransformerGenerateConfig
