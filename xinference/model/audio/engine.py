# Copyright 2022-2026 XProbe Inc.
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

import platform
from typing import TYPE_CHECKING

from ..utils import has_cuda_device
from .engine_family import SUPPORTED_ENGINES, AudioEngineModel
from .qwen3_asr import Qwen3ASRModel
from .vllm import VLLMQwen3ASRModel

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2


class TransformersQwen3ASRAudioModel(Qwen3ASRModel, AudioEngineModel):
    required_libs = ("qwen_asr",)

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_family == "qwen3_asr"


class VLLMQwen3ASRAudioModel(VLLMQwen3ASRModel, AudioEngineModel):
    required_libs = ("qwen_asr", "vllm")

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        if platform.system() != "Linux":
            return False
        if not has_cuda_device():
            return False
        return model_family.model_family == "qwen3_asr"


def register_builtin_audio_engines() -> None:
    # the first registered engine is the default one for a model
    SUPPORTED_ENGINES["transformers"] = [TransformersQwen3ASRAudioModel]
    SUPPORTED_ENGINES["vLLM"] = [VLLMQwen3ASRAudioModel]
