# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
from .engine_family import SUPPORTED_ENGINES, ImageEngineModel
from .sglang.core import SGLANG_SUPPORTED_IMAGE_MODELS, SGLangDiffusionModel
from .stable_diffusion.core import DiffusionModel
from .vllm.core import VLLM_SUPPORTED_IMAGE_MODELS, VLLMDiffusionModel

if TYPE_CHECKING:
    from .core import ImageModelFamilyV2


class DiffusersImageModel(DiffusionModel, ImageEngineModel):
    engine_model_format = "diffusers"
    engine_quantization = "none"
    required_libs = ("diffusers",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_family != "ocr"


class VLLMImageModel(VLLMDiffusionModel, ImageEngineModel):
    engine_model_format = "diffusers"
    engine_quantization = "none"
    required_libs = ("vllm_omni",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        if platform.system() != "Linux":
            return False
        if not has_cuda_device():
            return False
        return model_family.model_name in VLLM_SUPPORTED_IMAGE_MODELS


class SGLangImageModel(SGLangDiffusionModel, ImageEngineModel):
    engine_model_format = "diffusers"
    engine_quantization = "none"
    required_libs = ("sglang",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        if platform.system() != "Linux":
            return False
        if not has_cuda_device():
            return False
        return model_family.model_name in SGLANG_SUPPORTED_IMAGE_MODELS


def register_builtin_image_engines() -> None:
    SUPPORTED_ENGINES["diffusers"] = [DiffusersImageModel]
    SUPPORTED_ENGINES["vLLM"] = [VLLMImageModel]
    SUPPORTED_ENGINES["SGLang"] = [SGLangImageModel]
