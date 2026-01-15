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

from typing import TYPE_CHECKING

from .engine_family import SUPPORTED_ENGINES, ImageEngineModel
from .stable_diffusion.core import DiffusionModel

if TYPE_CHECKING:
    from .core import ImageModelFamilyV2


class DiffusersImageModel(DiffusionModel, ImageEngineModel):
    engine_model_format = "diffusers"
    engine_quantization = "none"
    required_libs = ("diffusers",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_family != "ocr"


class VLLMImageModel(ImageEngineModel):
    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        _ = model_family
        return False

    @classmethod
    def check_lib(cls):
        return (
            False,
            "Engine vLLM is not compatible with current image model or environment",
        )


class SGLangImageModel(ImageEngineModel):
    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        _ = model_family
        return False

    @classmethod
    def check_lib(cls):
        return (
            False,
            "Engine SGLang is not compatible with current image model or environment",
        )


def register_builtin_image_engines() -> None:
    SUPPORTED_ENGINES["diffusers"] = [DiffusersImageModel]
    SUPPORTED_ENGINES["vLLM"] = [VLLMImageModel]
    SUPPORTED_ENGINES["SGLang"] = [SGLangImageModel]
