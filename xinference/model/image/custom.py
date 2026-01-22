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
import logging
from typing import List, Optional

from ..._compat import Literal
from ..custom import ModelRegistry
from .core import ImageModelFamilyV2

logger = logging.getLogger(__name__)


class CustomImageModelFamilyV2(ImageModelFamilyV2):
    version: Literal[2] = 2
    model_id: Optional[str]  # type: ignore
    model_revision: Optional[str]  # type: ignore
    model_uri: Optional[str]
    controlnet: Optional[List["CustomImageModelFamilyV2"]]


UD_IMAGES: List[CustomImageModelFamilyV2] = []


class ImageModelRegistry(ModelRegistry):
    model_type = "image"

    def __init__(self):
        from .core import BUILTIN_IMAGE_MODELS

        super().__init__()
        self.models = UD_IMAGES
        self.builtin_models = list(BUILTIN_IMAGE_MODELS.keys())


def get_user_defined_images() -> List[ImageModelFamilyV2]:
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("image")
    return registry.get_custom_models()


def register_image(model_spec: CustomImageModelFamilyV2, persist: bool):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("image")
    registry.register(model_spec, persist)
    if model_spec.model_ability and "ocr" in model_spec.model_ability:
        from .ocr.ocr_family import generate_engine_config_by_model_name

        generate_engine_config_by_model_name(model_spec)
    else:
        from .engine_family import generate_engine_config_by_model_name

        generate_engine_config_by_model_name(model_spec)


def unregister_image(model_name: str, raise_error: bool = True):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("image")
    registry.unregister(model_name, raise_error)
    from .engine_family import IMAGE_ENGINES
    from .ocr.ocr_family import OCR_ENGINES

    if model_name in OCR_ENGINES:
        del OCR_ENGINES[model_name]
    if model_name in IMAGE_ENGINES:
        del IMAGE_ENGINES[model_name]
