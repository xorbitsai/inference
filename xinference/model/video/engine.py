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

from typing import TYPE_CHECKING

from .diffusers import DiffusersVideoModel
from .engine_family import SUPPORTED_ENGINES, VideoEngineModel

if TYPE_CHECKING:
    from .core import VideoModelFamilyV2


class DiffusersVideoEngineModel(DiffusersVideoModel, VideoEngineModel):
    engine_model_format = "diffusers"
    engine_quantization = "none"
    required_libs = ("diffusers",)

    @classmethod
    def match(cls, model_family: "VideoModelFamilyV2") -> bool:
        engine = getattr(model_family, "engine", None)
        return not engine or engine.lower() == "diffusers"

    @classmethod
    def is_model_family_supported(cls, model_family: "VideoModelFamilyV2") -> bool:
        return cls.match(model_family)


def register_builtin_video_engines() -> None:
    SUPPORTED_ENGINES["diffusers"] = [DiffusersVideoEngineModel]
