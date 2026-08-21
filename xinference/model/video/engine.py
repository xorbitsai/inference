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

import importlib.util
import platform
import sys
from typing import TYPE_CHECKING, Tuple, Union

from .diffusers import DiffusersVideoModel
from .engine_family import SUPPORTED_ENGINES, VideoEngineModel
from .mlx_video import MLXVideoModel

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


MLX_VIDEO_MODEL_NAMES = {
    "LTX-2-distilled",
    "LTX-2-dev",
    "LTX-2.3-distilled",
    "LTX-2.3-dev",
    "Wan2.1-1.3B",
    "Wan2.1-14B",
    "Wan2.2-A14B",
    "Wan2.2-i2v-A14B",
    "Wan2.2-ti2v-5B",
}


class MLXVideoEngineModel(MLXVideoModel, VideoEngineModel):
    engine_model_format = "mlx"
    engine_quantization = "none"
    required_libs = ("mlx", "mlx_video")

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        if not cls._is_apple_silicon():
            return False, "The MLX video engine requires Apple Silicon"
        if sys.version_info < (3, 11):
            return False, "Blaizzy/mlx-video requires Python 3.11 or newer"
        if importlib.util.find_spec("mlx") is None:
            return False, "Library 'mlx' is not installed"
        try:
            wan_module = importlib.util.find_spec("mlx_video.models.wan_2.generate")
            ltx_module = importlib.util.find_spec("mlx_video.models.ltx_2.generate")
        except (ImportError, ModuleNotFoundError):
            wan_module = ltx_module = None
        if wan_module is None or ltx_module is None:
            return (
                False,
                "Blaizzy/mlx-video is not installed; the unrelated PyPI package "
                "with the same name is not compatible",
            )
        return True

    @staticmethod
    def _is_apple_silicon() -> bool:
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    @classmethod
    def match(cls, model_family: "VideoModelFamilyV2") -> bool:
        return (
            cls._is_apple_silicon()
            and model_family.model_name in MLX_VIDEO_MODEL_NAMES
            and (getattr(model_family, "engine", "") or "").lower() == "mlx"
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "VideoModelFamilyV2") -> bool:
        return model_family.model_name in MLX_VIDEO_MODEL_NAMES


def register_builtin_video_engines() -> None:
    # The first registered engine remains the default for models with several
    # runtimes, preserving the existing diffusers behavior for Wan models.
    SUPPORTED_ENGINES["diffusers"] = [DiffusersVideoEngineModel]
    SUPPORTED_ENGINES["MLX"] = [MLXVideoEngineModel]
