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
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from ..utils import has_cuda_device
from .docanalyze.mineru25 import Mineru2_5Model
from .engine_family import SUPPORTED_ENGINES, ImageEngineModel
from .fibo2 import Fibo2ImageModel
from .flux3 import Flux3Model
from .hidream_o1 import HIDREAM_O1_MODEL_NAMES, HiDreamO1Model
from .iris import IrisModel
from .ming_image import MingImageModel
from .minimax_h3 import MiniMaxH3ImageModel
from .moss_image import MOSSImageModel
from .sensenova_u1 import SenseNovaU1Model
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
        return model_family.model_family not in (
            "ocr",
            "sensenova_u1",
            "docanalyze",
            "ming_image",
            "iris",
            "moss_image",
            "fibo2",
            "flux3",
            "minimax_h3_image",
        )


class Fibo2DiffusersImageModel(Fibo2ImageModel, ImageEngineModel):
    engine_model_format = "diffusers"
    engine_quantization = "none"
    required_libs = ("torch", "diffusers", "transformers")

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_family == "fibo2"


class MOSSDiffusersImageModel(MOSSImageModel, ImageEngineModel):
    engine_model_format = "diffusers"
    engine_quantization = "none"
    required_libs = ("torch", "diffusers", "transformers")

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_family == "moss_image"


class Flux3ImageModel(Flux3Model, ImageEngineModel):
    engine_model_format = "diffusers"
    engine_quantization = "none"
    required_libs = ("torch", "transformers", "diffusers")

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_family == "flux3"


class MiniMaxH3DiffusersImageModel(MiniMaxH3ImageModel, ImageEngineModel):
    engine_model_format = "diffusers"
    engine_quantization = "none"
    required_libs = ("torch", "transformers", "diffusers")

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        from importlib.metadata import PackageNotFoundError, version

        from packaging.requirements import Requirement
        from packaging.version import InvalidVersion

        result = super().check_lib()
        if result is not True:
            return result
        try:
            installed = version("diffusers")
            compatible = installed in Requirement(cls._diffusers_requirement).specifier
        except (PackageNotFoundError, InvalidVersion) as exc:
            return False, f"Cannot determine the installed Diffusers version: {exc}"
        if not compatible:
            return (
                False,
                f"{cls._display_name} requires {cls._diffusers_requirement}; "
                f"found diffusers=={installed}.",
            )
        return True

    @classmethod
    def preflight(
        cls,
        *,
        python_executable: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        if python_executable is not None:
            import json
            import os
            import subprocess

            # Match the model subprocess's interpreter and allocated devices;
            # importing the worker's Diffusers/Torch can give a different result.
            process = subprocess.run(
                [
                    python_executable,
                    "-c",
                    "import json, sys; "
                    "from xinference.model.image.engine import MiniMaxH3DiffusersImageModel; "
                    "MiniMaxH3DiffusersImageModel.preflight(**json.load(sys.stdin))",
                ],
                input=json.dumps(
                    {
                        "device": kwargs.get("device"),
                        "cpu_offload": kwargs.get("cpu_offload", True),
                    }
                ),
                env={**os.environ, **(env or {})},
                capture_output=True,
                text=True,
                timeout=60,
            )
            if process.returncode:
                raise ValueError(
                    f"{cls._display_name} runtime preflight failed: {process.stderr.strip()}"
                )
            return
        result = cls.check_lib()
        if result is not True:
            reason = (
                result[1]
                if isinstance(result, tuple)
                else "Missing runtime dependencies"
            )
            raise ImportError(
                f"{cls._display_name} requires {cls._diffusers_requirement}. {reason}"
            )
        cls._get_runtime(kwargs.get("device"), kwargs.get("cpu_offload", True))

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_family == "minimax_h3_image"


class TransformersSenseNovaU1ImageModel(SenseNovaU1Model, ImageEngineModel):
    engine_model_format = "pytorch"
    engine_quantization = "none"
    required_libs = ("torch", "transformers")

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_family == "sensenova_u1"


class HiDreamO1ImageModel(HiDreamO1Model, ImageEngineModel):
    engine_model_format = "pytorch"
    engine_quantization = "none"
    required_libs = ("transformers", "diffusers", "einops")

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return (
            model_family.model_family == "hidream_o1"
            and model_family.model_name in HIDREAM_O1_MODEL_NAMES
        )


class TransformersIrisImageModel(IrisModel, ImageEngineModel):
    engine_model_format = "pytorch"
    engine_quantization = "none"
    required_libs = ("torch", "torchvision", "transformers", "omegaconf", "safetensors")

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_family == "iris"


class MingImageEngineModel(MingImageModel, ImageEngineModel):
    engine_model_format = "pytorch"
    engine_quantization = "none"
    required_libs = ("torch", "transformers", "diffusers")

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_family == "ming_image"


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


class TransformersMinerUImageModel(Mineru2_5Model, ImageEngineModel):
    required_libs = ("torch", "transformers")

    @classmethod
    def match(cls, model_family):
        return model_family.model_name == "MinerU2.5"


class VLLMMinerUImageModel(Mineru2_5Model, ImageEngineModel):
    required_libs = ("vllm",)

    @classmethod
    def match(cls, model_family):
        return (
            platform.system() == "Linux"
            and has_cuda_device()
            and model_family.model_name == "MinerU2.5"
        )


def register_builtin_image_engines() -> None:
    SUPPORTED_ENGINES["diffusers"] = [
        MiniMaxH3DiffusersImageModel,
        Flux3ImageModel,
        HiDreamO1ImageModel,
        MingImageEngineModel,
        MOSSDiffusersImageModel,
        Fibo2DiffusersImageModel,
        DiffusersImageModel,
    ]
    SUPPORTED_ENGINES["transformers"] = [
        TransformersSenseNovaU1ImageModel,
        TransformersMinerUImageModel,
        TransformersIrisImageModel,
    ]
    SUPPORTED_ENGINES["vLLM"] = [
        VLLMImageModel,
        VLLMMinerUImageModel,
    ]
    SUPPORTED_ENGINES["SGLang"] = [SGLangImageModel]
