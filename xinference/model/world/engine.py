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

from typing import TYPE_CHECKING, Tuple, Union

from ..utils import has_cuda_device
from .engine_family import SUPPORTED_ENGINES, WorldEngineModel
from .model import AstraModel, HYWorldPlayModel, MatrixGameModel

if TYPE_CHECKING:
    from .core import WorldModelFamilyV1


class PyTorchWorldEngineModel(WorldEngineModel):
    required_libs = ("torch",)

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        dependency_check = super().check_lib()
        if dependency_check is not True:
            return dependency_check
        if not has_cuda_device():
            return False, "The PyTorch world engine requires an NVIDIA CUDA GPU"
        return True


class PyTorchMatrixGameModel(MatrixGameModel, PyTorchWorldEngineModel):
    @classmethod
    def match(cls, model_family: "WorldModelFamilyV1") -> bool:
        return model_family.model_family == "Matrix-Game-3.0"


class PyTorchHYWorldPlayModel(HYWorldPlayModel, PyTorchWorldEngineModel):
    @classmethod
    def match(cls, model_family: "WorldModelFamilyV1") -> bool:
        return model_family.model_family == "HY-WorldPlay"


class PyTorchAstraModel(AstraModel, PyTorchWorldEngineModel):
    @classmethod
    def match(cls, model_family: "WorldModelFamilyV1") -> bool:
        return model_family.model_family == "Astra"


def register_builtin_world_engines() -> None:
    SUPPORTED_ENGINES.clear()
    # Registration order defines the default engine. More world runtimes can be
    # added without changing the public generation API or model actor contract.
    SUPPORTED_ENGINES["PyTorch"] = [
        PyTorchMatrixGameModel,
        PyTorchHYWorldPlayModel,
        PyTorchAstraModel,
    ]
