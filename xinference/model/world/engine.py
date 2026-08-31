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

from importlib import metadata
from typing import TYPE_CHECKING, Tuple, Union

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

from ..utils import has_cuda_device
from .engine_family import SUPPORTED_ENGINES, WorldEngineModel
from .model import AstraModel, HYWorldPlayModel, MatrixGameModel

if TYPE_CHECKING:
    from .core import WorldModelFamilyV1


class PyTorchWorldEngineModel(WorldEngineModel):
    required_libs = ("torch",)

    @classmethod
    def check_host(cls) -> Union[bool, Tuple[bool, str]]:
        if not has_cuda_device():
            return False, "The PyTorch world engine requires an NVIDIA CUDA GPU"
        return True


class PyTorchMatrixGameModel(MatrixGameModel, PyTorchWorldEngineModel):
    @classmethod
    def match(cls, model_family: "WorldModelFamilyV1") -> bool:
        return model_family.model_family == "Matrix-Game-3.0"


class PyTorchHYWorldPlayModel(HYWorldPlayModel, PyTorchWorldEngineModel):
    @classmethod
    def check_host(cls) -> Union[bool, Tuple[bool, str]]:
        host_result = super().check_host()
        if host_result is not True:
            return host_result

        try:
            torch_version = metadata.version("torch")
        except metadata.PackageNotFoundError:
            return False, "HY-WorldPlay requires host torch>=2.6.0"

        try:
            if Version(torch_version) < Version("2.6.0"):
                return (
                    False,
                    "HY-WorldPlay requires host torch>=2.6.0; "
                    f"found torch {torch_version}",
                )
        except InvalidVersion:
            return False, f"HY-WorldPlay cannot validate host torch {torch_version!r}"

        try:
            torchvision_version = metadata.version("torchvision")
            torchvision_requirements = metadata.requires("torchvision") or []
        except metadata.PackageNotFoundError:
            return False, "HY-WorldPlay requires host torchvision"

        for requirement_text in torchvision_requirements:
            requirement = Requirement(requirement_text)
            if requirement.name.lower() != "torch":
                continue
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            if not requirement.specifier.contains(torch_version, prereleases=True):
                return (
                    False,
                    "HY-WorldPlay requires a compatible host torch/torchvision "
                    f"pair; found torch {torch_version} and torchvision "
                    f"{torchvision_version} (requires torch{requirement.specifier})",
                )
            break

        return True

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
