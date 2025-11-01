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
from typing import TYPE_CHECKING, List, Optional

from ..._compat import (
    Literal,
)
from ..custom import ModelRegistry
from .core import VideoModelFamilyV2

logger = logging.getLogger(__name__)


class CustomVideoModelFamilyV2(VideoModelFamilyV2):
    version: Literal[2] = 2
    model_id: Optional[str]  # type: ignore
    model_revision: Optional[str]  # type: ignore
    model_uri: Optional[str]


if TYPE_CHECKING:
    from typing import TypeVar

    _T = TypeVar("_T", bound="CustomVideoModelFamilyV2")


class VideoModelRegistry(ModelRegistry):
    model_type = "video"

    def __init__(self):
        super().__init__()

    def get_user_defined_models(self) -> List["CustomVideoModelFamilyV2"]:
        return self.get_custom_models()


video_registry = VideoModelRegistry()


def register_video(model_spec: CustomVideoModelFamilyV2, persist: bool = True):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("video")
    registry.register(model_spec, persist)


def unregister_video(model_name: str, raise_error: bool = True):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("video")
    registry.unregister(model_name, raise_error)


def get_registered_videos() -> List[CustomVideoModelFamilyV2]:
    """
    Get all video families registered in the registry (both user-defined and editor-defined).
    This excludes hardcoded builtin models.
    """
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("video")
    return registry.get_custom_models()
