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
from typing import Any, List, Optional

from ..._compat import (
    ROOT_KEY,
    ErrorWrapper,
    Literal,
    Protocol,
    StrBytes,
    ValidationError,
    load_str_bytes,
)
from ..custom import ModelRegistry
from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class CustomAudioModelFamilyV2(AudioModelFamilyV2):
    version: Literal[2] = 2
    model_id: Optional[str]  # type: ignore
    model_revision: Optional[str]  # type: ignore
    model_uri: Optional[str]

    @classmethod
    def parse_raw(
        cls: Any,
        b: StrBytes,
        *,
        content_type: Optional[str] = None,
        encoding: str = "utf8",
        proto: Protocol = None,
        allow_pickle: bool = False,
    ) -> AudioModelFamilyV2:
        # See source code of BaseModel.parse_raw
        try:
            obj = load_str_bytes(
                b,
                proto=proto,
                content_type=content_type,
                encoding=encoding,
                allow_pickle=allow_pickle,
                json_loads=cls.__config__.json_loads,
            )
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            raise ValidationError([ErrorWrapper(e, loc=ROOT_KEY)], cls)

        audio_spec: AudioModelFamilyV2 = cls.parse_obj(obj)

        # check model_family
        if audio_spec.model_family is None:
            raise ValueError(
                f"You must specify `model_family` when registering custom Audio models."
            )
        assert isinstance(audio_spec.model_family, str)
        return audio_spec


UD_AUDIOS: List[CustomAudioModelFamilyV2] = []


class AudioModelRegistry(ModelRegistry):
    model_type = "audio"

    def __init__(self):
        from . import BUILTIN_AUDIO_MODELS

        super().__init__()
        self.models = UD_AUDIOS
        self.builtin_models = list(BUILTIN_AUDIO_MODELS.keys())


def get_user_defined_audios() -> List[CustomAudioModelFamilyV2]:
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("audio")
    return registry.get_custom_models()


def register_audio(model_spec: CustomAudioModelFamilyV2, persist: bool):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("audio")
    registry.register(model_spec, persist)


def unregister_audio(model_name: str, raise_error: bool = True):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("audio")
    registry.unregister(model_name, raise_error)
