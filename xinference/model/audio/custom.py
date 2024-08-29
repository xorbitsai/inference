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
import os
from threading import Lock
from typing import Any, List, Optional

from ..._compat import (
    ROOT_KEY,
    ErrorWrapper,
    Protocol,
    StrBytes,
    ValidationError,
    load_str_bytes,
)
from ...constants import XINFERENCE_CACHE_DIR, XINFERENCE_MODEL_DIR
from .core import AudioModelFamilyV1

logger = logging.getLogger(__name__)

UD_AUDIO_LOCK = Lock()


class CustomAudioModelFamilyV1(AudioModelFamilyV1):
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
    ) -> AudioModelFamilyV1:
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

        audio_spec: AudioModelFamilyV1 = cls.parse_obj(obj)

        # check model_family
        if audio_spec.model_family is None:
            raise ValueError(
                f"You must specify `model_family` when registering custom Audio models."
            )
        assert isinstance(audio_spec.model_family, str)
        return audio_spec


UD_AUDIOS: List[CustomAudioModelFamilyV1] = []


def get_user_defined_audios() -> List[CustomAudioModelFamilyV1]:
    with UD_AUDIO_LOCK:
        return UD_AUDIOS.copy()


def register_audio(model_spec: CustomAudioModelFamilyV1, persist: bool):
    from ...constants import XINFERENCE_MODEL_DIR
    from ..utils import is_valid_model_name, is_valid_model_uri
    from . import BUILTIN_AUDIO_MODELS, MODELSCOPE_AUDIO_MODELS

    if not is_valid_model_name(model_spec.model_name):
        raise ValueError(f"Invalid model name {model_spec.model_name}.")

    model_uri = model_spec.model_uri
    if model_uri and not is_valid_model_uri(model_uri):
        raise ValueError(f"Invalid model URI {model_uri}.")

    with UD_AUDIO_LOCK:
        for model_name in (
            list(BUILTIN_AUDIO_MODELS.keys())
            + list(MODELSCOPE_AUDIO_MODELS.keys())
            + [spec.model_name for spec in UD_AUDIOS]
        ):
            if model_spec.model_name == model_name:
                raise ValueError(
                    f"Model name conflicts with existing model {model_spec.model_name}"
                )

        UD_AUDIOS.append(model_spec)

    if persist:
        persist_path = os.path.join(
            XINFERENCE_MODEL_DIR, "audio", f"{model_spec.model_name}.json"
        )
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, mode="w") as fd:
            fd.write(model_spec.json())


def unregister_audio(model_name: str, raise_error: bool = True):
    with UD_AUDIO_LOCK:
        model_spec = None
        for i, f in enumerate(UD_AUDIOS):
            if f.model_name == model_name:
                model_spec = f
                break
        if model_spec:
            UD_AUDIOS.remove(model_spec)

            persist_path = os.path.join(
                XINFERENCE_MODEL_DIR, "audio", f"{model_spec.model_name}.json"
            )
            if os.path.exists(persist_path):
                os.remove(persist_path)

            cache_dir = os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
            if os.path.exists(cache_dir):
                logger.warning(
                    f"Remove the cache of user-defined model {model_spec.model_name}. "
                    f"Cache directory: {cache_dir}"
                )
                if os.path.isdir(cache_dir):
                    os.rmdir(cache_dir)
                else:
                    logger.warning(
                        f"Cache directory is not a soft link, please remove it manually."
                    )
        else:
            if raise_error:
                raise ValueError(f"Model {model_name} not found")
            else:
                logger.warning(f"Custom audio model {model_name} not found")
