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

import codecs
import json
import os
import platform
import sys
import warnings
from typing import Any, Dict

from ...constants import XINFERENCE_MODEL_DIR
from .core import (
    AUDIO_MODEL_DESCRIPTIONS,
    MODEL_NAME_TO_REVISION,
    AudioModelFamilyV1,
    generate_audio_description,
    get_audio_model_descriptions,
    get_cache_status,
)
from .custom import (
    CustomAudioModelFamilyV1,
    get_user_defined_audios,
    register_audio,
    unregister_audio,
)

BUILTIN_AUDIO_MODELS: Dict[str, Any] = {}
MODELSCOPE_AUDIO_MODELS: Dict[str, Any] = {}


def register_custom_model():
    # if persist=True, load them when init
    user_defined_audio_dir = os.path.join(XINFERENCE_MODEL_DIR, "audio")
    if os.path.isdir(user_defined_audio_dir):
        for f in os.listdir(user_defined_audio_dir):
            try:
                with codecs.open(
                    os.path.join(user_defined_audio_dir, f), encoding="utf-8"
                ) as fd:
                    user_defined_audio_family = CustomAudioModelFamilyV1.parse_obj(
                        json.load(fd)
                    )
                    register_audio(user_defined_audio_family, persist=False)
            except Exception as e:
                warnings.warn(f"{user_defined_audio_dir}/{f} has error, {e}")


def _need_filter(spec: dict):
    if (sys.platform != "darwin" or platform.processor() != "arm") and spec.get(
        "engine", ""
    ).upper() == "MLX":
        return True
    return False


def _install():
    load_model_family_from_json("model_spec.json", BUILTIN_AUDIO_MODELS)
    load_model_family_from_json("model_spec_modelscope.json", MODELSCOPE_AUDIO_MODELS)

    # register model description after recording model revision
    for model_spec_info in [BUILTIN_AUDIO_MODELS, MODELSCOPE_AUDIO_MODELS]:
        for model_name, model_spec in model_spec_info.items():
            if model_spec.model_name not in AUDIO_MODEL_DESCRIPTIONS:
                AUDIO_MODEL_DESCRIPTIONS.update(generate_audio_description(model_spec))

    register_custom_model()

    # register model description
    for ud_audio in get_user_defined_audios():
        AUDIO_MODEL_DESCRIPTIONS.update(generate_audio_description(ud_audio))


def load_model_family_from_json(json_filename, target_families):
    json_path = os.path.join(os.path.dirname(__file__), json_filename)
    target_families.update(
        dict(
            (spec["model_name"], AudioModelFamilyV1(**spec))
            for spec in json.load(codecs.open(json_path, "r", encoding="utf-8"))
            if not _need_filter(spec)
        )
    )
    for model_name, model_spec in target_families.items():
        MODEL_NAME_TO_REVISION[model_name].append(model_spec.model_revision)

    del json_path
