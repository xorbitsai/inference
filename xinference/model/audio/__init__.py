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

import codecs
import json
import os
import platform
import sys
import warnings
from typing import Dict, List

from ...constants import XINFERENCE_MODEL_DIR
from ..utils import flatten_model_src
from .core import (
    AUDIO_MODEL_DESCRIPTIONS,
    AudioModelFamilyV2,
    generate_audio_description,
    get_audio_model_descriptions,
)
from .custom import (
    CustomAudioModelFamilyV2,
    get_user_defined_audios,
    register_audio,
    unregister_audio,
)

BUILTIN_AUDIO_MODELS: Dict[str, List["AudioModelFamilyV2"]] = {}


def register_custom_model():
    from ..custom import migrate_from_v1_to_v2

    # migrate from v1 to v2 first
    migrate_from_v1_to_v2("audio", CustomAudioModelFamilyV2)

    # if persist=True, load them when init
    user_defined_audio_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "audio")
    if os.path.isdir(user_defined_audio_dir):
        for f in os.listdir(user_defined_audio_dir):
            try:
                with codecs.open(
                    os.path.join(user_defined_audio_dir, f), encoding="utf-8"
                ) as fd:
                    user_defined_audio_family = CustomAudioModelFamilyV2.parse_obj(
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
    # Install models with intelligent merging based on timestamps
    from ..utils import install_models_with_merge

    install_models_with_merge(
        BUILTIN_AUDIO_MODELS,
        "model_spec.json",
        "audio",
        "audio_models.json",
        has_downloaded_models,
        load_model_family_from_json,
    )

    # register model description after recording model revision
    for model_name, model_specs in BUILTIN_AUDIO_MODELS.items():
        model_spec = [x for x in model_specs if x.model_hub == "huggingface"][0]
        if model_spec.model_name not in AUDIO_MODEL_DESCRIPTIONS:
            AUDIO_MODEL_DESCRIPTIONS.update(generate_audio_description(model_spec))

    register_custom_model()

    # register model description
    for ud_audio in get_user_defined_audios():
        AUDIO_MODEL_DESCRIPTIONS.update(generate_audio_description(ud_audio))


def register_builtin_model():
    """Register built-in audio models."""
    _install()


def has_downloaded_models():
    """Check if downloaded JSON configurations exist."""
    builtin_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "audio")
    json_file_path = os.path.join(builtin_dir, "audio_models.json")
    return os.path.exists(json_file_path)


def load_downloaded_models():
    """Load downloaded JSON configurations from the builtin directory."""
    builtin_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "audio")
    json_file_path = os.path.join(builtin_dir, "audio_models.json")

    try:
        load_model_family_from_json(json_file_path, BUILTIN_AUDIO_MODELS)
    except Exception as e:
        warnings.warn(
            f"Failed to load downloaded audio models from {json_file_path}: {e}"
        )
        # Fall back to built-in models if download fails
        load_model_family_from_json("model_spec.json", BUILTIN_AUDIO_MODELS)


def load_model_family_from_json(json_filename, target_families):
    # Handle both relative (module directory) and absolute paths
    if os.path.isabs(json_filename):
        json_path = json_filename
    else:
        json_path = os.path.join(os.path.dirname(__file__), json_filename)

    flattened_model_specs = []
    for spec in json.load(codecs.open(json_path, "r", encoding="utf-8")):
        flattened_model_specs.extend(flatten_model_src(spec))

    for spec in flattened_model_specs:
        if not _need_filter(spec):
            if spec["model_name"] not in target_families:
                target_families[spec["model_name"]] = [AudioModelFamilyV2(**spec)]
            else:
                target_families[spec["model_name"]].append(AudioModelFamilyV2(**spec))

    del json_path
