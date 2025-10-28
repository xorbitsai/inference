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
import logging
import os
import warnings
from typing import Any, Dict

from ..utils import flatten_model_src

logger = logging.getLogger(__name__)


def convert_video_model_format(model_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert video model hub JSON format to Xinference expected format.
    """
    logger.debug(
        f"convert_video_model_format called for: {model_json.get('model_name', 'Unknown')}"
    )

    # Ensure required fields for video models
    converted = model_json.copy()

    # Add missing required fields
    if "version" not in converted:
        converted["version"] = 2
    if "model_lang" not in converted:
        converted["model_lang"] = ["en"]

    # Handle model_specs
    if "model_specs" not in converted or not converted["model_specs"]:
        converted["model_specs"] = [
            {
                "model_format": "pytorch",
                "model_size_in_billions": None,
                "quantization": "none",
                "model_hub": "huggingface",
            }
        ]
    else:
        # Ensure each spec has required fields
        for spec in converted["model_specs"]:
            if "quantization" not in spec:
                spec["quantization"] = "none"
            if "model_hub" not in spec:
                spec["model_hub"] = "huggingface"

    return converted


from .core import (
    BUILTIN_VIDEO_MODELS,
    VIDEO_MODEL_DESCRIPTIONS,
    VideoModelFamilyV2,
    generate_video_description,
    get_video_model_descriptions,
)
from .custom import (
    CustomVideoModelFamilyV2,
    register_video,
    unregister_video,
)


def register_custom_model():
    from ...constants import XINFERENCE_MODEL_DIR
    from ..custom import migrate_from_v1_to_v2

    # migrate from v1 to v2 first
    migrate_from_v1_to_v2("video", CustomVideoModelFamilyV2)

    user_defined_video_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "video")
    if os.path.isdir(user_defined_video_dir):
        for f in os.listdir(user_defined_video_dir):
            try:
                with codecs.open(
                    os.path.join(user_defined_video_dir, f), encoding="utf-8"
                ) as fd:
                    user_defined_video_family = CustomVideoModelFamilyV2.parse_obj(
                        json.load(fd)
                    )
                    register_video(user_defined_video_family, persist=False)
            except Exception as e:
                warnings.warn(f"{user_defined_video_dir}/{f} has error, {e}")


def register_builtin_model():
    """
    Dynamically load built-in video models from builtin/video directory.
    This function is called every time model list is requested,
    ensuring real-time updates without server restart.
    """
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("video")
    existing_model_names = {spec.model_name for spec in registry.get_custom_models()}

    # Use the builtin registry to load models
    from .builtin import BuiltinVideoModelRegistry

    builtin_registry = BuiltinVideoModelRegistry()
    builtin_models = builtin_registry.get_builtin_models()

    for model in builtin_models:
        # Only register if model doesn't already exist
        if model.model_name not in existing_model_names:
            # Add to BUILTIN_VIDEO_MODELS directly for proper builtin registration
            if model.model_name not in BUILTIN_VIDEO_MODELS:
                BUILTIN_VIDEO_MODELS[model.model_name] = []
            BUILTIN_VIDEO_MODELS[model.model_name].append(model)
            existing_model_names.add(model.model_name)


def _install():
    load_model_family_from_json("model_spec.json", BUILTIN_VIDEO_MODELS)

    # register model description
    for model_name, model_specs in BUILTIN_VIDEO_MODELS.items():
        model_spec = [x for x in model_specs if x.model_hub == "huggingface"][0]
        VIDEO_MODEL_DESCRIPTIONS.update(generate_video_description(model_spec))


def load_model_family_from_json(json_filename, target_families):
    json_path = os.path.join(os.path.dirname(__file__), json_filename)
    flattened_model_specs = []
    for spec in json.load(codecs.open(json_path, "r", encoding="utf-8")):
        flattened_model_specs.extend(flatten_model_src(spec))

    for spec in flattened_model_specs:
        if spec["model_name"] not in target_families:
            target_families[spec["model_name"]] = [VideoModelFamilyV2(**spec)]
        else:
            target_families[spec["model_name"]].append(VideoModelFamilyV2(**spec))

    del json_path
