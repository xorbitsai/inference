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
import platform
import sys
import warnings
from typing import Any, Dict, List

from ...constants import XINFERENCE_MODEL_DIR
from ..utils import flatten_model_src

logger = logging.getLogger(__name__)


def convert_audio_model_format(model_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert audio model hub JSON format to Xinference expected format.
    """
    logger.debug(
        f"convert_audio_model_format called for: {model_json.get('model_name', 'Unknown')}"
    )

    # Apply conversion logic to handle null model_id and other issues
    if model_json.get("model_id") is None and "model_src" in model_json:
        model_src = model_json["model_src"]
        # Extract model_id from available sources
        if "huggingface" in model_src and "model_id" in model_src["huggingface"]:
            model_json["model_id"] = model_src["huggingface"]["model_id"]
        elif "modelscope" in model_src and "model_id" in model_src["modelscope"]:
            model_json["model_id"] = model_src["modelscope"]["model_id"]

        # Extract model_revision if available
        if model_json.get("model_revision") is None:
            if (
                "huggingface" in model_src
                and "model_revision" in model_src["huggingface"]
            ):
                model_json["model_revision"] = model_src["huggingface"][
                    "model_revision"
                ]
            elif (
                "modelscope" in model_src
                and "model_revision" in model_src["modelscope"]
            ):
                model_json["model_revision"] = model_src["modelscope"]["model_revision"]

    # Ensure required fields for audio models
    if "version" not in model_json:
        model_json["version"] = 2
    if "model_lang" not in model_json:
        model_json["model_lang"] = [
            "en",
            "zh",
        ]  # Audio models often support multiple languages

    return model_json


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


def register_builtin_model():
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("audio")
    existing_model_names = {spec.model_name for spec in registry.get_custom_models()}

    builtin_audio_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "audio")
    if os.path.isdir(builtin_audio_dir):
        # First, try to load from the complete JSON file
        complete_json_path = os.path.join(builtin_audio_dir, "audio_models.json")
        if os.path.exists(complete_json_path):
            try:
                with codecs.open(complete_json_path, encoding="utf-8") as fd:
                    model_data = json.load(fd)

                # Handle different formats
                models_to_register = []
                if isinstance(model_data, list):
                    # Multiple models in a list
                    models_to_register = model_data
                elif isinstance(model_data, dict):
                    # Single model
                    if "model_name" in model_data:
                        models_to_register = [model_data]
                    else:
                        # Models dict - extract models
                        for key, value in model_data.items():
                            if isinstance(value, dict) and "model_name" in value:
                                models_to_register.append(value)

                # Register all models from the complete JSON
                for model_data in models_to_register:
                    try:
                        # Apply conversion logic to handle null model_id and other issues
                        converted_data = convert_audio_model_format(model_data)
                        builtin_audio_family = AudioModelFamilyV2.parse_obj(
                            converted_data
                        )

                        # Only register if model doesn't already exist
                        if builtin_audio_family.model_name not in existing_model_names:
                            # Add to BUILTIN_AUDIO_MODELS directly for proper builtin registration
                            if (
                                builtin_audio_family.model_name
                                not in BUILTIN_AUDIO_MODELS
                            ):
                                BUILTIN_AUDIO_MODELS[
                                    builtin_audio_family.model_name
                                ] = []
                            BUILTIN_AUDIO_MODELS[
                                builtin_audio_family.model_name
                            ].append(builtin_audio_family)
                            existing_model_names.add(builtin_audio_family.model_name)
                    except Exception as e:
                        warnings.warn(
                            f"Error parsing audio model {model_data.get('model_name', 'Unknown')}: {e}"
                        )

                logger.info(
                    f"Successfully registered {len(models_to_register)} audio models from complete JSON"
                )

            except Exception as e:
                warnings.warn(
                    f"Error loading complete JSON file {complete_json_path}: {e}"
                )
                # Fall back to individual files if complete JSON loading fails

        # Fall back: load individual JSON files (backward compatibility)
        individual_files = [
            f
            for f in os.listdir(builtin_audio_dir)
            if f.endswith(".json") and f != "audio_models.json"
        ]
        for f in individual_files:
            try:
                with codecs.open(
                    os.path.join(builtin_audio_dir, f), encoding="utf-8"
                ) as fd:
                    model_data = json.load(fd)

                    # Apply conversion logic to handle null model_id and other issues
                    if model_data.get("model_id") is None and "model_src" in model_data:
                        model_src = model_data["model_src"]
                        # Extract model_id from available sources
                        if (
                            "huggingface" in model_src
                            and "model_id" in model_src["huggingface"]
                        ):
                            model_data["model_id"] = model_src["huggingface"][
                                "model_id"
                            ]
                        elif (
                            "modelscope" in model_src
                            and "model_id" in model_src["modelscope"]
                        ):
                            model_data["model_id"] = model_src["modelscope"]["model_id"]

                        # Extract model_revision if available
                        if model_data.get("model_revision") is None:
                            if (
                                "huggingface" in model_src
                                and "model_revision" in model_src["huggingface"]
                            ):
                                model_data["model_revision"] = model_src["huggingface"][
                                    "model_revision"
                                ]
                            elif (
                                "modelscope" in model_src
                                and "model_revision" in model_src["modelscope"]
                            ):
                                model_data["model_revision"] = model_src["modelscope"][
                                    "model_revision"
                                ]

                    builtin_audio_family = AudioModelFamilyV2.parse_obj(model_data)

                    # Only register if model doesn't already exist
                    if builtin_audio_family.model_name not in existing_model_names:
                        # Add to BUILTIN_AUDIO_MODELS directly for proper builtin registration
                        if builtin_audio_family.model_name not in BUILTIN_AUDIO_MODELS:
                            BUILTIN_AUDIO_MODELS[builtin_audio_family.model_name] = []
                        BUILTIN_AUDIO_MODELS[builtin_audio_family.model_name].append(
                            builtin_audio_family
                        )
                        existing_model_names.add(builtin_audio_family.model_name)
            except Exception as e:
                warnings.warn(f"{builtin_audio_dir}/{f} has error, {e}")


def _need_filter(spec: dict):
    if (sys.platform != "darwin" or platform.processor() != "arm") and spec.get(
        "engine", ""
    ).upper() == "MLX":
        return True
    return False


def _install():
    load_model_family_from_json("model_spec.json", BUILTIN_AUDIO_MODELS)

    # register model description after recording model revision
    for model_name, model_specs in BUILTIN_AUDIO_MODELS.items():
        model_spec = [x for x in model_specs if x.model_hub == "huggingface"][0]
        if model_spec.model_name not in AUDIO_MODEL_DESCRIPTIONS:
            AUDIO_MODEL_DESCRIPTIONS.update(generate_audio_description(model_spec))

    register_custom_model()

    # register model description
    for ud_audio in get_user_defined_audios():
        AUDIO_MODEL_DESCRIPTIONS.update(generate_audio_description(ud_audio))


def load_model_family_from_json(json_filename, target_families):
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
