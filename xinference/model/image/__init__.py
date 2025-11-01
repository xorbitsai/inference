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


from .core import (
    BUILTIN_IMAGE_MODELS,
    IMAGE_MODEL_DESCRIPTIONS,
    ImageModelFamilyV2,
    generate_image_description,
    get_image_model_descriptions,
)
from .custom import (
    CustomImageModelFamilyV2,
    get_registered_images,
    register_image,
    unregister_image,
)


def register_custom_model():
    from ...constants import XINFERENCE_MODEL_DIR
    from ..custom import migrate_from_v1_to_v2

    # migrate from v1 to v2 first
    migrate_from_v1_to_v2("image", CustomImageModelFamilyV2)

    user_defined_image_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "image")
    if os.path.isdir(user_defined_image_dir):
        for f in os.listdir(user_defined_image_dir):
            try:
                with codecs.open(
                    os.path.join(user_defined_image_dir, f), encoding="utf-8"
                ) as fd:
                    user_defined_image_family = CustomImageModelFamilyV2.parse_obj(
                        json.load(fd)
                    )
                    register_image(user_defined_image_family, persist=False)
            except Exception as e:
                warnings.warn(f"{user_defined_image_dir}/{f} has error, {e}")


def register_builtin_model():
    import json

    from ...constants import XINFERENCE_MODEL_DIR
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("image")
    existing_model_names = {spec.model_name for spec in registry.get_custom_models()}

    builtin_image_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "image")
    if os.path.isdir(builtin_image_dir):
        # First, try to load from the complete JSON file
        complete_json_path = os.path.join(builtin_image_dir, "image_models.json")
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
                        # Convert format using flatten_model_src
                        from ..utils import flatten_model_src

                        flattened_list = flatten_model_src(model_data)
                        converted_data = (
                            flattened_list[0] if flattened_list else model_data
                        )
                        builtin_image_family = ImageModelFamilyV2.parse_obj(
                            converted_data
                        )

                        # Only register if model doesn't already exist
                        if builtin_image_family.model_name not in existing_model_names:
                            # Add to BUILTIN_IMAGE_MODELS directly for proper builtin registration
                            if (
                                builtin_image_family.model_name
                                not in BUILTIN_IMAGE_MODELS
                            ):
                                BUILTIN_IMAGE_MODELS[
                                    builtin_image_family.model_name
                                ] = []
                            BUILTIN_IMAGE_MODELS[
                                builtin_image_family.model_name
                            ].append(builtin_image_family)
                            # Update model descriptions for the new builtin model
                            IMAGE_MODEL_DESCRIPTIONS.update(
                                generate_image_description(builtin_image_family)
                            )
                            existing_model_names.add(builtin_image_family.model_name)
                    except Exception as e:
                        warnings.warn(
                            f"Error parsing image model {model_data.get('model_name', 'Unknown')}: {e}"
                        )

                logger.info(
                    f"Successfully registered {len(models_to_register)} image models from complete JSON"
                )

            except Exception as e:
                warnings.warn(
                    f"Error loading complete JSON file {complete_json_path}: {e}"
                )
                # Fall back to individual files if complete JSON loading fails

        # Fall back: load individual JSON files (backward compatibility)
        individual_files = [
            f
            for f in os.listdir(builtin_image_dir)
            if f.endswith(".json") and f != "image_models.json"
        ]
        for f in individual_files:
            try:
                with codecs.open(
                    os.path.join(builtin_image_dir, f), encoding="utf-8"
                ) as fd:
                    builtin_image_family = ImageModelFamilyV2.parse_obj(json.load(fd))

                    # Only register if model doesn't already exist
                    if builtin_image_family.model_name not in existing_model_names:
                        # Add to BUILTIN_IMAGE_MODELS directly for proper builtin registration
                        if builtin_image_family.model_name not in BUILTIN_IMAGE_MODELS:
                            BUILTIN_IMAGE_MODELS[builtin_image_family.model_name] = []
                        BUILTIN_IMAGE_MODELS[builtin_image_family.model_name].append(
                            builtin_image_family
                        )
                        # Update model descriptions for the new builtin model
                        IMAGE_MODEL_DESCRIPTIONS.update(
                            generate_image_description(builtin_image_family)
                        )
                        existing_model_names.add(builtin_image_family.model_name)
            except Exception as e:
                warnings.warn(f"{builtin_image_dir}/{f} has error, {e}")


def _install():
    load_model_family_from_json("model_spec.json", BUILTIN_IMAGE_MODELS)

    # Load models from complete JSON file (from update_model_type)
    register_builtin_model()

    # register model description
    for model_name, model_specs in BUILTIN_IMAGE_MODELS.items():
        model_spec = [x for x in model_specs if x.model_hub == "huggingface"][0]
        IMAGE_MODEL_DESCRIPTIONS.update(generate_image_description(model_spec))

    register_custom_model()

    for ud_image in get_registered_images():
        IMAGE_MODEL_DESCRIPTIONS.update(generate_image_description(ud_image))


def load_model_family_from_json(json_filename, target_families):
    json_path = os.path.join(os.path.dirname(__file__), json_filename)
    flattened_model_specs = []
    for spec in json.load(codecs.open(json_path, "r", encoding="utf-8")):
        flattened_model_specs.extend(flatten_model_src(spec))

    for spec in flattened_model_specs:
        if spec["model_name"] not in target_families:
            target_families[spec["model_name"]] = [ImageModelFamilyV2(**spec)]
        else:
            target_families[spec["model_name"]].append(ImageModelFamilyV2(**spec))

    del json_path
