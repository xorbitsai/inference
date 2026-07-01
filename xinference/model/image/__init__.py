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
import warnings

from ...constants import XINFERENCE_MODEL_DIR
from ..utils import flatten_model_src, flatten_quantizations
from .core import (
    BUILTIN_IMAGE_MODELS,
    IMAGE_MODEL_DESCRIPTIONS,
    ImageModelFamilyV2,
    generate_image_description,
    get_image_model_descriptions,
)
from .custom import (
    CustomImageModelFamilyV2,
    get_user_defined_images,
    register_image,
    unregister_image,
)
from .engine import register_builtin_image_engines
from .engine_family import (
    generate_engine_config_by_model_name as generate_image_engine_config,
)
from .ocr import register_builtin_ocr_engines
from .ocr.ocr_family import generate_engine_config_by_model_name


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


def _install():
    # Install models with intelligent merging based on timestamps
    from ..utils import install_models_with_merge

    install_models_with_merge(
        BUILTIN_IMAGE_MODELS,
        "model_spec.json",
        "image",
        "image_models.json",
        has_downloaded_models,
        load_model_family_from_json,
    )

    # register model description
    for model_name, model_specs in BUILTIN_IMAGE_MODELS.items():
        model_spec = [x for x in model_specs if x.model_hub == "huggingface"][0]
        IMAGE_MODEL_DESCRIPTIONS.update(generate_image_description(model_spec))

    register_builtin_image_engines()
    register_builtin_ocr_engines()
    for model_specs in BUILTIN_IMAGE_MODELS.values():
        for model_spec in model_specs:
            if model_spec.model_ability and "ocr" not in model_spec.model_ability:
                generate_image_engine_config(model_spec)
            if model_spec.model_ability and "ocr" in model_spec.model_ability:
                generate_engine_config_by_model_name(model_spec)

    register_custom_model()

    for ud_image in get_user_defined_images():
        IMAGE_MODEL_DESCRIPTIONS.update(generate_image_description(ud_image))
        if ud_image.model_ability and "ocr" not in ud_image.model_ability:
            generate_image_engine_config(ud_image)
        if ud_image.model_ability and "ocr" in ud_image.model_ability:
            generate_engine_config_by_model_name(ud_image)


def register_builtin_model():
    """Register built-in image models."""
    _install()


def has_downloaded_models():
    """Check if downloaded JSON configurations exist."""
    builtin_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "image")
    json_file_path = os.path.join(builtin_dir, "image_models.json")
    return os.path.exists(json_file_path)


def load_downloaded_models():
    """Load downloaded JSON configurations from the builtin directory."""
    builtin_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "image")
    json_file_path = os.path.join(builtin_dir, "image_models.json")

    try:
        load_model_family_from_json(json_file_path, BUILTIN_IMAGE_MODELS)
    except Exception as e:
        warnings.warn(
            f"Failed to load downloaded image models from {json_file_path}: {e}"
        )
        # Fall back to built-in models if download fails
        load_model_family_from_json("model_spec.json", BUILTIN_IMAGE_MODELS)


def load_model_family_from_json(json_filename, target_families):
    # Handle both relative (module directory) and absolute paths
    if os.path.isabs(json_filename):
        json_path = json_filename
    else:
        json_path = os.path.join(os.path.dirname(__file__), json_filename)

    flattened_model_specs = []
    for spec in json.load(codecs.open(json_path, "r", encoding="utf-8")):
        base_info = {
            key: value
            for key, value in spec.items()
            if key not in ("model_src", "model_specs")
        }
        if "model_specs" in spec:
            for model_spec in spec["model_specs"]:
                spec_base = base_info.copy()
                spec_base.update(
                    {k: v for k, v in model_spec.items() if k != "model_src"}
                )
                if "model_src" in model_spec:
                    spec_entry = spec_base.copy()
                    spec_entry["model_src"] = model_spec["model_src"]
                    if any(
                        "quantizations" in hub_info
                        for hub_info in model_spec["model_src"].values()
                    ):
                        flattened_model_specs.extend(flatten_quantizations(spec_entry))
                    else:
                        flattened_model_specs.extend(flatten_model_src(spec_entry))
                else:
                    flattened_model_specs.append(spec_base)
        else:
            flattened_model_specs.extend(flatten_model_src(spec))

    for spec in flattened_model_specs:
        if spec["model_name"] not in target_families:
            target_families[spec["model_name"]] = [ImageModelFamilyV2(**spec)]
        else:
            target_families[spec["model_name"]].append(ImageModelFamilyV2(**spec))

    del json_path
