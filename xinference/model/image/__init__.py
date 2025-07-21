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
import warnings

from ..utils import flatten_model_src
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
    load_model_family_from_json("model_spec.json", BUILTIN_IMAGE_MODELS)

    # register model description
    for model_name, model_specs in BUILTIN_IMAGE_MODELS.items():
        model_spec = [x for x in model_specs if x.model_hub == "huggingface"][0]
        IMAGE_MODEL_DESCRIPTIONS.update(generate_image_description(model_spec))

    register_custom_model()

    for ud_image in get_user_defined_images():
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
