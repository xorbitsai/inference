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
from itertools import chain

from .core import (
    BUILTIN_IMAGE_MODELS,
    IMAGE_MODEL_DESCRIPTIONS,
    MODEL_NAME_TO_REVISION,
    MODELSCOPE_IMAGE_MODELS,
    ImageModelFamilyV1,
    generate_image_description,
    get_cache_status,
    get_image_model_descriptions,
)
from .custom import (
    CustomImageModelFamilyV1,
    get_user_defined_images,
    register_image,
    unregister_image,
)


def register_custom_model():
    from ...constants import XINFERENCE_MODEL_DIR

    user_defined_image_dir = os.path.join(XINFERENCE_MODEL_DIR, "image")
    if os.path.isdir(user_defined_image_dir):
        for f in os.listdir(user_defined_image_dir):
            try:
                with codecs.open(
                    os.path.join(user_defined_image_dir, f), encoding="utf-8"
                ) as fd:
                    user_defined_image_family = CustomImageModelFamilyV1.parse_obj(
                        json.load(fd)
                    )
                    register_image(user_defined_image_family, persist=False)
            except Exception as e:
                warnings.warn(f"{user_defined_image_dir}/{f} has error, {e}")


def _install():
    _model_spec_json = os.path.join(os.path.dirname(__file__), "model_spec.json")
    _model_spec_modelscope_json = os.path.join(
        os.path.dirname(__file__), "model_spec_modelscope.json"
    )
    BUILTIN_IMAGE_MODELS.update(
        dict(
            (spec["model_name"], ImageModelFamilyV1(**spec))
            for spec in json.load(codecs.open(_model_spec_json, "r", encoding="utf-8"))
        )
    )
    for model_name, model_spec in BUILTIN_IMAGE_MODELS.items():
        MODEL_NAME_TO_REVISION[model_name].append(model_spec.model_revision)

    MODELSCOPE_IMAGE_MODELS.update(
        dict(
            (spec["model_name"], ImageModelFamilyV1(**spec))
            for spec in json.load(
                codecs.open(_model_spec_modelscope_json, "r", encoding="utf-8")
            )
        )
    )
    for model_name, model_spec in MODELSCOPE_IMAGE_MODELS.items():
        MODEL_NAME_TO_REVISION[model_name].append(model_spec.model_revision)

    # register model description
    for model_name, model_spec in chain(
        MODELSCOPE_IMAGE_MODELS.items(), BUILTIN_IMAGE_MODELS.items()
    ):
        IMAGE_MODEL_DESCRIPTIONS.update(generate_image_description(model_spec))

    register_custom_model()

    for ud_image in get_user_defined_images():
        IMAGE_MODEL_DESCRIPTIONS.update(generate_image_description(ud_image))

    del _model_spec_json
    del _model_spec_modelscope_json
