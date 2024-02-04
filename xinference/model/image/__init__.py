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
from itertools import chain

from .core import (
    IMAGE_MODEL_DESCRIPTIONS,
    ImageModelFamilyV1,
    generate_image_description,
    get_cache_status,
    get_image_model_descriptions,
)

_model_spec_json = os.path.join(os.path.dirname(__file__), "model_spec.json")
_model_spec_modelscope_json = os.path.join(
    os.path.dirname(__file__), "model_spec_modelscope.json"
)
BUILTIN_IMAGE_MODELS = dict(
    (spec["model_name"], ImageModelFamilyV1(**spec))
    for spec in json.load(codecs.open(_model_spec_json, "r", encoding="utf-8"))
)
MODELSCOPE_IMAGE_MODELS = dict(
    (spec["model_name"], ImageModelFamilyV1(**spec))
    for spec in json.load(
        codecs.open(_model_spec_modelscope_json, "r", encoding="utf-8")
    )
)

# register model description
for model_name, model_spec in chain(
    MODELSCOPE_IMAGE_MODELS.items(), BUILTIN_IMAGE_MODELS.items()
):
    IMAGE_MODEL_DESCRIPTIONS.update(generate_image_description(model_spec))

del _model_spec_json
