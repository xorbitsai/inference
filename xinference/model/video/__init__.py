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
    BUILTIN_VIDEO_MODELS,
    MODEL_NAME_TO_REVISION,
    MODELSCOPE_VIDEO_MODELS,
    VIDEO_MODEL_DESCRIPTIONS,
    VideoModelFamilyV1,
    generate_video_description,
    get_cache_status,
    get_video_model_descriptions,
)


def _install():
    load_model_family_from_json("model_spec.json", BUILTIN_VIDEO_MODELS)
    load_model_family_from_json("model_spec_modelscope.json", MODELSCOPE_VIDEO_MODELS)

    # register model description
    for model_name, model_spec in chain(
        MODELSCOPE_VIDEO_MODELS.items(), BUILTIN_VIDEO_MODELS.items()
    ):
        VIDEO_MODEL_DESCRIPTIONS.update(generate_video_description(model_spec))


def load_model_family_from_json(json_filename, target_families):
    json_path = os.path.join(os.path.dirname(__file__), json_filename)
    target_families.update(
        dict(
            (spec["model_name"], VideoModelFamilyV1(**spec))
            for spec in json.load(codecs.open(json_path, "r", encoding="utf-8"))
        )
    )
    for model_name, model_spec in target_families.items():
        MODEL_NAME_TO_REVISION[model_name].append(model_spec.model_revision)

    del json_path
