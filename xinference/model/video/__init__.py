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

from ..utils import flatten_model_src
from .core import (
    BUILTIN_VIDEO_MODELS,
    VIDEO_MODEL_DESCRIPTIONS,
    VideoModelFamilyV2,
    generate_video_description,
    get_video_model_descriptions,
)


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
