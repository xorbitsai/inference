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

from .core import (
    BUILTIN_LVLM_FAMILIES,
    BUILTIN_MODELSCOPE_LVLM_FAMILIES,
    MODEL_CLASSES,
    MODEL_NAME_TO_REVISION,
    LVLMFamilyV1,
    LVLMPromptStyleV1,
)
from .qwen_vl import QwenVLChat

MODEL_CLASSES.append(QwenVLChat)


def _install():
    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_spec.json"
    )
    for json_obj in json.load(codecs.open(json_path, "r", encoding="utf-8")):
        model_family = LVLMFamilyV1.parse_obj(json_obj)
        BUILTIN_LVLM_FAMILIES.append(model_family)
        for model_spec in model_family.model_specs:
            MODEL_NAME_TO_REVISION[model_family.model_name].append(
                model_spec.model_revision
            )


_install()
