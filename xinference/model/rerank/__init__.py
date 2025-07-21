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
from typing import Dict, List

from ...constants import XINFERENCE_MODEL_DIR
from ..utils import flatten_model_src
from .core import (
    RERANK_MODEL_DESCRIPTIONS,
    RerankModelFamilyV2,
    generate_rerank_description,
    get_rerank_model_descriptions,
)
from .custom import (
    CustomRerankModelFamilyV2,
    get_user_defined_reranks,
    register_rerank,
    unregister_rerank,
)

BUILTIN_RERANK_MODELS: Dict[str, List["RerankModelFamilyV2"]] = {}


def register_custom_model():
    from ..custom import migrate_from_v1_to_v2

    # migrate from v1 to v2 first
    migrate_from_v1_to_v2("rerank", CustomRerankModelFamilyV2)

    # if persist=True, load them when init
    user_defined_rerank_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "rerank")
    if os.path.isdir(user_defined_rerank_dir):
        for f in os.listdir(user_defined_rerank_dir):
            try:
                with codecs.open(
                    os.path.join(user_defined_rerank_dir, f), encoding="utf-8"
                ) as fd:
                    user_defined_rerank_spec = CustomRerankModelFamilyV2.parse_obj(
                        json.load(fd)
                    )
                    register_rerank(user_defined_rerank_spec, persist=False)
            except Exception as e:
                warnings.warn(f"{user_defined_rerank_dir}/{f} has error, {e}")


def _install():
    load_model_family_from_json("model_spec.json", BUILTIN_RERANK_MODELS)

    for model_name, model_specs in BUILTIN_RERANK_MODELS.items():
        model_spec = [x for x in model_specs if x.model_hub == "huggingface"][0]
        if model_spec.model_name not in RERANK_MODEL_DESCRIPTIONS:
            RERANK_MODEL_DESCRIPTIONS.update(generate_rerank_description(model_spec))

    register_custom_model()

    # register model description
    for ud_rerank in get_user_defined_reranks():
        RERANK_MODEL_DESCRIPTIONS.update(generate_rerank_description(ud_rerank))


def load_model_family_from_json(json_filename, target_families):
    _model_spec_json = os.path.join(os.path.dirname(__file__), json_filename)
    flattened_model_specs = []
    for spec in json.load(codecs.open(_model_spec_json, "r", encoding="utf-8")):
        flattened_model_specs.extend(flatten_model_src(spec))

    for spec in flattened_model_specs:
        if spec["model_name"] not in target_families:
            target_families[spec["model_name"]] = [RerankModelFamilyV2(**spec)]
        else:
            target_families[spec["model_name"]].append(RerankModelFamilyV2(**spec))

    del _model_spec_json
