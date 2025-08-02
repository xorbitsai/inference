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
from typing import Any, Dict, List

from ...constants import XINFERENCE_MODEL_DIR
from ..utils import flatten_quantizations
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
from .rerank_family import (
    BUILTIN_RERANK_MODELS,
    RERANK_ENGINES,
    SENTENCE_TRANSFORMER_CLASSES,
    SUPPORTED_ENGINES,
    VLLM_CLASSES,
)


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


def generate_engine_config_by_model_name(model_family: "RerankModelFamilyV2"):
    model_name = model_family.model_name
    engines: Dict[str, List[Dict[str, Any]]] = RERANK_ENGINES.get(
        model_name, {}
    )  # structure for engine query
    for spec in [x for x in model_family.model_specs if x.model_hub == "huggingface"]:
        model_format = spec.model_format
        quantization = spec.quantization
        for engine in SUPPORTED_ENGINES:
            CLASSES = SUPPORTED_ENGINES[engine]
            for cls in CLASSES:
                # Every engine needs to implement match method
                if cls.match(model_family, spec, quantization):
                    # we only match the first class for an engine
                    if engine not in engines:
                        engines[engine] = [
                            {
                                "model_name": model_name,
                                "model_format": model_format,
                                "quantization": quantization,
                                "rerank_class": cls,
                            }
                        ]
                    else:
                        engines[engine].append(
                            {
                                "model_name": model_name,
                                "model_format": model_format,
                                "quantization": quantization,
                                "rerank_class": cls,
                            }
                        )
                    break
    RERANK_ENGINES[model_name] = engines


def _install():
    _model_spec_json = os.path.join(os.path.dirname(__file__), "model_spec.json")
    for json_obj in json.load(codecs.open(_model_spec_json, "r", encoding="utf-8")):
        flattened = []
        for spec in json_obj["model_specs"]:
            flattened.extend(flatten_quantizations(spec))
        json_obj["model_specs"] = flattened
        BUILTIN_RERANK_MODELS[json_obj["model_name"]] = RerankModelFamilyV2(**json_obj)

    for model_name, model_spec in BUILTIN_RERANK_MODELS.items():
        if model_spec.model_name not in RERANK_MODEL_DESCRIPTIONS:
            RERANK_MODEL_DESCRIPTIONS.update(generate_rerank_description(model_spec))

    from .sentence_transformers.core import SentenceTransformerRerankModel
    from .vllm.core import VLLMRerankModel

    SENTENCE_TRANSFORMER_CLASSES.extend([SentenceTransformerRerankModel])
    VLLM_CLASSES.extend([VLLMRerankModel])

    SUPPORTED_ENGINES["sentence_transformers"] = SENTENCE_TRANSFORMER_CLASSES
    SUPPORTED_ENGINES["vllm"] = VLLM_CLASSES

    for model_spec in BUILTIN_RERANK_MODELS.values():
        generate_engine_config_by_model_name(model_spec)

    register_custom_model()

    # register model description
    for ud_rerank in get_user_defined_reranks():
        RERANK_MODEL_DESCRIPTIONS.update(generate_rerank_description(ud_rerank))
