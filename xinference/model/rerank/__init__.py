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
    LLAMA_CPP_CLASSES,
    RERANK_ENGINES,
    SENTENCE_TRANSFORMER_CLASSES,
    SUPPORTED_ENGINES,
    VLLM_CLASSES,
)


def register_builtin_model():
    """Register built-in rerank models."""
    _install()


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


def check_format_with_engine(model_format, engine):
    if model_format in ["ggufv2"] and engine not in ["llama.cpp"]:
        return False
    if model_format not in ["ggufv2"] and engine == "llama.cpp":
        return False
    return True


def generate_engine_config_by_model_name(model_family: "RerankModelFamilyV2"):
    model_name = model_family.model_name
    engines: Dict[str, List[Dict[str, Any]]] = RERANK_ENGINES.get(
        model_name, {}
    )  # structure for engine query
    for spec in [x for x in model_family.model_specs if x.model_hub == "huggingface"]:
        model_format = spec.model_format
        quantization = spec.quantization
        for engine in SUPPORTED_ENGINES:
            if not check_format_with_engine(model_format, engine):
                continue
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


def has_downloaded_models():
    """Check if downloaded JSON configurations exist."""
    builtin_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "rerank")
    json_file_path = os.path.join(builtin_dir, "rerank_models.json")
    return os.path.exists(json_file_path)


def load_downloaded_models():
    """Load downloaded JSON configurations from the builtin directory."""
    builtin_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "rerank")
    json_file_path = os.path.join(builtin_dir, "rerank_models.json")

    try:
        load_model_family_from_json(json_file_path, BUILTIN_RERANK_MODELS)
    except Exception as e:
        warnings.warn(
            f"Failed to load downloaded rerank models from {json_file_path}: {e}"
        )
        # Fall back to built-in models if download fails
        load_model_family_from_json("model_spec.json", BUILTIN_RERANK_MODELS)


def load_model_family_from_json(json_filename, target_families):
    # Handle both relative (module directory) and absolute paths
    if os.path.isabs(json_filename):
        json_path = json_filename
    else:
        json_path = os.path.join(os.path.dirname(__file__), json_filename)

    for json_obj in json.load(codecs.open(json_path, "r", encoding="utf-8")):
        flattened = []
        for spec in json_obj["model_specs"]:
            flattened.extend(flatten_quantizations(spec))
        json_obj["model_specs"] = flattened
        if json_obj["model_name"] not in target_families:
            target_families[json_obj["model_name"]] = [RerankModelFamilyV2(**json_obj)]
        else:
            target_families[json_obj["model_name"]].append(
                RerankModelFamilyV2(**json_obj)
            )

    del json_path


def _install():
    # Install models with intelligent merging based on timestamps
    from ..utils import install_models_with_merge

    install_models_with_merge(
        BUILTIN_RERANK_MODELS,
        "model_spec.json",
        "rerank",
        "rerank_models.json",
        has_downloaded_models,
        load_model_family_from_json,
    )

    for model_name, model_spec_list in BUILTIN_RERANK_MODELS.items():
        # model_spec_list is a list of RerankModelFamilyV2 objects
        model_spec = model_spec_list[0]  # Take the first model family
        if model_spec.model_name not in RERANK_MODEL_DESCRIPTIONS:
            RERANK_MODEL_DESCRIPTIONS.update(generate_rerank_description(model_spec))

    from .llama_cpp.core import XllamaCppRerankModel
    from .sentence_transformers.core import SentenceTransformerRerankModel
    from .vllm.core import VLLMRerankModel

    SENTENCE_TRANSFORMER_CLASSES.extend([SentenceTransformerRerankModel])
    VLLM_CLASSES.extend([VLLMRerankModel])
    LLAMA_CPP_CLASSES.extend([XllamaCppRerankModel])

    SUPPORTED_ENGINES["sentence_transformers"] = SENTENCE_TRANSFORMER_CLASSES
    SUPPORTED_ENGINES["vllm"] = VLLM_CLASSES
    SUPPORTED_ENGINES["llama.cpp"] = LLAMA_CPP_CLASSES

    for model_spec_list in BUILTIN_RERANK_MODELS.values():
        for model_spec in model_spec_list:
            generate_engine_config_by_model_name(model_spec)

    register_custom_model()

    # register model description
    for ud_rerank in get_user_defined_reranks():
        RERANK_MODEL_DESCRIPTIONS.update(generate_rerank_description(ud_rerank))
