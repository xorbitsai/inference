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

from ..utils import flatten_quantizations
from .core import (
    EMBEDDING_MODEL_DESCRIPTIONS,
    EmbeddingModelFamilyV2,
    generate_embedding_description,
    get_embedding_model_descriptions,
)
from .custom import (
    CustomEmbeddingModelFamilyV2,
    get_user_defined_embeddings,
    register_embedding,
    unregister_embedding,
)
from .embed_family import (
    BUILTIN_EMBEDDING_MODELS,
    EMBEDDING_ENGINES,
    FLAG_EMBEDDER_CLASSES,
    LLAMA_CPP_CLASSES,
    SENTENCE_TRANSFORMER_CLASSES,
    SUPPORTED_ENGINES,
    VLLM_CLASSES,
)


def register_custom_model():
    from ...constants import XINFERENCE_MODEL_DIR
    from ..custom import migrate_from_v1_to_v2

    # migrate from v1 to v2 first
    migrate_from_v1_to_v2("embedding", CustomEmbeddingModelFamilyV2)

    user_defined_embedding_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "embedding")
    if os.path.isdir(user_defined_embedding_dir):
        for f in os.listdir(user_defined_embedding_dir):
            try:
                with codecs.open(
                    os.path.join(user_defined_embedding_dir, f), encoding="utf-8"
                ) as fd:
                    user_defined_llm_family = CustomEmbeddingModelFamilyV2.parse_obj(
                        json.load(fd)
                    )
                    register_embedding(user_defined_llm_family, persist=False)
            except Exception as e:
                warnings.warn(f"{user_defined_embedding_dir}/{f} has error, {e}")


def check_format_with_engine(model_format, engine):
    if model_format in ["ggufv2"] and engine not in ["llama.cpp"]:
        return False
    if model_format not in ["ggufv2"] and engine == "llama.cpp":
        return False
    return True


def generate_engine_config_by_model_name(model_family: "EmbeddingModelFamilyV2"):
    model_name = model_family.model_name
    engines: Dict[str, List[Dict[str, Any]]] = EMBEDDING_ENGINES.get(
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
                                "embedding_class": cls,
                            }
                        ]
                    else:
                        engines[engine].append(
                            {
                                "model_name": model_name,
                                "model_format": model_format,
                                "quantization": quantization,
                                "embedding_class": cls,
                            }
                        )
                    break
    EMBEDDING_ENGINES[model_name] = engines


# will be called in xinference/model/__init__.py
def _install():
    _model_spec_json = os.path.join(os.path.dirname(__file__), "model_spec.json")

    for json_obj in json.load(codecs.open(_model_spec_json, "r", encoding="utf-8")):
        flattened = []
        for spec in json_obj["model_specs"]:
            flattened.extend(flatten_quantizations(spec))
        json_obj["model_specs"] = flattened
        BUILTIN_EMBEDDING_MODELS[json_obj["model_name"]] = EmbeddingModelFamilyV2(
            **json_obj
        )

    for model_name, model_spec in BUILTIN_EMBEDDING_MODELS.items():
        if model_spec.model_name not in EMBEDDING_MODEL_DESCRIPTIONS:
            EMBEDDING_MODEL_DESCRIPTIONS.update(
                generate_embedding_description(model_spec)
            )

    from .flag.core import FlagEmbeddingModel
    from .llama_cpp.core import XllamaCppEmbeddingModel
    from .sentence_transformers.core import SentenceTransformerEmbeddingModel
    from .vllm.core import VLLMEmbeddingModel

    SENTENCE_TRANSFORMER_CLASSES.extend([SentenceTransformerEmbeddingModel])
    FLAG_EMBEDDER_CLASSES.extend([FlagEmbeddingModel])
    VLLM_CLASSES.extend([VLLMEmbeddingModel])
    LLAMA_CPP_CLASSES.extend([XllamaCppEmbeddingModel])

    SUPPORTED_ENGINES["sentence_transformers"] = SENTENCE_TRANSFORMER_CLASSES
    SUPPORTED_ENGINES["flag"] = FLAG_EMBEDDER_CLASSES
    SUPPORTED_ENGINES["vllm"] = VLLM_CLASSES
    SUPPORTED_ENGINES["llama.cpp"] = LLAMA_CPP_CLASSES

    # Init embedding engine
    for model_spec in BUILTIN_EMBEDDING_MODELS.values():
        generate_engine_config_by_model_name(model_spec)

    register_custom_model()

    # register model description
    for ud_embedding in get_user_defined_embeddings():
        EMBEDDING_MODEL_DESCRIPTIONS.update(
            generate_embedding_description(ud_embedding)
        )

    del _model_spec_json
