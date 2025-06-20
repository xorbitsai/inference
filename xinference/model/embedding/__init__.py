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

from .core import (
    EMBEDDING_MODEL_DESCRIPTIONS,
    MODEL_NAME_TO_REVISION,
    EmbeddingModelSpec,
    generate_embedding_description,
    get_cache_status,
    get_embedding_model_descriptions,
)
from .custom import (
    CustomEmbeddingModelSpec,
    get_user_defined_embeddings,
    register_embedding,
    unregister_embedding,
)
from .embed_family import (
    BUILTIN_EMBEDDING_MODELS,
    EMBEDDING_ENGINES,
    FLAG_EMBEDDER_CLASSES,
    MODELSCOPE_EMBEDDING_MODELS,
    SENTENCE_TRANSFORMER_CLASSES,
    SUPPORTED_ENGINES,
    VLLM_CLASSES,
)


def register_custom_model():
    from ...constants import XINFERENCE_MODEL_DIR

    user_defined_embedding_dir = os.path.join(XINFERENCE_MODEL_DIR, "embedding")
    if os.path.isdir(user_defined_embedding_dir):
        for f in os.listdir(user_defined_embedding_dir):
            try:
                with codecs.open(
                    os.path.join(user_defined_embedding_dir, f), encoding="utf-8"
                ) as fd:
                    user_defined_llm_family = CustomEmbeddingModelSpec.parse_obj(
                        json.load(fd)
                    )
                    register_embedding(user_defined_llm_family, persist=False)
            except Exception as e:
                warnings.warn(f"{user_defined_embedding_dir}/{f} has error, {e}")


def generate_engine_config_by_model_name(model_spec: "EmbeddingModelSpec"):
    model_name = model_spec.model_name
    engines: Dict[str, List[Dict[str, Any]]] = EMBEDDING_ENGINES.get(
        model_name, {}
    )  # structure for engine query
    for engine in SUPPORTED_ENGINES:
        CLASSES = SUPPORTED_ENGINES[engine]
        for cls in CLASSES:
            # Every engine needs to implement match method
            if cls.match(model_spec):
                # we only match the first class for an engine
                engines[engine] = [
                    {
                        "model_name": model_name,
                        "embedding_class": cls,
                    }
                ]
                break
    EMBEDDING_ENGINES[model_name] = engines


# will be called in xinference/model/__init__.py
def _install():
    _model_spec_json = os.path.join(os.path.dirname(__file__), "model_spec.json")
    _model_spec_modelscope_json = os.path.join(
        os.path.dirname(__file__), "model_spec_modelscope.json"
    )
    ################### HuggingFace Model List Info Init ###################
    BUILTIN_EMBEDDING_MODELS.update(
        dict(
            (spec["model_name"], EmbeddingModelSpec(**spec))
            for spec in json.load(codecs.open(_model_spec_json, "r", encoding="utf-8"))
        )
    )
    for model_name, model_spec in BUILTIN_EMBEDDING_MODELS.items():
        MODEL_NAME_TO_REVISION[model_name].append(model_spec.model_revision)

    ################### ModelScope Model List Info Init ###################
    MODELSCOPE_EMBEDDING_MODELS.update(
        dict(
            (spec["model_name"], EmbeddingModelSpec(**spec))
            for spec in json.load(
                codecs.open(_model_spec_modelscope_json, "r", encoding="utf-8")
            )
        )
    )
    for model_name, model_spec in MODELSCOPE_EMBEDDING_MODELS.items():
        MODEL_NAME_TO_REVISION[model_name].append(model_spec.model_revision)

    # TODO: consider support more download hub in future...
    # register model description after recording model revision
    for model_spec_info in [BUILTIN_EMBEDDING_MODELS, MODELSCOPE_EMBEDDING_MODELS]:
        for model_name, model_spec in model_spec_info.items():
            if model_spec.model_name not in EMBEDDING_MODEL_DESCRIPTIONS:
                EMBEDDING_MODEL_DESCRIPTIONS.update(
                    generate_embedding_description(model_spec)
                )

    from .flag.core import FlagEmbeddingModel
    from .sentence_transformers.core import SentenceTransformerEmbeddingModel
    from .vllm.core import VLLMEmbeddingModel

    SENTENCE_TRANSFORMER_CLASSES.extend([SentenceTransformerEmbeddingModel])
    FLAG_EMBEDDER_CLASSES.extend([FlagEmbeddingModel])
    VLLM_CLASSES.extend([VLLMEmbeddingModel])

    SUPPORTED_ENGINES["sentence_transformers"] = SENTENCE_TRANSFORMER_CLASSES
    SUPPORTED_ENGINES["flag"] = FLAG_EMBEDDER_CLASSES
    SUPPORTED_ENGINES["vllm"] = VLLM_CLASSES

    # Init embedding engine
    for model_infos in [BUILTIN_EMBEDDING_MODELS, MODELSCOPE_EMBEDDING_MODELS]:
        for model_spec in model_infos.values():
            generate_engine_config_by_model_name(model_spec)

    register_custom_model()

    # register model description
    for ud_embedding in get_user_defined_embeddings():
        EMBEDDING_MODEL_DESCRIPTIONS.update(
            generate_embedding_description(ud_embedding)
        )

    del _model_spec_json
    del _model_spec_modelscope_json
