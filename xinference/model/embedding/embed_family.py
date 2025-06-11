# Copyright 2022-2025 XProbe Inc.
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

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type

from ..utils import is_valid_model_name

if TYPE_CHECKING:
    from .core import EmbeddingModel, EmbeddingModelSpec

FLAG_EMBEDDER_CLASSES: List[Type["EmbeddingModel"]] = []
SENTENCE_TRANSFORMER_CLASSES: List[Type["EmbeddingModel"]] = []

BUILTIN_EMBEDDING_MODELS: Dict[str, Any] = {}
MODELSCOPE_EMBEDDING_MODELS: Dict[str, Any] = {}

logger = logging.getLogger(__name__)


# Desc: this file used to manage embedding models information.
def match_embedding(
    model_name: str,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
) -> "EmbeddingModelSpec":
    from ..utils import download_from_modelscope

    # The model info has benn init by __init__.py with model_spec.json file
    from .custom import get_user_defined_embeddings

    # first, check whether it is a user-defined embedding model
    for model_spec in get_user_defined_embeddings():
        if model_name == model_spec.model_name:
            return model_spec

    if download_hub == "modelscope" and model_name in MODELSCOPE_EMBEDDING_MODELS:
        logger.debug(f"Embedding model {model_name} found in ModelScope.")
        return MODELSCOPE_EMBEDDING_MODELS[model_name]
    elif download_hub == "huggingface" and model_name in BUILTIN_EMBEDDING_MODELS:
        logger.debug(f"Embedding model {model_name} found in Huggingface.")
        return BUILTIN_EMBEDDING_MODELS[model_name]
    elif download_from_modelscope() and model_name in MODELSCOPE_EMBEDDING_MODELS:
        logger.debug(f"Embedding model {model_name} found in ModelScope.")
        return MODELSCOPE_EMBEDDING_MODELS[model_name]
    elif model_name in BUILTIN_EMBEDDING_MODELS:
        logger.debug(f"Embedding model {model_name} found in Huggingface.")
        return BUILTIN_EMBEDDING_MODELS[model_name]
    else:
        raise ValueError(
            f"Embedding model {model_name} not found, available"
            f"Huggingface: {BUILTIN_EMBEDDING_MODELS.keys()}"
            f"ModelScope: {MODELSCOPE_EMBEDDING_MODELS.keys()}"
        )


from threading import Lock

# { embedding model name -> { engine name -> engine params } }
EMBEDDING_ENGINES: Dict[str, Dict[str, List[Dict[str, Type["EmbeddingModel"]]]]] = {}
SUPPORTED_ENGINES: Dict[str, List[Type["EmbeddingModel"]]] = {}
UD_EMBEDDING_FAMILIES_LOCK = Lock()
# user defined embedding models
UD_EMBEDDING_SPECS: Dict[str, "EmbeddingModelSpec"] = {}


def register_embedding(custom_embedding_spec: "EmbeddingModelSpec", persist: bool):
    from ..utils import is_valid_model_uri
    from . import generate_engine_config_by_model_name

    if not is_valid_model_name(custom_embedding_spec.model_name):
        raise ValueError(f"Invalid model name {custom_embedding_spec.model_name}.")

    model_uri = custom_embedding_spec.model_uri
    if model_uri and not is_valid_model_uri(model_uri):
        raise ValueError(f"Invalid model URI {model_uri}.")

    with UD_EMBEDDING_FAMILIES_LOCK:
        if (
            custom_embedding_spec.model_name in BUILTIN_EMBEDDING_MODELS
            or custom_embedding_spec.model_name in MODELSCOPE_EMBEDDING_MODELS
            or custom_embedding_spec.model_name in UD_EMBEDDING_SPECS
        ):
            raise ValueError(
                f"Model name conflicts with existing model {custom_embedding_spec.model_name}"
            )

    UD_EMBEDDING_SPECS[custom_embedding_spec.model_name] = custom_embedding_spec
    generate_engine_config_by_model_name(custom_embedding_spec)


# TODO: add persist feature
def unregister_embedding(custom_embedding_spec: "EmbeddingModelSpec"):
    with UD_EMBEDDING_FAMILIES_LOCK:
        model_name = custom_embedding_spec.model_name
        if model_name in UD_EMBEDDING_SPECS:
            del UD_EMBEDDING_SPECS[model_name]
        if model_name in EMBEDDING_ENGINES:
            del EMBEDDING_ENGINES[model_name]


def check_engine_by_model_name_and_engine(
    model_name: str,
    model_engine: str,
) -> Type["EmbeddingModel"]:
    def get_model_engine_from_spell(engine_str: str) -> str:
        for engine in EMBEDDING_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    if model_name not in EMBEDDING_ENGINES:
        raise ValueError(f"Model {model_name} not found.")
    model_engine = get_model_engine_from_spell(model_engine)
    if model_engine not in EMBEDDING_ENGINES[model_name]:
        raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
    match_params = EMBEDDING_ENGINES[model_name][model_engine]
    for param in match_params:
        if model_name == param["model_name"]:
            return param["embedding_class"]
    raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
