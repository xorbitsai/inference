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
import logging
import os
import warnings
from typing import Any, Dict, List

from ..utils import flatten_quantizations

logger = logging.getLogger(__name__)


def convert_embedding_model_format(model_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert embedding model hub JSON format to Xinference expected format.
    """
    logger.debug(
        f"convert_embedding_model_format called for: {model_json.get('model_name', 'Unknown')}"
    )

    # Ensure required fields for embedding models
    converted = model_json.copy()

    # Add missing required fields based on EmbeddingModelFamilyV2 requirements
    if "version" not in converted:
        converted["version"] = 2
    if "model_lang" not in converted:
        converted["model_lang"] = ["en"]

    # Handle model_specs
    if "model_specs" not in converted or not converted["model_specs"]:
        converted["model_specs"] = [
            {
                "model_format": "pytorch",
                "model_size_in_billions": None,
                "quantization": "none",
                "model_hub": "huggingface",
            }
        ]
    else:
        # Ensure each spec has required fields
        for spec in converted["model_specs"]:
            if "quantization" not in spec:
                spec["quantization"] = "none"
            if "model_hub" not in spec:
                spec["model_hub"] = "huggingface"

    return converted


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


def register_builtin_model():
    import json

    from ...constants import XINFERENCE_MODEL_DIR
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("embedding")
    existing_model_names = {spec.model_name for spec in registry.get_custom_models()}

    builtin_embedding_dir = os.path.join(
        XINFERENCE_MODEL_DIR, "v2", "builtin", "embedding"
    )
    if os.path.isdir(builtin_embedding_dir):
        # First, try to load from the complete JSON file
        complete_json_path = os.path.join(
            builtin_embedding_dir, "embedding_models.json"
        )
        if os.path.exists(complete_json_path):
            try:
                with codecs.open(complete_json_path, encoding="utf-8") as fd:
                    model_data = json.load(fd)

                # Handle different formats
                models_to_register = []
                if isinstance(model_data, list):
                    # Multiple models in a list
                    models_to_register = model_data
                elif isinstance(model_data, dict):
                    # Single model
                    if "model_name" in model_data:
                        models_to_register = [model_data]
                    else:
                        # Models dict - extract models
                        for key, value in model_data.items():
                            if isinstance(value, dict) and "model_name" in value:
                                models_to_register.append(value)

                # Register all models from the complete JSON
                from .embed_family import BUILTIN_EMBEDDING_MODELS

                for model_data in models_to_register:
                    try:
                        # Convert format if needed (embedding models might have different requirements)
                        converted_data = convert_embedding_model_format(model_data)
                        builtin_embedding_family = EmbeddingModelFamilyV2.parse_obj(
                            converted_data
                        )

                        # Only register if model doesn't already exist
                        if (
                            builtin_embedding_family.model_name
                            not in existing_model_names
                        ):
                            # Add to BUILTIN_EMBEDDING_MODELS directly for proper builtin registration
                            BUILTIN_EMBEDDING_MODELS[
                                builtin_embedding_family.model_name
                            ] = builtin_embedding_family
                            existing_model_names.add(
                                builtin_embedding_family.model_name
                            )
                    except Exception as e:
                        warnings.warn(
                            f"Error parsing embedding model {model_data.get('model_name', 'Unknown')}: {e}"
                        )

                logger.info(
                    f"Successfully registered {len(models_to_register)} embedding models from complete JSON"
                )

            except Exception as e:
                warnings.warn(
                    f"Error loading complete JSON file {complete_json_path}: {e}"
                )
                # Fall back to individual files if complete JSON loading fails

        # Fall back: load individual JSON files (backward compatibility)
        individual_files = [
            f
            for f in os.listdir(builtin_embedding_dir)
            if f.endswith(".json") and f != "embedding_models.json"
        ]
        if individual_files:
            from .embed_family import BUILTIN_EMBEDDING_MODELS
        for f in individual_files:
            try:
                with codecs.open(
                    os.path.join(builtin_embedding_dir, f), encoding="utf-8"
                ) as fd:
                    builtin_embedding_family = EmbeddingModelFamilyV2.parse_obj(
                        json.load(fd)
                    )

                    # Only register if model doesn't already exist
                    if builtin_embedding_family.model_name not in existing_model_names:
                        # Add to BUILTIN_EMBEDDING_MODELS directly for proper builtin registration
                        BUILTIN_EMBEDDING_MODELS[
                            builtin_embedding_family.model_name
                        ] = builtin_embedding_family
                        existing_model_names.add(builtin_embedding_family.model_name)
            except Exception as e:
                warnings.warn(f"{builtin_embedding_dir}/{f} has error, {e}")


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
