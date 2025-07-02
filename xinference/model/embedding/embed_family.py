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
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, Union

if TYPE_CHECKING:
    from .core import EmbeddingModel, EmbeddingModelFamilyV1, EmbeddingSpecV1

FLAG_EMBEDDER_CLASSES: List[Type["EmbeddingModel"]] = []
SENTENCE_TRANSFORMER_CLASSES: List[Type["EmbeddingModel"]] = []
VLLM_CLASSES: List[Type["EmbeddingModel"]] = []
LLAMA_CPP_CLASSES: List[Type["EmbeddingModel"]] = []

BUILTIN_EMBEDDING_MODELS: Dict[str, Any] = {}
BUILTIN_MODELSCOPE_EMBEDDING_MODELS: Dict[str, Any] = {}

logger = logging.getLogger(__name__)


# Desc: this file used to manage embedding models information.
def match_embedding(
    model_name: str,
    model_format: Optional[str] = None,
    quantization: Optional[str] = None,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
) -> tuple["EmbeddingModelFamilyV1", "EmbeddingSpecV1", str]:
    from ..utils import download_from_modelscope

    # The model info has benn init by __init__.py with model_spec.json file
    from .custom import get_user_defined_embeddings

    # first, check whether it is a user-defined embedding model
    candidate_model_families = []
    for model_family in get_user_defined_embeddings():
        if model_name == model_family.model_name:
            candidate_model_families.append(model_family)
            break

    if (
        download_hub == "modelscope"
        and model_name in BUILTIN_MODELSCOPE_EMBEDDING_MODELS
    ):
        logger.debug(f"Embedding model {model_name} found in ModelScope.")
        candidate_model_families.append(BUILTIN_MODELSCOPE_EMBEDDING_MODELS[model_name])
    elif download_hub == "huggingface" and model_name in BUILTIN_EMBEDDING_MODELS:
        logger.debug(f"Embedding model {model_name} found in Huggingface.")
        candidate_model_families.append(BUILTIN_EMBEDDING_MODELS[model_name])
    elif (
        download_from_modelscope() and model_name in BUILTIN_MODELSCOPE_EMBEDDING_MODELS
    ):
        logger.debug(f"Embedding model {model_name} found in ModelScope.")
        candidate_model_families.append(BUILTIN_MODELSCOPE_EMBEDDING_MODELS[model_name])
    elif model_name in BUILTIN_EMBEDDING_MODELS:
        logger.debug(f"Embedding model {model_name} found in Huggingface.")
        candidate_model_families.append(BUILTIN_EMBEDDING_MODELS[model_name])

    if not candidate_model_families:
        raise ValueError(
            f"Embedding model {model_name} not found, available"
            f"Huggingface: {BUILTIN_EMBEDDING_MODELS.keys()}"
            f"ModelScope: {BUILTIN_MODELSCOPE_EMBEDDING_MODELS.keys()}"
        )

    def _match_quantization(q: Union[str, None], quantizations: List[str]):
        # Currently, the quantization name could include both uppercase and lowercase letters,
        # so it is necessary to ensure that the case sensitivity does not
        # affect the matching results.
        if q is None:
            return q
        for quant in quantizations:
            if q.lower() == quant.lower():
                return quant

    def _apply_format_to_model_id(spec: "EmbeddingSpecV1", q: str) -> "EmbeddingSpecV1":
        # Different quantized versions of some models use different model ids,
        # Here we check the `{}` in the model id to format the id.
        if spec.model_id and "{" in spec.model_id:
            spec.model_id = spec.model_id.format(quantization=q)
        return spec

    for family in candidate_model_families:
        if model_name != family.model_name:
            continue
        for spec in family.model_specs:
            matched_quantization = _match_quantization(quantization, spec.quantizations)
            if (
                model_format
                and model_format != spec.model_format
                or quantization
                and matched_quantization is None
            ):
                continue
            # Copy spec to avoid _apply_format_to_model_id modify the original spec.
            spec = spec.copy()
            if quantization:
                return (
                    family,
                    _apply_format_to_model_id(spec, matched_quantization),
                    matched_quantization,
                )
            else:
                # TODO: If user does not specify quantization, just use the first one
                _q = "none" if spec.model_format == "pytorch" else spec.quantizations[0]
                return family, _apply_format_to_model_id(spec, _q), _q

    raise ValueError(
        f"Embedding model {model_name} not found, available"
        f"Huggingface: {BUILTIN_EMBEDDING_MODELS.keys()}"
        f"ModelScope: {BUILTIN_MODELSCOPE_EMBEDDING_MODELS.keys()}"
    )


# { embedding model name -> { engine name -> engine params } }
EMBEDDING_ENGINES: Dict[str, Dict[str, List[Dict[str, Type["EmbeddingModel"]]]]] = {}
SUPPORTED_ENGINES: Dict[str, List[Type["EmbeddingModel"]]] = {}


def check_engine_by_model_name_and_engine(
    model_engine: str,
    model_name: str,
    model_format: Optional[str],
    quantization: Optional[str],
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
