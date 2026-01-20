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

import logging
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Type, Union

from ...constants import XINFERENCE_ENABLE_VIRTUAL_ENV

if TYPE_CHECKING:
    from .core import EmbeddingModel, EmbeddingModelFamilyV2, EmbeddingSpecV1

FLAG_EMBEDDER_CLASSES: List[Type["EmbeddingModel"]] = []
SENTENCE_TRANSFORMER_CLASSES: List[Type["EmbeddingModel"]] = []
VLLM_CLASSES: List[Type["EmbeddingModel"]] = []
LLAMA_CPP_CLASSES: List[Type["EmbeddingModel"]] = []

BUILTIN_EMBEDDING_MODELS: Dict[str, List["EmbeddingModelFamilyV2"]] = {}

logger = logging.getLogger(__name__)


def match_embedding(
    model_name: str,
    model_format: Optional[str] = None,
    quantization: Optional[str] = None,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
) -> "EmbeddingModelFamilyV2":
    from ..utils import download_from_modelscope
    from .custom import get_user_defined_embeddings

    target_family = None

    if model_name in BUILTIN_EMBEDDING_MODELS:
        # BUILTIN_EMBEDDING_MODELS stores a list of model families for each model name
        target_family_list = BUILTIN_EMBEDDING_MODELS[model_name]
        # Use the first (latest) model family from the list
        target_family = target_family_list[0] if target_family_list else None
    else:
        for model_family in get_user_defined_embeddings():
            if model_name == model_family.model_name:
                target_family = model_family
                break

    if target_family is None:
        raise ValueError(
            f"Embedding model {model_name} not found, available "
            f"models: {BUILTIN_EMBEDDING_MODELS.keys()}"
        )

    if download_hub == "modelscope" or download_from_modelscope():
        specs = [
            x for x in target_family.model_specs if x.model_hub == "modelscope"
        ] + [x for x in target_family.model_specs if x.model_hub == "huggingface"]
    else:
        specs = [x for x in target_family.model_specs if x.model_hub == "huggingface"]

    def _match_quantization(q: Union[str, None], _quantization: str):
        # Currently, the quantization name could include both uppercase and lowercase letters,
        # so it is necessary to ensure that the case sensitivity does not
        # affect the matching results.
        if q is None:
            return None
        return _quantization if q.lower() == _quantization.lower() else None

    def _apply_format_to_model_id(
        _spec: "EmbeddingSpecV1", q: str
    ) -> "EmbeddingSpecV1":
        # Different quantized versions of some models use different model ids,
        # Here we check the `{}` in the model id to format the id.
        if _spec.model_id and "{" in _spec.model_id:
            _spec.model_id = _spec.model_id.format(quantization=q)
        return _spec

    for spec in specs:
        matched_quantization = _match_quantization(quantization, spec.quantization)
        if (
            model_format
            and model_format != spec.model_format
            or quantization
            and matched_quantization is None
        ):
            continue
        # Copy spec to avoid _apply_format_to_model_id modify the original spec.
        spec = spec.copy()
        _family = target_family.copy()
        if quantization:
            _family.model_specs = [
                _apply_format_to_model_id(spec, matched_quantization)
            ]
            return _family
        else:
            # TODO: If user does not specify quantization, just use the first one
            _q = "none" if spec.model_format == "pytorch" else spec.quantization
            _family.model_specs = [_apply_format_to_model_id(spec, _q)]
            return _family

    raise ValueError(
        f"Embedding model {model_name} with format {model_format} and quantization {quantization} not found."
    )


# { embedding model name -> { engine name -> engine params } }
EMBEDDING_ENGINES: Dict[str, Dict[str, List[Dict[str, Type["EmbeddingModel"]]]]] = {}
SUPPORTED_ENGINES: Dict[str, List[Type["EmbeddingModel"]]] = {}


def check_engine_by_model_name_and_engine(
    model_engine: str,
    model_name: str,
    model_format: Optional[str],
    quantization: Optional[str],
    model_family: Optional["EmbeddingModelFamilyV2"] = None,
    enable_virtual_env: Optional[bool] = None,
) -> Type["EmbeddingModel"]:
    def get_model_engine_from_spell(engine_str: str) -> str:
        for engine in EMBEDDING_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    if enable_virtual_env is None:
        enable_virtual_env = XINFERENCE_ENABLE_VIRTUAL_ENV

    if model_name not in EMBEDDING_ENGINES:
        if enable_virtual_env and model_family is not None:
            from ..utils import (
                _collect_virtualenv_engine_markers,
                _normalize_match_result,
            )

            engine_key = get_model_engine_from_spell(model_engine)
            engine_markers = _collect_virtualenv_engine_markers(model_family)
            if engine_key.lower() in engine_markers:
                engine_classes = SUPPORTED_ENGINES.get(engine_key, [])
                for spec in model_family.model_specs:
                    spec_quant = getattr(spec, "quantization", None) or "none"
                    for cls in engine_classes:
                        match_func = getattr(cls, "match_json", None)
                        if not callable(match_func):
                            continue
                        try:
                            match_res = match_func(model_family, spec, spec_quant)
                        except Exception:
                            match_res = False
                        is_match, _, _, _ = _normalize_match_result(
                            match_res,
                            f"Model {model_name} cannot be run on engine {engine_key}.",
                            "model_compatibility",
                        )
                        if is_match:
                            return cls
        raise ValueError(f"Model {model_name} not found.")
    model_engine = get_model_engine_from_spell(model_engine)
    if model_engine not in EMBEDDING_ENGINES[model_name]:
        if enable_virtual_env and model_family is not None:
            from ..utils import (
                _collect_virtualenv_engine_markers,
                _normalize_match_result,
            )

            engine_key = get_model_engine_from_spell(model_engine)
            engine_markers = _collect_virtualenv_engine_markers(model_family)
            if engine_key.lower() in engine_markers:
                engine_classes = SUPPORTED_ENGINES.get(engine_key, [])
                for spec in model_family.model_specs:
                    spec_quant = getattr(spec, "quantization", None) or "none"
                    for cls in engine_classes:
                        match_func = getattr(cls, "match_json", None)
                        if not callable(match_func):
                            continue
                        try:
                            match_res = match_func(model_family, spec, spec_quant)
                        except Exception:
                            match_res = False
                        is_match, _, _, _ = _normalize_match_result(
                            match_res,
                            f"Model {model_name} cannot be run on engine {engine_key}.",
                            "model_compatibility",
                        )
                        if is_match:
                            return cls
        raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
    match_params = EMBEDDING_ENGINES[model_name][model_engine]
    for param in match_params:
        if model_name != param["model_name"]:
            continue
        if (model_format and model_format != param["model_format"]) or (
            quantization and quantization != param["quantization"]
        ):
            continue
        return param["embedding_class"]
    raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
