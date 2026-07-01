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

if TYPE_CHECKING:
    from .core import RerankModel, RerankModelFamilyV2, RerankSpecV1

FLAG_RERANKER_CLASSES: List[Type["RerankModel"]] = []
SENTENCE_TRANSFORMER_CLASSES: List[Type["RerankModel"]] = []
VLLM_CLASSES: List[Type["RerankModel"]] = []
LLAMA_CPP_CLASSES: List[Type["RerankModel"]] = []

BUILTIN_RERANK_MODELS: Dict[str, "RerankModelFamilyV2"] = {}

logger = logging.getLogger(__name__)


def match_rerank(
    model_name: str,
    model_format: Optional[str] = None,
    quantization: Optional[str] = None,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
) -> "RerankModelFamilyV2":
    from ..utils import download_from_modelscope
    from .custom import get_user_defined_reranks

    target_family = None

    if model_name in BUILTIN_RERANK_MODELS:
        model_families = BUILTIN_RERANK_MODELS[model_name]
        # Handle the case where BUILTIN_RERANK_MODELS stores lists
        if isinstance(model_families, list):
            target_family = model_families[0]  # Take the first model family
        else:
            target_family = model_families
    else:
        for model_family in get_user_defined_reranks():
            if model_name == model_family.model_name:
                target_family = model_family
                break

    if target_family is None:
        raise ValueError(
            f"Rerank model {model_name} not found, available models: {BUILTIN_RERANK_MODELS.keys()}"
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

    def _apply_format_to_model_id(_spec: "RerankSpecV1", q: str) -> "RerankSpecV1":
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
        f"Rerank model {model_name} with format {model_format} and quantization {quantization} not found."
    )


# { rerank model name -> { engine name -> engine params } }
RERANK_ENGINES: Dict[str, Dict[str, List[Dict[str, Type["RerankModel"]]]]] = {}
SUPPORTED_ENGINES: Dict[str, List[Type["RerankModel"]]] = {}


def check_engine_by_model_name_and_engine(
    model_engine: str,
    model_name: str,
    model_format: Optional[str],
    quantization: Optional[str],
) -> Type["RerankModel"]:
    def get_model_engine_from_spell(engine_str: str) -> str:
        for engine in RERANK_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    if model_name not in RERANK_ENGINES:
        raise ValueError(f"Model {model_name} not found.")
    model_engine = get_model_engine_from_spell(model_engine)
    if model_engine not in RERANK_ENGINES[model_name]:
        raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
    match_params = RERANK_ENGINES[model_name][model_engine]
    for param in match_params:
        if model_name != param["model_name"]:
            continue
        if (model_format and model_format != param["model_format"]) or (
            quantization and quantization != param["quantization"]
        ):
            continue
        return param["rerank_class"]
    raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")


def check_engine_by_model_name_and_engine_with_virtual_env(
    model_engine: str,
    model_name: str,
    model_format: Optional[str],
    quantization: Optional[str],
    model_family: Optional["RerankModelFamilyV2"] = None,
) -> Type["RerankModel"]:
    from ..utils import _collect_virtualenv_engine_markers

    def get_model_engine_from_spell(engine_str: str) -> str:
        for engine in RERANK_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    if model_family is None:
        if model_name in BUILTIN_RERANK_MODELS:
            model_families = BUILTIN_RERANK_MODELS[model_name]
            model_family = (
                model_families[0]
                if isinstance(model_families, list)
                else model_families
            )
        else:
            from .custom import get_user_defined_reranks

            model_family = next(
                (f for f in get_user_defined_reranks() if f.model_name == model_name),
                None,
            )
    if model_family is None:
        raise ValueError(f"Model {model_name} not found.")

    engine_markers = _collect_virtualenv_engine_markers(model_family)
    if model_name not in RERANK_ENGINES:
        engine_key = get_model_engine_from_spell(model_engine)
        if engine_key.lower() in engine_markers:
            engine_classes = SUPPORTED_ENGINES.get(engine_key, [])
            if engine_classes:
                logger.warning(
                    "Bypassing engine compatibility checks for %s due to virtualenv marker.",
                    engine_key,
                )
                return engine_classes[0]
        raise ValueError(f"Model {model_name} not found.")

    model_engine = get_model_engine_from_spell(model_engine)
    if model_engine not in RERANK_ENGINES[model_name]:
        if model_engine.lower() in engine_markers:
            engine_classes = SUPPORTED_ENGINES.get(model_engine, [])
            if engine_classes:
                logger.warning(
                    "Bypassing engine compatibility checks for %s due to virtualenv marker.",
                    model_engine,
                )
                return engine_classes[0]
        raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")

    match_params = RERANK_ENGINES[model_name][model_engine]
    if match_params:
        return match_params[0]["rerank_class"]

    raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
