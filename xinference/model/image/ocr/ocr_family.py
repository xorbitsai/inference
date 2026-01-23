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

import importlib.util
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

logger = logging.getLogger(__name__)


class OCRModel:
    required_libs: Tuple[str, ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        raise NotImplementedError

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        for lib in cls.required_libs:
            if importlib.util.find_spec(lib) is None:
                return False, f"Library '{lib}' is not installed"
        return True


# { ocr model name -> { engine name -> engine params } }
OCR_ENGINES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
SUPPORTED_ENGINES: Dict[str, List[Type[OCRModel]]] = {}


def check_engine_by_model_name_and_engine(
    model_engine: str,
    model_name: str,
    model_format: Optional[str],
    quantization: Optional[str],
) -> Type[OCRModel]:
    def get_model_engine_from_spell(engine_str: str) -> str:
        for engine in OCR_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    if model_name not in OCR_ENGINES:
        raise ValueError(f"Model {model_name} not found.")
    model_engine = get_model_engine_from_spell(model_engine)
    if model_engine not in OCR_ENGINES[model_name]:
        raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
    match_params = OCR_ENGINES[model_name][model_engine]
    for param in match_params:
        if model_name != param["model_name"]:
            continue
        if (model_format and model_format != param["model_format"]) or (
            quantization and quantization != param["quantization"]
        ):
            continue
        return param["ocr_class"]
    raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")


def check_engine_by_model_name_and_engine_with_virtual_env(
    model_engine: str,
    model_name: str,
    model_format: Optional[str],
    quantization: Optional[str],
    model_family: Optional["ImageModelFamilyV2"] = None,
) -> Type[OCRModel]:
    from ...utils import _collect_virtualenv_engine_markers

    def get_model_engine_from_spell(engine_str: str) -> str:
        for engine in OCR_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    if model_family is None:
        raise ValueError(f"Model {model_name} not found.")

    engine_markers = _collect_virtualenv_engine_markers(model_family)
    if model_name not in OCR_ENGINES:
        if model_engine.lower() in engine_markers:
            engine_classes = SUPPORTED_ENGINES.get(model_engine, [])
            if engine_classes:
                logger.warning(
                    "Bypassing engine compatibility checks for %s due to virtualenv marker.",
                    model_engine,
                )
                return engine_classes[0]
        raise ValueError(f"Model {model_name} not found.")

    model_engine = get_model_engine_from_spell(model_engine)
    if model_engine not in OCR_ENGINES[model_name]:
        if model_engine.lower() in engine_markers:
            engine_classes = SUPPORTED_ENGINES.get(model_engine, [])
            if engine_classes:
                logger.warning(
                    "Bypassing engine compatibility checks for %s due to virtualenv marker.",
                    model_engine,
                )
                return engine_classes[0]
        raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")

    match_params = OCR_ENGINES[model_name][model_engine]
    if match_params:
        return match_params[0]["ocr_class"]

    raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")


def generate_engine_config_by_model_name(model_family: "ImageModelFamilyV2") -> None:
    model_name = model_family.model_name
    model_format = getattr(model_family, "model_format", None)
    quantization = getattr(model_family, "quantization", None)
    engines: Dict[str, List[Dict[str, Any]]] = OCR_ENGINES.get(model_name, {})
    for engine, classes in SUPPORTED_ENGINES.items():
        for cls in classes:
            if cls.match(model_family):
                engine_params = engines.get(engine, [])
                engine_params.append(
                    {
                        "model_name": model_name,
                        "model_format": model_format,
                        "quantization": quantization,
                        "ocr_class": cls,
                    }
                )
                engines[engine] = engine_params
                break
    if engines:
        OCR_ENGINES[model_name] = engines
