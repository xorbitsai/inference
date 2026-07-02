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
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class AudioEngineModel:
    required_libs: Tuple[str, ...] = ()

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        raise NotImplementedError

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        for lib in cls.required_libs:
            if importlib.util.find_spec(lib) is None:
                return False, f"Library '{lib}' is not installed"
        return True


# { audio model name -> { engine name -> engine params } }
AUDIO_ENGINES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
SUPPORTED_ENGINES: Dict[str, List[Type[AudioEngineModel]]] = {}


def check_engine_by_model_name_and_engine(
    model_engine: str,
    model_name: str,
) -> Type[AudioEngineModel]:
    def get_model_engine_from_spell(engine_str: str) -> str:
        for engine in AUDIO_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    if model_name not in AUDIO_ENGINES:
        raise ValueError(f"Audio model {model_name} not found.")
    model_engine = get_model_engine_from_spell(model_engine)
    if model_engine not in AUDIO_ENGINES[model_name]:
        raise ValueError(
            f"Audio model {model_name} cannot be run on engine {model_engine}."
        )
    match_params = AUDIO_ENGINES[model_name][model_engine]
    for param in match_params:
        if model_name == param["model_name"]:
            return param["audio_class"]
    raise ValueError(
        f"Audio model {model_name} cannot be run on engine {model_engine}."
    )


def check_engine_by_model_name_and_engine_with_virtual_env(
    model_engine: str,
    model_name: str,
    model_family: Optional["AudioModelFamilyV2"] = None,
) -> Type[AudioEngineModel]:
    from ..utils import _collect_virtualenv_engine_markers

    if model_family is None:
        raise ValueError(f"Audio model {model_name} not found.")

    engine_markers = _collect_virtualenv_engine_markers(model_family)

    def _engine_class_by_marker() -> Optional[Type[AudioEngineModel]]:
        if model_engine.lower() in engine_markers:
            for engine, engine_classes in SUPPORTED_ENGINES.items():
                if engine.lower() == model_engine.lower() and engine_classes:
                    logger.warning(
                        "Bypassing engine compatibility checks for %s due to "
                        "virtualenv marker.",
                        model_engine,
                    )
                    return engine_classes[0]
        return None

    if model_name not in AUDIO_ENGINES:
        engine_cls = _engine_class_by_marker()
        if engine_cls is not None:
            return engine_cls
        raise ValueError(f"Audio model {model_name} not found.")

    try:
        return check_engine_by_model_name_and_engine(model_engine, model_name)
    except ValueError:
        engine_cls = _engine_class_by_marker()
        if engine_cls is not None:
            return engine_cls
        raise


def generate_engine_config_by_model_name(model_family: "AudioModelFamilyV2") -> None:
    model_name = model_family.model_name
    engines: Dict[str, List[Dict[str, Any]]] = AUDIO_ENGINES.get(model_name, {})
    for engine, classes in SUPPORTED_ENGINES.items():
        for cls in classes:
            if cls.match(model_family):
                engine_params = engines.get(engine, [])
                already_exists = any(
                    param["model_name"] == model_name for param in engine_params
                )
                if not already_exists:
                    engine_params.append(
                        {
                            "model_name": model_name,
                            "audio_class": cls,
                        }
                    )
                engines[engine] = engine_params
                break
    if engines:
        AUDIO_ENGINES[model_name] = engines
