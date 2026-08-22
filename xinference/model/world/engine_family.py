# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
    from .core import WorldModelFamilyV1

logger = logging.getLogger(__name__)


class WorldEngineModel:
    """Base class for a world-model runtime implementation."""

    required_libs: Tuple[str, ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def match(cls, model_family: "WorldModelFamilyV1") -> bool:
        raise NotImplementedError

    @classmethod
    def is_model_family_supported(cls, model_family: "WorldModelFamilyV1") -> bool:
        """Whether the engine implements the model, ignoring host constraints."""

        return cls.match(model_family)

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        for lib in cls.required_libs:
            if importlib.util.find_spec(lib) is None:
                return False, f"Library '{lib}' is not installed"
        return True


# {world model name -> {engine name -> engine params}}
WORLD_ENGINES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
SUPPORTED_ENGINES: Dict[str, List[Type[WorldEngineModel]]] = {}


def get_supported_engines_for_model(
    model_families: List["WorldModelFamilyV1"],
) -> Dict[str, List[Type[WorldEngineModel]]]:
    return {
        engine: [
            cls
            for cls in classes
            if any(cls.is_model_family_supported(family) for family in model_families)
        ]
        for engine, classes in SUPPORTED_ENGINES.items()
        if any(
            cls.is_model_family_supported(family)
            for cls in classes
            for family in model_families
        )
    }


def _canonical_engine(model_name: str, model_engine: str) -> str:
    return next(
        (
            engine
            for engine in WORLD_ENGINES.get(model_name, {})
            if engine.lower() == model_engine.lower()
        ),
        model_engine,
    )


def check_engine_by_model_name_and_engine(
    model_engine: str,
    model_name: str,
) -> Type[WorldEngineModel]:
    if model_name not in WORLD_ENGINES:
        raise ValueError(f"World model {model_name} not found.")
    model_engine = _canonical_engine(model_name, model_engine)
    if model_engine not in WORLD_ENGINES[model_name]:
        raise ValueError(
            f"World model {model_name} cannot be run on engine {model_engine}."
        )
    for param in WORLD_ENGINES[model_name][model_engine]:
        if param["model_name"] == model_name:
            return param["world_class"]
    raise ValueError(
        f"World model {model_name} cannot be run on engine {model_engine}."
    )


def check_engine_by_model_name_and_engine_with_virtual_env(
    model_engine: str,
    model_name: str,
    model_family: Optional["WorldModelFamilyV1"] = None,
) -> Type[WorldEngineModel]:
    from ..utils import _collect_virtualenv_engine_markers

    if model_family is None:
        raise ValueError(f"World model {model_name} not found.")

    try:
        return check_engine_by_model_name_and_engine(model_engine, model_name)
    except ValueError:
        engine_markers = _collect_virtualenv_engine_markers(model_family)
        if model_engine.lower() not in engine_markers:
            raise
        for engine, engine_classes in SUPPORTED_ENGINES.items():
            if engine.lower() != model_engine.lower():
                continue
            for engine_class in engine_classes:
                if engine_class.is_model_family_supported(model_family):
                    logger.warning(
                        "Bypassing engine dependency checks for %s due to "
                        "virtualenv marker.",
                        model_engine,
                    )
                    return engine_class
        raise


def generate_engine_config_by_model_name(
    model_family: "WorldModelFamilyV1",
) -> None:
    model_name = model_family.model_name
    engines = WORLD_ENGINES.get(model_name, {})
    for engine, classes in SUPPORTED_ENGINES.items():
        for cls in classes:
            if not cls.match(model_family):
                continue
            engine_params = engines.get(engine, [])
            if not any(param["model_name"] == model_name for param in engine_params):
                engine_params.append(
                    {
                        "model_name": model_name,
                        "model_format": model_family.model_format,
                        "world_class": cls,
                    }
                )
            engines[engine] = engine_params
            break
    if engines:
        WORLD_ENGINES[model_name] = engines
