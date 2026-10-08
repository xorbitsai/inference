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
    from .core import VideoModelFamilyV2

logger = logging.getLogger(__name__)


class VideoEngineModel:
    required_libs: Tuple[str, ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def match(cls, model_family: "VideoModelFamilyV2") -> bool:
        raise NotImplementedError

    @classmethod
    def is_model_family_supported(cls, model_family: "VideoModelFamilyV2") -> bool:
        """Whether this engine implements the model, ignoring host constraints."""

        return cls.match(model_family)

    @classmethod
    def check_host(cls) -> Union[bool, Tuple[bool, str]]:
        """Check non-installable host constraints before virtualenv fallback."""

        return True

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        for lib in cls.required_libs:
            if importlib.util.find_spec(lib) is None:
                return False, f"Library '{lib}' is not installed"
        return True


# { video model name -> { engine name -> engine params } }
VIDEO_ENGINES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
SUPPORTED_ENGINES: Dict[str, List[Type[VideoEngineModel]]] = {}


class VideoEngineNotAvailableError(ValueError):
    """The engine registry has no entry that a virtualenv may provide."""


class VideoEngineConfigurationError(ValueError):
    """The requested engine tuple or host is intrinsically incompatible."""


def get_supported_engines_for_model(
    model_families: List["VideoModelFamilyV2"],
) -> Dict[str, List[Type[VideoEngineModel]]]:
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


def _normalize_engine_name(model_name: str, model_engine: str) -> str:
    from .core import VIDEO_REGISTRY_LOCK

    with VIDEO_REGISTRY_LOCK:
        engines = list(VIDEO_ENGINES.get(model_name, {}))
    for engine in engines:
        if engine.lower() == model_engine.lower():
            return engine
    return model_engine


def check_engine_by_model_name_and_engine(
    model_engine: str,
    model_name: str,
    model_format: Optional[str] = None,
    quantization: Optional[str] = None,
) -> Type[VideoEngineModel]:
    from .core import VIDEO_REGISTRY_LOCK

    with VIDEO_REGISTRY_LOCK:
        model_engines = VIDEO_ENGINES.get(model_name)
        model_engines = dict(model_engines) if model_engines is not None else None
    if model_engines is None:
        raise VideoEngineNotAvailableError(f"Video model {model_name} not found.")
    model_engine = _normalize_engine_name(model_name, model_engine)
    if model_engine not in model_engines:
        raise VideoEngineNotAvailableError(
            f"Video model {model_name} cannot be run on engine {model_engine}."
        )
    for param in model_engines[model_engine]:
        if model_name != param["model_name"]:
            continue
        if (model_format and model_format != param["model_format"]) or (
            quantization and quantization != param["quantization"]
        ):
            continue
        engine_class = param["video_class"]
        host_result = engine_class.check_host()
        if host_result is not True:
            if isinstance(host_result, tuple) and len(host_result) >= 2:
                raise VideoEngineConfigurationError(str(host_result[1]))
            raise VideoEngineConfigurationError(
                f"Engine {model_engine} is incompatible with the current host"
            )
        return engine_class
    raise VideoEngineConfigurationError(
        f"Video model {model_name} cannot be run on engine {model_engine}."
    )


def check_engine_by_model_name_and_engine_with_virtual_env(
    model_engine: str,
    model_name: str,
    model_format: Optional[str] = None,
    quantization: Optional[str] = None,
    model_family: Optional["VideoModelFamilyV2"] = None,
) -> Type[VideoEngineModel]:
    from ..utils import _collect_virtualenv_engine_markers

    if model_family is None:
        raise ValueError(f"Video model {model_name} not found.")

    expected_format = getattr(model_family, "model_format", None)
    expected_quantization = getattr(model_family, "quantization", None) or "none"
    if (model_format and model_format != expected_format) or (
        quantization and quantization != expected_quantization
    ):
        raise VideoEngineConfigurationError(
            f"Video model {model_name} cannot be run on engine {model_engine} "
            f"with format {model_format or expected_format} and quantization "
            f"{quantization or expected_quantization}."
        )

    engine_markers = _collect_virtualenv_engine_markers(model_family)

    def _host_error(engine_class: Type[VideoEngineModel]) -> Optional[str]:
        result = engine_class.check_host()
        if result is True:
            return None
        if isinstance(result, tuple) and len(result) >= 2:
            return str(result[1])
        return f"Engine {model_engine} is incompatible with the current host"

    def _engine_class_by_marker() -> Optional[Type[VideoEngineModel]]:
        if model_engine.lower() not in engine_markers:
            return None
        for engine, engine_classes in SUPPORTED_ENGINES.items():
            if engine.lower() != model_engine.lower():
                continue
            for engine_class in engine_classes:
                if engine_class.is_model_family_supported(model_family):
                    logger.warning(
                        "Bypassing engine compatibility checks for %s due to "
                        "virtualenv marker.",
                        model_engine,
                    )
                    return engine_class
        return None

    try:
        engine_class = check_engine_by_model_name_and_engine(
            model_engine, model_name, model_format, quantization
        )
    except VideoEngineNotAvailableError:
        marker_engine_class = _engine_class_by_marker()
        if marker_engine_class is None:
            raise
        engine_class = marker_engine_class

    host_error = _host_error(engine_class)
    if host_error is not None:
        raise VideoEngineConfigurationError(host_error)
    return engine_class


def generate_engine_config_by_model_name(
    model_family: "VideoModelFamilyV2",
    target_engines: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None,
) -> None:
    model_name = model_family.model_name
    model_format = getattr(model_family, "model_format", None)
    quantization = getattr(model_family, "quantization", None)
    registry = VIDEO_ENGINES if target_engines is None else target_engines
    engines: Dict[str, List[Dict[str, Any]]] = registry.get(model_name, {})
    for engine, classes in SUPPORTED_ENGINES.items():
        for cls in classes:
            if not cls.match(model_family):
                continue
            engine_params = engines.get(engine, [])
            engine_model_format = getattr(cls, "engine_model_format", model_format)
            engine_quantization = quantization
            if engine_quantization is None:
                engine_quantization = getattr(cls, "engine_quantization", None)
            param = {
                "model_name": model_name,
                "model_format": engine_model_format,
                "quantization": engine_quantization,
                "video_class": cls,
            }
            if not any(
                existing["model_name"] == model_name
                and existing["model_format"] == engine_model_format
                and existing["quantization"] == engine_quantization
                for existing in engine_params
            ):
                engine_params.append(param)
            engines[engine] = engine_params
            break
    if engines:
        if target_engines is None:
            from .core import VIDEO_REGISTRY_LOCK

            with VIDEO_REGISTRY_LOCK:
                VIDEO_ENGINES[model_name] = engines
        else:
            target_engines[model_name] = engines
