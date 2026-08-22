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
import logging
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional

from ..core import CacheableModelSpec, VirtualEnvSettings
from ..utils import ModelInstanceInfoMixin

logger = logging.getLogger(__name__)

WORLD_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)
BUILTIN_WORLD_MODELS: Dict[str, List["WorldModelFamilyV1"]] = {}


def get_world_model_descriptions():
    import copy

    return copy.deepcopy(WORLD_MODEL_DESCRIPTIONS)


class WorldModelFamilyV1(CacheableModelSpec, ModelInstanceInfoMixin):
    version: Literal[1]
    model_family: str
    model_name: str
    model_ability: List[str]
    model_format: str = "pytorch"
    model_engine: Optional[str]
    default_model_config: Optional[Dict[str, Any]]
    default_generate_config: Optional[Dict[str, Any]]
    source_url: str
    source_revision: str
    source_subdir: Optional[str]
    auxiliary_model_id: Optional[str]
    auxiliary_model_revision: Optional[str]
    auxiliary_model_allow_patterns: Optional[List[str]]
    virtualenv: Optional[VirtualEnvSettings]

    class Config:
        extra = "allow"

    def to_description(self):
        return {
            "model_type": "world",
            "address": getattr(self, "address", None),
            "accelerators": getattr(self, "accelerators", None),
            "model_name": self.model_name,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "model_ability": self.model_ability,
            "model_format": self.model_format,
            "model_engine": getattr(self, "model_engine", None),
        }

    def to_version_info(self):
        from ..cache_manager import CacheManager

        cache_manager = CacheManager(self)
        return {
            "model_version": self.model_name,
            "model_file_location": cache_manager.get_cache_dir(),
            "cache_status": cache_manager.get_cache_status(),
        }


def generate_world_description(
    world_model: WorldModelFamilyV1,
) -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    res[world_model.model_name].append(world_model.to_version_info())
    return res


def match_world_model(
    model_name: str,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
) -> WorldModelFamilyV1:
    from ..utils import download_from_modelscope

    if model_name not in BUILTIN_WORLD_MODELS:
        raise ValueError(
            f"World model {model_name} not found, available model list: "
            f"{BUILTIN_WORLD_MODELS.keys()}"
        )

    model_families = BUILTIN_WORLD_MODELS[model_name]
    if download_hub == "modelscope" or (
        download_hub is None and download_from_modelscope()
    ):
        candidates = [x for x in model_families if x.model_hub == "modelscope"]
        if candidates:
            return candidates[0]
    candidates = [x for x in model_families if x.model_hub == "huggingface"]
    return (candidates or model_families)[0]


def resolve_world_model_engine(
    model_name: str, model_engine: Optional[str] = None
) -> Optional[str]:
    from .engine_family import WORLD_ENGINES

    available_engines = WORLD_ENGINES.get(model_name)
    if not available_engines:
        return model_engine
    if model_engine is None:
        return next(iter(available_engines))
    return next(
        (
            engine
            for engine in available_engines
            if engine.lower() == model_engine.lower()
        ),
        model_engine,
    )


def create_world_model_instance(
    model_uid: str,
    model_name: str,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
    model_path: Optional[str] = None,
    model_engine: Optional[str] = None,
    **kwargs,
):
    from ..cache_manager import CacheManager
    from .engine_family import (
        check_engine_by_model_name_and_engine,
        check_engine_by_model_name_and_engine_with_virtual_env,
    )

    enable_virtual_env = kwargs.pop("enable_virtual_env", None)
    model_engine = resolve_world_model_engine(model_name, model_engine)
    model_spec = match_world_model(model_name, download_hub)
    if model_engine is None:
        raise ValueError(f"World model {model_name} has no available engine.")

    if enable_virtual_env is None:
        from ...constants import XINFERENCE_ENABLE_VIRTUAL_ENV

        enable_virtual_env = XINFERENCE_ENABLE_VIRTUAL_ENV
    if enable_virtual_env:
        model_cls = check_engine_by_model_name_and_engine_with_virtual_env(
            model_engine,
            model_name,
            model_family=model_spec,
        )
    else:
        model_cls = check_engine_by_model_name_and_engine(model_engine, model_name)

    if not model_path:
        model_path = CacheManager(model_spec).cache()
    model_spec = model_spec.copy(update={"model_engine": model_engine})
    return model_cls(model_uid, model_path, model_spec, **kwargs)
