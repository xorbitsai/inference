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
import threading
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional

from ..core import CacheableModelSpec, VirtualEnvSettings
from ..utils import ModelInstanceInfoMixin
from .engine_family import VideoEngineModel

logger = logging.getLogger(__name__)

VIDEO_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)
BUILTIN_VIDEO_MODELS: Dict[str, List["VideoModelFamilyV2"]] = {}
VIDEO_REGISTRY_LOCK = threading.RLock()


def get_video_model_descriptions():
    import copy

    with VIDEO_REGISTRY_LOCK:
        return copy.deepcopy(VIDEO_MODEL_DESCRIPTIONS)


class VideoModelFamilyV2(CacheableModelSpec, ModelInstanceInfoMixin):
    version: Literal[2]
    model_family: str
    model_name: str
    model_id: str
    model_revision: str
    model_hub: str = "huggingface"
    model_ability: Optional[List[str]]
    default_model_config: Optional[Dict[str, Any]]
    default_generate_config: Optional[Dict[str, Any]]
    engine: Optional[str]
    model_engine: Optional[str] = None
    model_format: Optional[str]
    cache_name: Optional[str]
    gguf_model_id: Optional[str]
    gguf_quantizations: Optional[List[str]]
    gguf_model_file_name_template: Optional[str]
    lightning_model_id: Optional[str]
    lightning_model_revision: Optional[str]
    lightning_versions: Optional[List[str]]
    lightning_model_file_name_template: Optional[str]
    lightning_version_configs: Optional[Dict[str, Dict[str, Any]]]
    text_encoder_model_id: Optional[str] = None
    text_encoder_model_revision: Optional[str] = None
    virtualenv: Optional[VirtualEnvSettings]

    class Config:
        extra = "allow"

    def to_description(self):
        return {
            "model_type": "video",
            "address": getattr(self, "address", None),
            "accelerators": getattr(self, "accelerators", None),
            "model_name": self.model_name,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "model_ability": self.model_ability,
            "model_engine": getattr(self, "model_engine", None),
        }

    def to_version_info(self):
        from ..cache_manager import CacheManager

        cache_manager = CacheManager(self)

        return {
            "model_version": self.cache_name or self.model_name,
            "model_file_location": cache_manager.get_cache_dir(),
            "cache_status": cache_manager.get_cache_status(),
        }


def resolve_video_model_name_and_engine(
    model_name: str,
    model_engine: Optional[str] = None,
    use_default_engine: bool = False,
) -> tuple[str, Optional[str]]:
    if use_default_engine or model_engine is not None:
        from .engine_family import VIDEO_ENGINES

        with VIDEO_REGISTRY_LOCK:
            available_engines = VIDEO_ENGINES.get(model_name)
            available_engines = dict(available_engines) if available_engines else None
        if available_engines and model_engine is None:
            model_engine = next(iter(available_engines))
        elif available_engines and model_engine is not None:
            model_engine = next(
                (
                    engine
                    for engine in available_engines
                    if engine.lower() == model_engine.lower()
                ),
                model_engine,
            )
    return model_name, model_engine


def generate_video_description(
    video_model: VideoModelFamilyV2,
) -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    res[video_model.model_name].append(video_model.to_version_info())
    return res


def match_diffusion(
    model_name: str,
    download_hub: Optional[
        Literal["auto", "huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
    model_engine: Optional[str] = None,
) -> VideoModelFamilyV2:
    from ..utils import download_from_modelscope
    from . import BUILTIN_VIDEO_MODELS

    with VIDEO_REGISTRY_LOCK:
        model_families = list(BUILTIN_VIDEO_MODELS.get(model_name, []))
        available_model_names = list(BUILTIN_VIDEO_MODELS)

    if not model_families:
        raise ValueError(
            f"Video model {model_name} not found, available"
            f"model list: {available_model_names}"
        )

    if model_engine is not None:
        model_families = [
            family
            for family in model_families
            if (family.engine or "").lower() == model_engine.lower()
        ]
        if not model_families:
            raise ValueError(
                f"Video model {model_name} cannot be run on engine {model_engine}."
            )

    preferred_hub = (
        "modelscope"
        if download_hub == "modelscope"
        or (download_hub in (None, "auto") and download_from_modelscope())
        else "huggingface"
    )
    hub_families = [
        family for family in model_families if family.model_hub == preferred_hub
    ]
    if not hub_families:
        engine_suffix = f" on engine {model_engine}" if model_engine else ""
        raise ValueError(
            f"Video model {model_name}{engine_suffix} does not provide a "
            f"{preferred_hub} source. Choose a supported download hub explicitly."
        )
    return hub_families[0]


def create_video_model_instance(
    model_uid: str,
    model_name: str,
    download_hub: Optional[
        Literal["auto", "huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
    model_path: Optional[str] = None,
    gguf_quantization: Optional[str] = None,
    gguf_model_path: Optional[str] = None,
    model_engine: Optional[str] = None,
    model_format: Optional[str] = None,
    quantization: Optional[str] = None,
    lightning_version: Optional[str] = None,
    lightning_model_path: Optional[str] = None,
    **kwargs,
) -> VideoEngineModel:
    from .cache_manager import VideoCacheManager
    from .engine_family import (
        VIDEO_ENGINES,
        check_engine_by_model_name_and_engine,
        check_engine_by_model_name_and_engine_with_virtual_env,
    )

    enable_virtual_env = kwargs.pop("enable_virtual_env", None)
    model_name, model_engine = resolve_video_model_name_and_engine(
        model_name, model_engine, use_default_engine=True
    )
    model_spec = match_diffusion(model_name, download_hub, model_engine=model_engine)

    if model_engine is None:
        available_engines = VIDEO_ENGINES.get(model_name, {})
        model_engine = next(iter(available_engines), "diffusers")

    if enable_virtual_env is None:
        from ...constants import XINFERENCE_ENABLE_VIRTUAL_ENV

        enable_virtual_env = XINFERENCE_ENABLE_VIRTUAL_ENV

    if enable_virtual_env:
        model_cls = check_engine_by_model_name_and_engine_with_virtual_env(
            model_engine,
            model_spec.model_name,
            model_format,
            quantization,
            model_family=model_spec,
        )
    else:
        model_cls = check_engine_by_model_name_and_engine(
            model_engine,
            model_spec.model_name,
            model_format,
            quantization,
        )

    if not model_path:
        cache_manager = VideoCacheManager(model_spec)
        model_path = cache_manager.cache()
    if not gguf_model_path and gguf_quantization:
        cache_manager = VideoCacheManager(model_spec)
        gguf_model_path = cache_manager.cache_gguf(gguf_quantization)
    if (
        lightning_version or lightning_model_path
    ) and not model_spec.lightning_versions:
        raise ValueError(f"Model {model_name} does not support lightning acceleration")
    if not lightning_model_path and lightning_version:
        cache_manager = VideoCacheManager(model_spec)
        lightning_model_path = cache_manager.cache_lightning(lightning_version)
    assert model_path is not None

    model_spec = model_spec.copy()
    model_spec.model_engine = model_engine
    if model_engine.lower() == "mlx":
        kwargs["_xinference_enable_virtual_env"] = enable_virtual_env
    model = model_cls(
        model_uid,
        model_path,
        model_spec,
        gguf_model_path=gguf_model_path,
        lightning_version=lightning_version,
        lightning_model_path=lightning_model_path,
        **kwargs,
    )
    return model
