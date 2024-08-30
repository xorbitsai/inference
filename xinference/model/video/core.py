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
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

from ...constants import XINFERENCE_CACHE_DIR
from ..core import CacheableModelSpec, ModelDescription
from ..utils import valid_model_revision
from .diffusers import DiffUsersVideoModel

MAX_ATTEMPTS = 3

logger = logging.getLogger(__name__)

MODEL_NAME_TO_REVISION: Dict[str, List[str]] = defaultdict(list)
VIDEO_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)
BUILTIN_VIDEO_MODELS: Dict[str, "VideoModelFamilyV1"] = {}
MODELSCOPE_VIDEO_MODELS: Dict[str, "VideoModelFamilyV1"] = {}


def get_video_model_descriptions():
    import copy

    return copy.deepcopy(VIDEO_MODEL_DESCRIPTIONS)


class VideoModelFamilyV1(CacheableModelSpec):
    model_family: str
    model_name: str
    model_id: str
    model_revision: str
    model_hub: str = "huggingface"
    model_ability: Optional[List[str]]
    default_model_config: Optional[Dict[str, Any]]
    default_generate_config: Optional[Dict[str, Any]]


class VideoModelDescription(ModelDescription):
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_spec: VideoModelFamilyV1,
        model_path: Optional[str] = None,
    ):
        super().__init__(address, devices, model_path=model_path)
        self._model_spec = model_spec

    def to_dict(self):
        return {
            "model_type": "video",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._model_spec.model_name,
            "model_family": self._model_spec.model_family,
            "model_revision": self._model_spec.model_revision,
            "model_ability": self._model_spec.model_ability,
        }

    def to_version_info(self):
        if self._model_path is None:
            is_cached = get_cache_status(self._model_spec)
            file_location = get_cache_dir(self._model_spec)
        else:
            is_cached = True
            file_location = self._model_path

        return [
            {
                "model_version": self._model_spec.model_name,
                "model_file_location": file_location,
                "cache_status": is_cached,
            }
        ]


def generate_video_description(
    video_model: VideoModelFamilyV1,
) -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    res[video_model.model_name].extend(
        VideoModelDescription(None, None, video_model).to_version_info()
    )
    return res


def match_diffusion(
    model_name: str,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
) -> VideoModelFamilyV1:
    from ..utils import download_from_modelscope
    from . import BUILTIN_VIDEO_MODELS, MODELSCOPE_VIDEO_MODELS

    if download_hub == "modelscope" and model_name in MODELSCOPE_VIDEO_MODELS:
        logger.debug(f"Video model {model_name} found in ModelScope.")
        return MODELSCOPE_VIDEO_MODELS[model_name]
    elif download_hub == "huggingface" and model_name in BUILTIN_VIDEO_MODELS:
        logger.debug(f"Video model {model_name} found in Huggingface.")
        return BUILTIN_VIDEO_MODELS[model_name]
    elif download_from_modelscope() and model_name in MODELSCOPE_VIDEO_MODELS:
        logger.debug(f"Video model {model_name} found in ModelScope.")
        return MODELSCOPE_VIDEO_MODELS[model_name]
    elif model_name in BUILTIN_VIDEO_MODELS:
        logger.debug(f"Video model {model_name} found in Huggingface.")
        return BUILTIN_VIDEO_MODELS[model_name]
    else:
        raise ValueError(
            f"Video model {model_name} not found, available"
            f"model list: {BUILTIN_VIDEO_MODELS.keys()}"
        )


def cache(model_spec: VideoModelFamilyV1):
    from ..utils import cache

    return cache(model_spec, VideoModelDescription)


def get_cache_dir(model_spec: VideoModelFamilyV1):
    return os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name))


def get_cache_status(
    model_spec: VideoModelFamilyV1,
) -> bool:
    cache_dir = get_cache_dir(model_spec)
    meta_path = os.path.join(cache_dir, "__valid_download")

    model_name = model_spec.model_name
    if model_name in BUILTIN_VIDEO_MODELS and model_name in MODELSCOPE_VIDEO_MODELS:
        hf_spec = BUILTIN_VIDEO_MODELS[model_name]
        ms_spec = MODELSCOPE_VIDEO_MODELS[model_name]

        return any(
            [
                valid_model_revision(meta_path, hf_spec.model_revision),
                valid_model_revision(meta_path, ms_spec.model_revision),
            ]
        )
    else:  # Usually for UT
        return valid_model_revision(meta_path, model_spec.model_revision)


def create_video_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[DiffUsersVideoModel, VideoModelDescription]:
    model_spec = match_diffusion(model_name, download_hub)
    if not model_path:
        model_path = cache(model_spec)
    assert model_path is not None

    model = DiffUsersVideoModel(
        model_uid,
        model_path,
        model_spec,
        **kwargs,
    )
    model_description = VideoModelDescription(
        subpool_addr, devices, model_spec, model_path=model_path
    )
    return model, model_description
