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
from typing import Tuple

from pydantic import BaseModel

from ...constants import XINFERENCE_CACHE_DIR
from ..core import ModelDescription
from .stable_diffusion.core import DiffusionModel

MAX_ATTEMPTS = 3

logger = logging.getLogger(__name__)


class MultimodalModelFamilyV1(BaseModel):
    model_family: str
    model_name: str
    model_id: str
    model_revision: str


class MultimodalModelDescription(ModelDescription):
    def __init__(self, model_spec: MultimodalModelFamilyV1):
        self._model_spec = model_spec

    def to_dict(self):
        return {
            "model_type": "multimodal",
            "model_name": self._model_spec.model_name,
            "model_family": self._model_spec.model_family,
            "model_revision": self._model_spec.model_revision,
        }


def match_diffusion(model_name: str) -> MultimodalModelFamilyV1:
    from . import BUILTIN_MULTIMODAL_MODELS

    if model_name in BUILTIN_MULTIMODAL_MODELS:
        return BUILTIN_MULTIMODAL_MODELS[model_name]
    else:
        raise ValueError(
            f"Embedding model {model_name} not found, available"
            f"model list: {BUILTIN_MULTIMODAL_MODELS.keys()}"
        )


def cache(model_spec: MultimodalModelFamilyV1):
    # TODO: cache from uri
    import huggingface_hub

    cache_dir = os.path.realpath(
        os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
    )
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    for current_attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            huggingface_hub.snapshot_download(
                model_spec.model_id,
                revision=model_spec.model_revision,
                local_dir=cache_dir,
                local_dir_use_symlinks=True,
                resume_download=True,
            )
            break
        except huggingface_hub.utils.LocalEntryNotFoundError:
            remaining_attempts = MAX_ATTEMPTS - current_attempt
            logger.warning(
                f"Attempt {current_attempt} failed. Remaining attempts: {remaining_attempts}"
            )
    else:
        raise RuntimeError(
            f"Failed to download model '{model_spec.model_name}' after {MAX_ATTEMPTS} attempts"
        )
    return cache_dir


def create_multimodal_model_instance(
    model_uid: str, model_name: str, **kwargs
) -> Tuple[DiffusionModel, MultimodalModelDescription]:
    model_spec = match_diffusion(model_name)
    model_path = cache(model_spec)
    model = DiffusionModel(model_uid, model_path, **kwargs)
    model_description = MultimodalModelDescription(model_spec)
    return model, model_description
