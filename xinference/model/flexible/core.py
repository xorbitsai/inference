# Copyright 2022-2024 XProbe Inc.
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

import json
import logging
import os
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Optional, Tuple

from ...constants import XINFERENCE_CACHE_DIR, XINFERENCE_MODEL_DIR
from ..core import CacheableModelSpec, ModelDescription
from .utils import get_launcher

logger = logging.getLogger(__name__)

FLEXIBLE_MODEL_LOCK = Lock()


class FlexibleModelSpec(CacheableModelSpec):
    model_id: Optional[str]  # type: ignore
    model_description: Optional[str]
    model_uri: Optional[str]
    launcher: str
    launcher_args: Optional[str]

    def parser_args(self):
        return json.loads(self.launcher_args)


class FlexibleModelDescription(ModelDescription):
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_spec: FlexibleModelSpec,
        model_path: Optional[str] = None,
    ):
        super().__init__(address, devices, model_path=model_path)
        self._model_spec = model_spec

    def to_dict(self):
        return {
            "model_type": "flexible",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._model_spec.model_name,
            "launcher": self._model_spec.launcher,
            "launcher_args": self._model_spec.launcher_args,
        }

    def get_model_version(self) -> str:
        return f"{self._model_spec.model_name}"

    def to_version_info(self):
        return {
            "model_version": self.get_model_version(),
            "cache_status": True,
            "model_file_location": self._model_spec.model_uri,
            "launcher": self._model_spec.launcher,
            "launcher_args": self._model_spec.launcher_args,
        }


def generate_flexible_model_description(
    model_spec: FlexibleModelSpec,
) -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    res[model_spec.model_name].append(
        FlexibleModelDescription(None, None, model_spec).to_version_info()
    )
    return res


FLEXIBLE_MODELS: List[FlexibleModelSpec] = []
FLEXIBLE_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)


def get_flexible_models():
    with FLEXIBLE_MODEL_LOCK:
        return FLEXIBLE_MODELS.copy()


def get_flexible_model_descriptions():
    import copy

    return copy.deepcopy(FLEXIBLE_MODEL_DESCRIPTIONS)


def register_flexible_model(model_spec: FlexibleModelSpec, persist: bool):
    from ..utils import is_valid_model_name, is_valid_model_uri

    if not is_valid_model_name(model_spec.model_name):
        raise ValueError(f"Invalid model name {model_spec.model_name}.")

    model_uri = model_spec.model_uri
    if model_uri and not is_valid_model_uri(model_uri):
        raise ValueError(f"Invalid model URI {model_uri}.")

    if model_spec.launcher_args:
        try:
            model_spec.parser_args()
        except Exception:
            raise ValueError(f"Invalid model launcher args {model_spec.launcher_args}.")

    with FLEXIBLE_MODEL_LOCK:
        for model_name in [spec.model_name for spec in FLEXIBLE_MODELS]:
            if model_spec.model_name == model_name:
                raise ValueError(
                    f"Model name conflicts with existing model {model_spec.model_name}"
                )
        FLEXIBLE_MODELS.append(model_spec)

    if persist:
        persist_path = os.path.join(
            XINFERENCE_MODEL_DIR, "flexible", f"{model_spec.model_name}.json"
        )
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, mode="w") as fd:
            fd.write(model_spec.json())


def unregister_flexible_model(model_name: str, raise_error: bool = True):
    with FLEXIBLE_MODEL_LOCK:
        model_spec = None
        for i, f in enumerate(FLEXIBLE_MODELS):
            if f.model_name == model_name:
                model_spec = f
                break
        if model_spec:
            FLEXIBLE_MODELS.remove(model_spec)

            persist_path = os.path.join(
                XINFERENCE_MODEL_DIR, "flexible", f"{model_spec.model_name}.json"
            )
            if os.path.exists(persist_path):
                os.remove(persist_path)

            cache_dir = os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
            if os.path.exists(cache_dir):
                logger.warning(
                    f"Remove the cache of user-defined model {model_spec.model_name}. "
                    f"Cache directory: {cache_dir}"
                )
                if os.path.islink(cache_dir):
                    os.remove(cache_dir)
                else:
                    logger.warning(
                        f"Cache directory is not a soft link, please remove it manually."
                    )
        else:
            if raise_error:
                raise ValueError(f"Model {model_name} not found")
            else:
                logger.warning(f"Model {model_name} not found")


class FlexibleModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        device: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._config = config

    def load(self):
        """
        Load the model.
        """

    def infer(self, **kwargs):
        """
        Call model to inference.
        """
        raise NotImplementedError("infer method not implemented.")

    @property
    def model_uid(self):
        return self._model_uid

    @property
    def model_path(self):
        return self._model_path

    @property
    def device(self):
        return self._device

    @property
    def config(self):
        return self._config


def match_flexible_model(model_name):
    for model_spec in get_flexible_models():
        if model_name == model_spec.model_name:
            return model_spec


def create_flexible_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[FlexibleModel, FlexibleModelDescription]:
    model_spec = match_flexible_model(model_name)
    if not model_path:
        model_path = model_spec.model_uri
    launcher_name = model_spec.launcher
    launcher_args = model_spec.parser_args()
    kwargs.update(launcher_args)

    model = get_launcher(launcher_name)(
        model_uid=model_uid, model_spec=model_spec, **kwargs
    )

    model_description = FlexibleModelDescription(
        subpool_addr, devices, model_spec, model_path=model_path
    )
    return model, model_description
