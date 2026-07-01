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

import codecs
import json
import logging
import os
import threading
import warnings
from typing import TYPE_CHECKING, Dict, List, Type

if TYPE_CHECKING:
    from .core import CacheableModelSpec

logger = logging.getLogger(__name__)


class ModelRegistry:
    model_type = "unknown"

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.models: List["CacheableModelSpec"] = []
        self.builtin_models: List[str] = []

    def find_model(self, model_name: str):
        model_spec = None
        for f in self.models:
            if f.model_name == model_name:
                model_spec = f
                break
        return model_spec

    def get_custom_models(self):
        with self.lock:
            return self.models.copy()

    def check_model_uri(self, model_spec: "CacheableModelSpec"):
        from .utils import is_valid_model_uri

        model_uri = model_spec.model_uri
        if model_uri and not is_valid_model_uri(model_uri):
            raise ValueError(f"Invalid model URI {model_uri}.")

    def add_ud_model(self, model_spec):
        self.models.append(model_spec)

    def register(self, model_spec: "CacheableModelSpec", persist: bool):
        from .cache_manager import CacheManager
        from .utils import is_valid_model_name

        if not is_valid_model_name(model_spec.model_name):
            raise ValueError(f"Invalid model name {model_spec.model_name}.")

        self.check_model_uri(model_spec)

        with self.lock:
            for model_name in self.builtin_models + [
                spec.model_name for spec in self.models
            ]:
                if model_spec.model_name == model_name:
                    raise ValueError(
                        f"Model name conflicts with existing model {model_spec.model_name}"
                    )

            self.add_ud_model(model_spec)

        if persist:
            cache_manager = CacheManager(model_spec)
            cache_manager.register_custom_model(self.model_type)

    def remove_ud_model(self, model_spec):
        self.models.remove(model_spec)

    def remove_ud_model_files(self, model_spec):
        from .cache_manager import CacheManager

        cache_manager = CacheManager(model_spec)
        cache_manager.unregister_custom_model(self.model_type)

    def unregister(
        self, model_name: str, raise_error: bool = True, remove_file: bool = True
    ):
        with self.lock:
            model_spec = self.find_model(model_name)
            if model_spec:
                self.remove_ud_model(model_spec)
                if remove_file:
                    self.remove_ud_model_files(model_spec)
            else:
                if raise_error:
                    raise ValueError(f"Model {model_name} not found")
                else:
                    logger.warning(
                        f"Custom {self.model_type} model {model_name} not found"
                    )


class RegistryManager:
    _instances: Dict[str, ModelRegistry] = {}

    @classmethod
    def get_registry(cls, model_type: str) -> ModelRegistry:
        from .audio.custom import AudioModelRegistry
        from .embedding.custom import EmbeddingModelRegistry
        from .flexible.custom import FlexibleModelRegistry
        from .image.custom import ImageModelRegistry
        from .llm.custom import LLMModelRegistry
        from .rerank.custom import RerankModelRegistry

        if model_type not in cls._instances:
            if model_type == "rerank":
                cls._instances[model_type] = RerankModelRegistry()
            elif model_type == "image":
                cls._instances[model_type] = ImageModelRegistry()
            elif model_type == "audio":
                cls._instances[model_type] = AudioModelRegistry()
            elif model_type == "llm":
                cls._instances[model_type] = LLMModelRegistry()
            elif model_type == "flexible":
                cls._instances[model_type] = FlexibleModelRegistry()
            elif model_type == "embedding":
                cls._instances[model_type] = EmbeddingModelRegistry()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        return cls._instances[model_type]


def migrate_from_v1_to_v2(model_type: str, model_spec_cls: Type):
    from ..constants import XINFERENCE_MODEL_DIR

    v1_user_defined_model_dir = os.path.join(XINFERENCE_MODEL_DIR, model_type)
    v2_user_defined_model_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", model_type)
    if os.path.isdir(v1_user_defined_model_dir):
        for f in os.listdir(v1_user_defined_model_dir):
            if os.path.exists(os.path.join(v2_user_defined_model_dir, f)):
                # skip if v2 has already
                continue

            try:
                with codecs.open(
                    os.path.join(v1_user_defined_model_dir, f), encoding="utf-8"
                ) as fd:
                    v1_model_json = json.load(fd)

                    v1_model_json["version"] = 2
                    for spec in v1_model_json.get("model_specs", []):
                        if "quantizations" in spec:
                            # change quantizations to quantization
                            spec["quantization"] = spec["quantizations"][0]

                    user_defined_model_family = model_spec_cls(**v1_model_json)
                    registry = RegistryManager.get_registry(model_type)
                    # register custom model file to v2
                    registry.register(user_defined_model_family, persist=True)
                    # unregister since it will be registered by v2
                    registry.unregister(
                        user_defined_model_family.model_name, remove_file=False
                    )
            except Exception as e:
                warnings.warn(
                    f"Fail to migrate {v1_user_defined_model_dir}/{f}, error: {e}"
                )
