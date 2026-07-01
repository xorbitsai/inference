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
import logging
from typing import List, Literal

from ..custom import ModelRegistry
from .core import RerankModelFamilyV2

logger = logging.getLogger(__name__)


class CustomRerankModelFamilyV2(RerankModelFamilyV2):
    version: Literal[2] = 2


UD_RERANKS: List[CustomRerankModelFamilyV2] = []


class RerankModelRegistry(ModelRegistry):
    model_type = "rerank"

    def __init__(self):
        from .rerank_family import BUILTIN_RERANK_MODELS

        super().__init__()
        self.models = UD_RERANKS
        self.builtin_models = list(BUILTIN_RERANK_MODELS.keys())

    def add_ud_model(self, model_spec):
        from . import generate_engine_config_by_model_name

        UD_RERANKS.append(model_spec)
        generate_engine_config_by_model_name(model_spec)

    def check_model_uri(self, model_family: "RerankModelFamilyV2"):
        from ..utils import is_valid_model_uri

        for spec in model_family.model_specs:
            model_uri = spec.model_uri
            if model_uri and not is_valid_model_uri(model_uri):
                raise ValueError(f"Invalid model URI {model_uri}.")

    def remove_ud_model(self, model_family: "CustomRerankModelFamilyV2"):
        from .rerank_family import RERANK_ENGINES

        UD_RERANKS.remove(model_family)
        del RERANK_ENGINES[model_family.model_name]

    def remove_ud_model_files(self, model_family: "CustomRerankModelFamilyV2"):
        from .cache_manager import RerankCacheManager

        _model_family = model_family.copy()
        for spec in model_family.model_specs:
            _model_family.model_specs = [spec]
            cache_manager = RerankCacheManager(_model_family)
            cache_manager.unregister_custom_model(self.model_type)


def get_user_defined_reranks() -> List[CustomRerankModelFamilyV2]:
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("rerank")
    return registry.get_custom_models()


def register_rerank(model_family: CustomRerankModelFamilyV2, persist: bool):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("rerank")
    registry.register(model_family, persist)


def unregister_rerank(model_name: str, raise_error: bool = True):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("rerank")
    registry.unregister(model_name, raise_error)
