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
from typing import List, Literal, Optional

from ..custom import ModelRegistry
from .core import RerankModelFamilyV2

logger = logging.getLogger(__name__)


class CustomRerankModelFamilyV2(RerankModelFamilyV2):
    version: Literal[2] = 2
    model_id: Optional[str]  # type: ignore
    model_revision: Optional[str]  # type: ignore
    model_uri: Optional[str]
    model_type: Literal["rerank"] = "rerank"  # for frontend


UD_RERANKS: List[CustomRerankModelFamilyV2] = []


class RerankModelRegistry(ModelRegistry):
    model_type = "rerank"

    def __init__(self):
        from . import BUILTIN_RERANK_MODELS

        super().__init__()
        self.models = UD_RERANKS
        self.builtin_models = list(BUILTIN_RERANK_MODELS.keys())


def get_user_defined_reranks() -> List[CustomRerankModelFamilyV2]:
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("rerank")
    return registry.get_custom_models()


def register_rerank(model_spec: CustomRerankModelFamilyV2, persist: bool):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("rerank")
    registry.register(model_spec, persist)


def unregister_rerank(model_name: str, raise_error: bool = True):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("rerank")
    registry.unregister(model_name, raise_error)
