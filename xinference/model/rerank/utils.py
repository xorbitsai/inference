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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import RerankModelFamilyV2


def get_model_version(rerank_model: "RerankModelFamilyV2") -> str:
    return rerank_model.model_name


instruction_cfg = {
    "minicpm-reranker": "Query: ",
}


def preprocess_sentence(query: str, instruction: Any, model_name: str) -> str:
    if instruction and isinstance(instruction, str):
        return f"{instruction}{query}"
    if instruction is None:
        for k, v in instruction_cfg.items():
            if k.lower() in model_name.lower():
                return f"{v}{query}"
    return query
