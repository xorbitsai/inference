# Copyright 2022-2025 XProbe Inc.
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

from ..llm_family import LLMFamilyV1, LLMSpecV1, register_transformer
from .core import PytorchChatModel, register_non_default_model

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model("gemma-3-1b-it")
class Gemma3TextChatModel(PytorchChatModel):
    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if model_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        llm_family = model_family.model_family or model_family.model_name
        if "gemma-3-1b-it".lower() in llm_family.lower():
            return True
        return False
