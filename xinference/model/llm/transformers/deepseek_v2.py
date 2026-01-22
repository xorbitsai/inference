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
from typing import Tuple, Union

import torch

from ..llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from .core import PytorchChatModel, register_non_default_model

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model("DeepseekV2ForCausalLM")
class DeepSeekV2PytorchChatModel(PytorchChatModel):
    DEEPSEEK_V2_ARCHITECTURES = {"DeepseekV2ForCausalLM"}

    def _load_model(self, **kwargs):
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                GenerationConfig,
            )
        except ImportError:
            error_message = "Failed to import module 'transformers'"
            installation_guide = [
                "Please make sure 'transformers' is installed. ",
                "You can install it by `pip install transformers`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=kwargs["trust_remote_code"],
        )
        logger.info(f"kwargs:{kwargs}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            **kwargs,
        )
        model.generation_config = GenerationConfig.from_pretrained(self.model_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        return model, tokenizer

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ["pytorch", "fp4"]:
            return False, "DeepSeek v2 transformer only supports pytorch/fp4 format"
        if not llm_family.has_architecture(*cls.DEEPSEEK_V2_ARCHITECTURES):
            return (
                False,
                f"Model architectures {llm_family.architectures} are not DeepSeek v2",
            )
        if "chat" not in llm_family.model_ability:
            return False, "DeepSeek v2 transformer requires chat ability"
        return True
