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

from typing import Optional

from ..llm_family import LLMFamilyV1, LLMSpecV1
from .core import PytorchChatModel, PytorchModel, PytorchModelConfig


class LlamaPytorchModel(PytorchModel):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_spec,
            quantization,
            model_path,
            pytorch_model_config=pytorch_model_config,
        )

    def _load_model(self, kwargs: dict):
        model, tokenizer = super()._load_model(kwargs)
        # Llama has no pad token by default
        # https://github.com/huggingface/transformers/blob/07998ef39926b76d3f6667025535d0859eed61c3/docs/source/en/llm_tutorial.md?plain=1#L125
        tokenizer.pad_token = tokenizer.eos_token
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    @classmethod
    def match(cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1") -> bool:
        if llm_spec.model_format != "pytorch":
            return False
        if "llama-2" not in llm_family.model_name:
            return False
        if "generate" not in llm_family.model_ability:
            return False
        return True


class LlamaPytorchChatModel(PytorchChatModel):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        pytorch_model_config: Optional["PytorchModelConfig"] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_spec,
            quantization,
            model_path,
            pytorch_model_config=pytorch_model_config,
        )
        self._use_fast_tokenizer = False

    def _load_model(self, kwargs: dict):
        model, tokenizer = super()._load_model(kwargs)
        # Llama has no pad token by default
        # https://github.com/huggingface/transformers/blob/07998ef39926b76d3f6667025535d0859eed61c3/docs/source/en/llm_tutorial.md?plain=1#L125
        tokenizer.pad_token = tokenizer.eos_token
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    @classmethod
    def match(cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1") -> bool:
        if llm_spec.model_format != "pytorch":
            return False
        if "llama-2" not in llm_family.model_name:
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True
