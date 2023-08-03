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
from .core import PytorchChatModel, PytorchModelConfig


class ChatglmPytorchChatModel(PytorchChatModel):
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
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            error_message = "Failed to import module 'transformers'"
            installation_guide = [
                "Please make sure 'transformers' is installed. ",
                "You can install it by `pip install transformers`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            revision=kwargs["revision"],
        )
        model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            **kwargs,
        )
        return model, tokenizer

    @classmethod
    def match(cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1") -> bool:
        if llm_spec.model_format != "pytorch":
            return False
        if "chatglm" not in llm_family.model_name:
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True
