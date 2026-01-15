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
from builtins import classmethod
from typing import List, Optional, Tuple, Union

from ....types import LoRA
from ...scheduler.request import InferenceRequest
from ..llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from .core import PytorchModel, PytorchModelConfig, register_non_default_model


@register_transformer
@register_non_default_model("OPTForCausalLM")
class OptPytorchModel(PytorchModel):
    OPT_ARCHITECTURES = {"OPTForCausalLM"}

    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV2",
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_path,
            pytorch_model_config=pytorch_model_config,
            peft_model=peft_model,
        )

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ["pytorch", "fp4"]:
            return False, "OPT transformer only supports pytorch/fp4 format"
        if not llm_family.has_architecture(*cls.OPT_ARCHITECTURES):
            return (
                False,
                f"Model architectures {llm_family.architectures} are not OPT",
            )
        return True

    def build_prefill_position_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Mainly for UT.
        Transformers code in `main` branch supports `position_ids` parameter (https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L1076),
        while in release branch, it doesn't (https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/opt/modeling_opt.py#L886).
        """
        return None

    def build_decode_position_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        return None
