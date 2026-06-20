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

from typing import List, Optional, Tuple, Union

from ....types import LoRA, PytorchModelConfig
from ..llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from .core import PytorchChatModel, register_non_default_model


@register_transformer
@register_non_default_model("HunYuanDenseV1ForCausalLM", "HYV3ForCausalLM")
class HyMT2PytorchModel(PytorchChatModel):
    """Adapter for Tencent Hunyuan Hy-MT2 multilingual translation models.

    Hy-MT2 ships two architectures:
      - HunYuanDenseV1ForCausalLM (1.8B / 7B dense)
      - HYV3ForCausalLM           (30B-A3B MoE)

    Both expose a chat-style interface (the model is "instructed" via a
    translation prompt) and require the bfloat16 dtype recommended by the
    upstream README; float16 on Apple MPS triggers an MPSFloatType embedding
    error inside the custom modeling code, so we pin bfloat16 here.
    """

    _ARCHITECTURES = {"HunYuanDenseV1ForCausalLM", "HYV3ForCausalLM"}

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

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        config = super()._sanitize_model_config(pytorch_model_config)
        # Hy-MT2 official README pins bfloat16; the default MPS preferred
        # dtype (float16) breaks the custom HunYuanDense embedding path.
        config.setdefault("torch_dtype", "bfloat16")
        config.setdefault("trust_remote_code", True)
        return config  # type: ignore

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ["pytorch"]:
            return False, "Hy-MT2 transformer only supports pytorch format"
        if not llm_family.has_architecture(*cls._ARCHITECTURES):
            return (
                False,
                "Hy-MT2 transformer only supports architectures: "
                f"{', '.join(sorted(cls._ARCHITECTURES))}",
            )
        return True
