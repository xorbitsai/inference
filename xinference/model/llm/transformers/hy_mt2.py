# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from typing import Any, Dict, List, Optional, Tuple, Union

from packaging.version import InvalidVersion, Version

from ....types import LoRA, PytorchModelConfig
from ...utils import allow_trust_remote_code, virtual_env_allows_missing_engine
from ..llm_family import (
    HY_MT3_MODEL_NAMES,
    LLMFamilyV2,
    LLMSpecV1,
    register_transformer,
)
from .core import PytorchChatModel, register_non_default_model


@register_transformer
@register_non_default_model("HunYuanDenseV1ForCausalLM", "HYV3ForCausalLM")
class HyMT2PytorchModel(PytorchChatModel):
    """Adapter for Tencent Hunyuan Hy-MT2 and Hy-MT3 translation models.

    Both series share two architectures:
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
        config["trust_remote_code"] = allow_trust_remote_code(self.model_family)
        return config  # type: ignore

    def apply_quantization_config(
        self, kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if (
            self._device == "mps"
            and self.model_family.model_name in HY_MT3_MODEL_NAMES
            and self.model_spec.model_format == "fp8"
            and self.model_spec.quantization_method == "compressed-tensors"
        ):
            raise ValueError(
                f"{self.model_family.model_name} FP8 is unsupported on MPS with "
                "Transformers. Set device=cpu or use CUDA with device=cuda. "
                "For Apple GPU inference, use the BF16 checkpoint with Transformers "
                "or the GGUF checkpoint with llama.cpp."
            )
        return super().apply_quantization_config(kwargs)

    @classmethod
    def validate_launch(cls, model_family: LLMFamilyV2) -> None:
        if (
            model_family.model_name != "Hy-MT3-30B-A3B"
            or model_family.model_specs[0].model_format != "pytorch"
            or virtual_env_allows_missing_engine()
        ):
            return

        minimum_version = Version("5.6.0")
        installed = None
        try:
            installed = distribution_version("transformers")
            if Version(installed) >= minimum_version:
                return
        except (PackageNotFoundError, InvalidVersion):
            pass
        raise ValueError(
            f"{model_family.model_name} BF16 requires Transformers >= {minimum_version}. "
            f"Found {installed or 'not installed'}. Upgrade Transformers or enable "
            "the model virtual environment."
        )

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if not llm_family.has_architecture(*cls._ARCHITECTURES):
            return (
                False,
                "Hy-MT Transformers supports architectures: "
                f"{', '.join(sorted(cls._ARCHITECTURES))}",
            )
        if llm_spec.model_format == "pytorch":
            return True
        if (
            llm_family.model_name in HY_MT3_MODEL_NAMES
            and llm_spec.model_format == "fp8"
        ):
            if llm_spec.quantization_method == "compressed-tensors":
                return True
            return False, "Hy-MT3-30B-A3B FP8 requires vLLM or SGLang"
        return False, "Hy-MT Transformers requires a supported PyTorch checkpoint"
