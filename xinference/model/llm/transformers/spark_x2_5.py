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

from typing import Dict, List, Optional, Tuple, Union

from ....types import PytorchModelConfig
from ...utils import allow_trust_remote_code
from ..core import chat_context_var
from ..llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from .core import PytorchChatModel, PytorchModel, register_non_default_model


class _SparkX25PytorchMixin:
    _ARCHITECTURE = "Spark2_5ForCausalLM"

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        config = super()._sanitize_model_config(pytorch_model_config)
        config["trust_remote_code"] = allow_trust_remote_code(self.model_family)
        config.setdefault("torch_dtype", "auto")
        return config  # type: ignore

    @classmethod
    def _match_spark_x2_5(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ("pytorch", "fp8"):
            return (
                False,
                "Spark-X2.5 Transformers supports pytorch and fp8 formats only",
            )
        normalized_quantization = (quantization or "none").lower()
        if llm_spec.model_format == "pytorch" and normalized_quantization not in (
            "none",
            "int8",
        ):
            return (
                False,
                "Spark-X2.5 pytorch format only supports none/Int8 quantization",
            )
        if llm_spec.model_format == "fp8" and normalized_quantization != "fp8":
            return False, "Spark-X2.5 fp8 format requires FP8 quantization"
        if not llm_family.has_architecture(cls._ARCHITECTURE):
            return False, "Model architecture is not Spark2_5ForCausalLM"
        return True


@register_transformer
@register_non_default_model("Spark2_5ForCausalLM")
class SparkX25PytorchModel(_SparkX25PytorchMixin, PytorchModel):
    """Transformers adapter for Spark-X2.5 base checkpoints."""

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        result = cls._match_spark_x2_5(llm_family, llm_spec, quantization)
        if result is not True:
            return result
        if "generate" not in llm_family.model_ability:
            return False, "Spark-X2.5 base Transformers requires generate ability"
        return True


@register_transformer
class SparkX25PytorchChatModel(_SparkX25PytorchMixin, PytorchChatModel):
    """Transformers adapter for Spark-X2.5 instruction checkpoints."""

    def _get_full_prompt(self, messages: List[Dict], tools, generate_config: dict):
        chat_template_kwargs = (
            self._get_chat_template_kwargs_from_generate_config(
                generate_config, self.reasoning_parser
            )
            or {}
        )
        chat_context_var.set(chat_template_kwargs)
        full_context_kwargs = chat_template_kwargs.copy()
        if tools:
            full_context_kwargs["tools"] = tools
        return self.get_full_context(
            messages,
            None,
            tokenizer=self._tokenizer,
            **full_context_kwargs,
        )

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        result = cls._match_spark_x2_5(llm_family, llm_spec, quantization)
        if result is not True:
            return result
        if "chat" not in llm_family.model_ability:
            return False, "Spark-X2.5 Transformers requires chat ability"
        return True
