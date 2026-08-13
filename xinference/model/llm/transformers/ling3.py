from typing import Optional, Tuple, Union

from ....types import PytorchModelConfig
from ...utils import allow_trust_remote_code
from ..llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from .core import PytorchChatModel, register_non_default_model


@register_transformer
@register_non_default_model("BailingMoeV3ForCausalLM")
class Ling3PytorchChatModel(PytorchChatModel):
    """Transformers adapter for the remote-code Ling-3.0 architecture."""

    _ARCHITECTURE = "BailingMoeV3ForCausalLM"

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        config = super()._sanitize_model_config(pytorch_model_config)
        config["trust_remote_code"] = allow_trust_remote_code(self.model_family)
        config.setdefault("torch_dtype", "auto")
        return config  # type: ignore

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ("pytorch", "fp8"):
            return (
                False,
                "Ling-3.0 Transformers supports BF16, FP8, and compressed-tensors INT4 checkpoints only",
            )
        normalized_quantization = str(quantization).lower()
        if llm_spec.model_format == "pytorch" and normalized_quantization not in (
            "none",
            "int4",
        ):
            return False, "Ling-3.0 Transformers only supports none/Int4 quantization"
        if llm_spec.model_format == "fp8" and normalized_quantization != "fp8":
            return False, "Ling-3.0 FP8 checkpoints require FP8 quantization"
        if not llm_family.has_architecture(cls._ARCHITECTURE):
            return False, "Model architecture is not BailingMoeV3ForCausalLM"
        if "chat" not in llm_family.model_ability:
            return False, "Ling-3.0 Transformers requires chat ability"
        return True
