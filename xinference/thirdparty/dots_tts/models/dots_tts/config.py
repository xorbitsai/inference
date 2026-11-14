from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from ...config.base import ConfigBase, StrictConfigBase
from ...modules.vocoder.config import AudioVAEConfig


class _EncoderConfig(ConfigBase):
    num_layers: int = 6
    num_heads: int = 16
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    modulation: bool = False
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_dropout: float = 0.0
    dropout: float = 0.0
    norm_layer: str = "LayerNorm"
    alibi_bias: bool = False
    rotary_bias: bool = False
    rotary_theta: float | None = 10000
    input_dim: int = 1024
    causal: bool = True


class _DiTConfig(ConfigBase):
    num_layers: int = 18
    num_heads: int = 16
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    modulation: bool = True
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_dropout: float = 0.0
    dropout: float = 0.0
    norm_layer: str = "LayerNorm"
    alibi_bias: bool = False
    rotary_bias: bool = True
    rotary_theta: float | None = 10000


class LossConfig(StrictConfigBase):
    ce_weight: float = 1.0
    fm_weight: float = 1.0
    eos_weight: float = 1.0


class MeanFlowConfig(ConfigBase):
    enabled: bool = False
    use_duration_embedding: bool = True


class SamplingConfig(StrictConfigBase):
    solver: Literal["flow_matching", "scm"]
    ode_method: Literal["euler"] = "euler"
    num_steps: Literal[1, 2] = 2
    guidance_scale: Literal[0.0] = 0.0
    tau_mid: float = Field(default=1.3, gt=0.0, lt=math.pi / 2)

    @model_validator(mode="after")
    def _validate_solver_contract(self) -> "SamplingConfig":
        expected_num_steps = 1 if self.solver == "flow_matching" else 2
        if self.num_steps != expected_num_steps:
            raise ValueError(
                f"{self.solver} artifact requires num_steps={expected_num_steps}; "
                f"got num_steps={self.num_steps}."
            )
        return self

    def resolve(
        self,
        *,
        ode_method: str | None,
        num_steps: int | None,
        guidance_scale: float | None,
    ) -> tuple[str, int, float]:
        expected = (self.ode_method, self.num_steps, self.guidance_scale)
        resolved = (
            self.ode_method if ode_method is None else str(ode_method),
            self.num_steps if num_steps is None else int(num_steps),
            self.guidance_scale if guidance_scale is None else float(guidance_scale),
        )
        if resolved != expected:
            raise ValueError(
                f"{self.solver} artifact requires "
                f"ode_method={expected[0]!r}, num_steps={expected[1]}, "
                f"guidance_scale={expected[2]}; got "
                f"ode_method={resolved[0]!r}, num_steps={resolved[1]}, "
                f"guidance_scale={resolved[2]}."
            )
        return resolved


class StreamingConfig(StrictConfigBase):
    interleave_mode: Literal["one_to_one", "buffered_ratio"] = "one_to_one"
    initial_lookahead: int | None = None
    ta_per_tta: int = 2
    warmup_ta: int = 1

    @model_validator(mode="after")
    def _validate_streaming_contract(self) -> "StreamingConfig":
        if self.interleave_mode == "one_to_one":
            return self
        if self.initial_lookahead is not None and self.initial_lookahead <= 0:
            raise ValueError("initial_lookahead must be positive.")
        if self.ta_per_tta <= 0:
            raise ValueError("ta_per_tta must be positive for buffered_ratio.")
        if self.warmup_ta < 0:
            raise ValueError("warmup_ta must be non-negative.")
        return self


class ModelConfig(ConfigBase):
    model_type: str = "dots_tts"
    latent_dim: int
    patch_size: int
    cfg_droprate: float = 0.2
    PatchEncoder: _EncoderConfig
    DiT: _DiTConfig
    vocoder: AudioVAEConfig
    fm_sigma: float = 0.0
    xvec_drop_rate: float = 0.2
    campplus_embedding_size: int | None = 512
    xvec_max_audio_seconds: float = 10.0
    meanflow: MeanFlowConfig | None = None
    sampling: SamplingConfig | None = None
    streaming: StreamingConfig | None = None


__all__ = [
    "LossConfig",
    "MeanFlowConfig",
    "ModelConfig",
    "SamplingConfig",
    "StreamingConfig",
]
