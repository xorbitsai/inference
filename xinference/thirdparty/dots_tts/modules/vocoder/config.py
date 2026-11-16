from __future__ import annotations

from pydantic import Field

from ...config.base import ConfigBase


class AudioVAEConfig(ConfigBase):
    sample_rate: int = 24000
    upsample_rates: list[int] = Field(default_factory=list)
    upsample_kernel_sizes: list[int] = Field(default_factory=list)
    upsample_initial_channel: int = 1536
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = Field(default_factory=list)
    resblock_dilation_sizes: list[list[int]] = Field(default_factory=list)
    downsample_rates: list[int] = Field(default_factory=list)
    downsample_channels: list[int] = Field(default_factory=list)
    activation: str = "snakebeta"
    snake_logscale: bool = True
    latent_dim: int = 128
    causal: bool = False
    mi_num_layers: int = 4
    causal_encoder: bool = False
    use_bias_at_final: bool = True
    use_tanh_at_final: bool = True


__all__ = ["AudioVAEConfig"]
