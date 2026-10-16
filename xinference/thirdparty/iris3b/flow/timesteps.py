# Modified for Xinference: use package-relative imports.
"""Training-time timestep index samplers."""

import torch

from ..registry import TIMESTEP_SAMPLERS


@TIMESTEP_SAMPLERS.register("logit_normal")
def logit_normal(
    batch: int,
    num_timesteps: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """SD3-style logit-normal density over the index grid."""
    u = torch.sigmoid(
        torch.normal(mean=logit_mean, std=logit_std, size=(batch,), device=device, generator=generator)
    )
    return (u * num_timesteps).long().clamp(max=num_timesteps - 1)


@TIMESTEP_SAMPLERS.register("uniform")
def uniform(
    batch: int,
    num_timesteps: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    **_: float,
) -> torch.Tensor:
    return torch.randint(0, num_timesteps, (batch,), device=device, generator=generator)
