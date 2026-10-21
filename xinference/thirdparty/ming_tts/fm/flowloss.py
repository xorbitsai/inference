import torch
import torch.nn as nn

from .dit import DiT
from .CFM import CFM


class FlowLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, z_channels, llm_cond_dim, **kwargs):
        super(FlowLoss, self).__init__()
        self.z_channels = z_channels
        self.cfm = CFM(
            model=DiT(in_channels=z_channels, llm_cond_dim=llm_cond_dim, **kwargs)
        )

    def forward(self, cond, target, latent_history, mask, patch_size):
        return self.cfm(cond=cond, target=target, latent_history=latent_history, mask=mask, patch_size=patch_size)

    def sample(self, z, latent_history, cfg=1.0, patch_size=1, sigma=0.25, temperature=0, steps=10):
        # The starting noise and every step's SDE noise are drawn up front, in
        # this order, shape and dtype, even at temperature 0: a fixed seed then
        # yields the same random stream as the reference sampler.
        noise = torch.randn((z.shape[0], patch_size, self.z_channels), device=z.device, dtype=z.dtype)
        sde_noise = torch.randn((steps, *noise.shape), device=z.device, dtype=z.dtype)
        return self.cfm.sample(
            noise=noise,
            c=z,
            latent_history=latent_history,
            sde_noise=sde_noise,
            cfg_scale=cfg,
            patch_size=patch_size,
            sigma=sigma,
            temperature=temperature,
        )
