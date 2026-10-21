import torch
import torch.nn as nn

from .dit import DiT
from .CFM import CFM
from .mlp import SimpleMLPAdaLN


class FlowLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, z_channels, llm_cond_dim, flow_model_type=None, **kwargs):
        super(FlowLoss, self).__init__()

        self.z_channels = z_channels
        if not flow_model_type:
            self.cfm = CFM(
                model=DiT(in_channels=z_channels, llm_cond_dim=llm_cond_dim, **kwargs)
            )
        elif flow_model_type == "vmlp":
            self.cfm = CFM(
                model=SimpleMLPAdaLN(**kwargs)
            )



    def forward(self, target, z, mask=None, latent_history=None, patch_size=1):
        loss = self.cfm(cond=z, target=target, latent_history=latent_history, mask=mask, patch_size=patch_size)
        return loss

    def sample(
        self,
        z,
        latent_history,
        cfg=1.0,
        patch_size=1,
        sigma=0.25,
        temperature=0.0,
        steps=10,
        generator=None,
        noise_dtype=None,
        initial_noise_on_cpu=False,
    ):
        # diffusion loss sampling
        if noise_dtype is None:
            noise_dtype = z.dtype
        noise_device = torch.device("cpu") if initial_noise_on_cpu else z.device
        noise = torch.randn(
            z.shape[0],
            self.z_channels,
            patch_size,
            device=noise_device,
            dtype=noise_dtype,
            generator=generator,
        )
        if initial_noise_on_cpu:
            # The native reference uses ``torch.randn(...).cuda()``.  The random
            # values therefore come from the CPU RNG and are copied to CUDA;
            # generating them directly on CUDA changes both the values and
            # the CUDA RNG state for every later Flow token.
            noise = noise.to(device=z.device)

        sampled_token_latent = self.cfm.sample(
            noise=noise,
            c=z,
            latent_history=latent_history,
            cfg_scale=cfg,
            patch_size=patch_size,
            sigma=sigma,
            temperature=temperature,
            steps=steps,
            generator=generator,
        )

        return sampled_token_latent
