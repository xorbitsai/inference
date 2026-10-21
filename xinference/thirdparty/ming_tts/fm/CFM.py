import torch
from torch import nn
import torch.nn.functional as F


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)


class CFM(nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        cond,
        target,
        latent_history,
        mask,
        patch_size,
    ):
        x1 = target
        batch, dtype = x1.shape[0], x1.dtype
        x0 = torch.randn_like(x1)
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        x = (1 - t) * x0 + t * x1
        flow = x1 - x0

        pred = self.model(x=x, t=time, c=cond, latent_history=latent_history, mask=mask.to(torch.bool))
        pred = pred[:, -patch_size:, :]

        loss = F.mse_loss(pred, flow, reduction="none")
        mask = (mask == 1)
        loss = loss[mask]

        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        noise,
        c,
        latent_history,
        sde_noise,
        cfg_scale=2.0,
        sway_sampling_coef=-1.0,
        use_epss=True,
        patch_size=1,
        sigma=0.25,
        temperature=0.0,
    ):
        """Integrate the flow from ``noise`` to a latent patch.

        ``noise`` is ``(B, patch_size, latent_dim)`` and ``sde_noise`` holds
        one tensor of that shape per step, so the number of steps is
        ``len(sde_noise)``. The stochastic term is scaled by
        ``sigma * sqrt(temperature)`` and vanishes at temperature 0.
        """
        steps = sde_noise.shape[0]

        def fn(t, x):
            pred_cfg = self.model.forward_with_cfg(
                x=x,
                t=t,
                c=c,
                latent_history=latent_history,
                patch_size=patch_size,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_scale

        if use_epss:  # Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=noise.device, dtype=noise.dtype)
        else:
            t = torch.linspace(0, 1, steps + 1, device=noise.device, dtype=noise.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        y = noise
        for step in range(steps):
            dt = t[step + 1] - t[step]
            y = y + fn(t[step], y) * dt
            y = y + sigma * (temperature ** 0.5) * (dt.abs() ** 0.5) * sde_noise[step]
        return y
