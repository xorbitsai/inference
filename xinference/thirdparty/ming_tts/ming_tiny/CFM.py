"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

class Solver:
    def __init__(
        self,
        func,
        y0,
        sigma=0.25,
        temperature=1.5,
        generator=None,
    ) -> None:
        self.func = func
        self.y0 = y0
        self.sigma = sigma
        self.temperature = temperature
        self.generator = generator


    def integrate(self, t):
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(t[:-1], t[1:]):
            dt = t1 - t0
            f0 = self.func(t0, y0)
            dy = dt * f0
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                j += 1



            noise = torch.randn(
                y0.shape,
                device=y0.device,
                dtype=y0.dtype,
                generator=self.generator,
            )
            shift = self.sigma * (self.temperature ** 0.5) * (abs(dt) ** 0.5) * noise
            y0 = y1 + shift

        return solution


    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


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
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
    ):
        super().__init__()

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.model = model

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs


    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        noise,
        c,
        latent_history,
        steps=10,
        cfg_scale=1.0,
        sway_sampling_coef=-1.0,
        seed: int | None = None,
        use_epss=True,
        patch_size=1,
        sigma=0.25,
        temperature=0.0,
        generator=None,
    ):
        # raw wave


        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # The custom Euler solver supplies a scalar timestep.  The
            # batched CFG path expands it inside DiT.forward_with_cfg(), but
            # the direct conditional/unconditional paths call DiT.forward()
            # and therefore must expand it here.
            direct_t = t
            if direct_t.ndim == 0:
                direct_t = direct_t.repeat(x.shape[0])
            if direct_t.ndim != 1 or direct_t.shape[0] != x.shape[0]:
                raise ValueError(
                    "CFM timestep must be scalar or have one value per "
                    f"sample, got {tuple(direct_t.shape)} for batch "
                    f"{x.shape[0]}"
                )

            # Pure unconditional branch.
            if cfg_scale < 1e-5:
                pred = self.model(
                    x=x,
                    t=direct_t,
                    c=torch.zeros_like(c),
                    latent_history=latent_history,
                )
                return pred[:, -patch_size:, :]

            # At exactly 1.0 no second DiT pass is necessary.
            if cfg_scale == 1.0:
                pred = self.model(
                    x=x,
                    t=direct_t,
                    c=c,
                    latent_history=latent_history,
                )
                return pred[:, -patch_size:, :]

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.model.forward_with_cfg(
                x=x,
                t=t,
                c=c,
                latent_history=latent_history,
                cfg_scale=cfg_scale,
                patch_size=patch_size,


            )
            pred_cond, pred_uncond = torch.chunk(pred_cfg, 2, dim=0)
            # Match the unconditional (0) and conditional (1) branches above.
            return pred_uncond + cfg_scale * (pred_cond - pred_uncond)


        y0 = noise.transpose(1, 2)

        t_start = 0


        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=noise.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=noise.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        solver = Solver(
            fn,
            y0,
            sigma=sigma,
            temperature=temperature,
            generator=generator,
        )
        trajectory = solver.integrate(t)
        # trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        sampled = trajectory[-1]
        out = sampled

        return out, trajectory

    def forward(
        self,
        cond,
        target,
        latent_history,
        mask,
        patch_size,
    ):

        # mel is x1
        x1 = target

        batch, dtype = x1.shape[0], x1.dtype

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # time = compute_density_for_timestep_sampling(
        #     weighting_scheme="logit_normal",
        #     batch_size=batch,
        #     logit_mean=0,
        #     logit_std=1,
        #     mode_scale=None,
        # ).to(dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1) # only for sample
        x = (1 - t) * x0 + t * x1
        flow = x1 - x0

        pred = self.model(x=x, t=time, c=cond, latent_history=latent_history, mask=mask.to(torch.bool))
        pred = pred[:, -patch_size:, :]

        loss = F.mse_loss(pred, flow, reduction="none") #[batch, seq_len ,Channel]
        mask = (mask == 1)
        loss = loss[mask] #rand_span_mask , [batch, seq_len]

        if loss.numel() == 0:
            import logging
            logging.warning(f"[CFM] WARNING: loss mask is empty! mask shape={mask.shape}, mask sum={mask.sum().item()}. Returning zero loss.")
            return torch.tensor(0.0, dtype=dtype, device=x1.device, requires_grad=True)

        return loss.mean()
