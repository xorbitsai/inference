# Modified for Xinference: use package-relative imports.
"""Rectified-flow training objective."""

from dataclasses import dataclass

import torch

from . import timesteps  # noqa: F401  (registers samplers)
from ..config import FlowConfig
from .schedule import FlowSchedule, resolution_shift
from ..registry import TIMESTEP_SAMPLERS


@dataclass
class FlowLossOutput:
    loss: torch.Tensor  # MSE flow loss: scalar, or [B] when reduction="none"
    features: dict[int, torch.Tensor]  # patch tokens captured by the model
    timestep_idx: torch.Tensor
    repa_loss: torch.Tensor | None = None  # set when the model computes it in-forward


class RectifiedFlow:
    """Velocity-matching loss on the shifted discrete schedule.

    prediction="v": the network outputs velocity ``v = eps - x0`` directly.
    prediction="x": the network outputs the clean image; the loss is still
    computed in velocity space via ``v_hat = (x_t - x0_hat) / max(sigma, s_min)``.
    """

    def __init__(self, cfg: FlowConfig, tokens: int | None = None):
        """``tokens``: the stage's image token count, used only by ``shift_law``."""
        self.cfg = cfg
        self.shift = cfg.shift
        if cfg.shift_law != "none":
            if tokens is None:
                raise ValueError("flow.shift_law requires the stage token count")
            self.shift = resolution_shift(tokens, cfg.shift_law, cfg.shift, cfg.shift_base_tokens)
        self.schedule = FlowSchedule(cfg.num_train_timesteps, self.shift)
        self._sampler = TIMESTEP_SAMPLERS.get(cfg.timestep_sampler)

    def sample_timesteps(self, batch: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return self._sampler(
            batch,
            self.cfg.num_train_timesteps,
            logit_mean=self.cfg.logit_mean,
            logit_std=self.cfg.logit_std,
            device=device,
        )

    def training_loss(
        self,
        model,
        x0: torch.Tensor,
        y: torch.Tensor,
        timestep_idx: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        model_kwargs: dict | None = None,
        reduction: str = "mean",
    ) -> FlowLossOutput:
        if timestep_idx is None:
            timestep_idx = self.sample_timesteps(x0.shape[0], device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t, t_model = self.schedule.add_noise(x0, noise, timestep_idx)
        out = model(x_t, t_model, y, **(model_kwargs or {}))

        target = noise - x0
        if self.cfg.prediction == "v":
            pred = out.x
        elif self.cfg.prediction == "x":
            sigma = self.schedule.sigma_at(timestep_idx).view(-1, *([1] * (x0.ndim - 1)))
            sigma = sigma.to(x0.dtype)
            pred = (x_t - out.x) / sigma.clamp(min=self.cfg.x_pred_sigma_min)
        else:
            raise ValueError(f"unknown prediction type '{self.cfg.prediction}'")

        per_sample = (pred.float() - target.float()).pow(2).mean(dim=list(range(1, x0.ndim)))
        loss = per_sample.mean() if reduction == "mean" else per_sample
        return FlowLossOutput(
            loss=loss,
            features=out.features,
            timestep_idx=timestep_idx,
            repa_loss=getattr(out, "repa_loss", None),
        )
