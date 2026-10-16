# Modified for Xinference: use package-relative imports.
"""End-to-end text-to-image generation (used by inference and train-time validation)."""

from pathlib import Path

import torch

from .flow.solver import FlowDPMSolver
from .models.dit import IrisDiT
from .text.base import TextEncoder


def load_for_inference(path: str | Path) -> tuple[dict, dict[str, torch.Tensor]]:
    """``(config dict, weights)`` from a training checkpoint or an exported directory.

    A training ``.pth`` contributes its embedded config and its EMA weights,
    or its raw weights when it was trained without EMA. The checkpoint is
    memory-mapped, so its optimizer state is never read. An exported directory
    holds ``config.yaml`` and ``model.safetensors``.
    """
    path = Path(path)
    if path.is_dir():
        from omegaconf import OmegaConf
        from safetensors.torch import load_file

        raw = OmegaConf.to_container(OmegaConf.load(path / "config.yaml"))
        return raw, load_file(path / "model.safetensors")
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if "config" not in payload:
        raise ValueError(f"{path} has no embedded config; it is not a training checkpoint")
    key = "state_dict_ema" if "state_dict_ema" in payload else "state_dict"
    return payload["config"], payload[key]


@torch.no_grad()
def generate(
    model: IrisDiT,
    text_encoder: TextEncoder,
    prompts: list[str],
    height: int,
    width: int,
    steps: int = 100,
    order: int = 2,
    cfg_scale: float = 3.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    shift: float = 4.0,
    negative_prompt: str = "",
    generator: torch.Generator | None = None,
    device: torch.device | str = "cuda",
    noise: torch.Tensor | None = None,
    num_train_timesteps: int = 1000,
    prediction: str = "v",
) -> torch.Tensor:
    """Sample [B, C, H, W] images, clamped to [-1, 1].

    Integrator state stays in fp32; the network is evaluated in its own
    parameter dtype. The CFG unconditional is ``text_encoder.null`` of the
    negative prompt, which the encoder formats exactly like a positive prompt
    (an empty negative prompt is the dropout null used in training).
    """
    model_dtype = next(model.parameters()).dtype
    encoding = text_encoder.encode(prompts)
    cond = encoding.embeddings.to(device=device, dtype=model_dtype)
    cond_mask = encoding.mask.to(device=device)
    uncond, cfg_mask = None, None
    if cfg_scale != 1.0:
        null = text_encoder.null(negative_prompt)
        uncond = null.embeddings.to(device=device, dtype=model_dtype).expand(
            len(prompts), *([-1] * (null.embeddings.ndim - 1))
        )
        null_mask = null.mask.to(device=device).expand(len(prompts), -1)
        # the solver's CFG batch is cat([uncond, cond]); the mask follows suit
        cfg_mask = torch.cat([null_mask, cond_mask], dim=0)

    if noise is None:
        noise = torch.randn(
            len(prompts), model.cfg.in_channels, height, width, generator=generator, device=device
        )
    z = noise.to(device=device, dtype=torch.float32)

    def model_fn(x: torch.Tensor, t_model: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # the solver either passes `cond` itself or the doubled CFG batch
        out = model(x.to(model_dtype), t_model, y, y_mask=cond_mask if y is cond else cfg_mask)
        return out.x.float()

    solver = FlowDPMSolver(
        model_fn,
        num_timesteps=num_train_timesteps,
        cfg_scale=cfg_scale,
        cfg_interval=tuple(cfg_interval),
        prediction=prediction,
    )
    sample = solver.sample(z, cond, uncond, steps=steps, order=order, shift=shift)
    return sample.clamp(-1, 1)


@torch.no_grad()
def generate_few_step(
    model: IrisDiT,
    text_encoder: TextEncoder,
    prompts: list[str],
    height: int,
    width: int,
    sigmas: list[float],
    generator: torch.Generator | None = None,
    device: torch.device | str = "cuda",
    noise: torch.Tensor | None = None,
    num_train_timesteps: int = 1000,
    prediction: str = "v",
) -> torch.Tensor:
    """Sample a few-step distilled model (Iris-3B Turbo), clamped to [-1, 1].

    One model call per noise level in ``sigmas`` (the descending schedule from
    1 saved in the checkpoint's ``distill.yaml``) and no CFG: guidance is
    distilled into the weights. Between calls the clean prediction moves to the
    next noise level along the flow, reusing the noise implied by the current
    state, so the image is fully determined by the initial noise.
    """
    model_dtype = next(model.parameters()).dtype
    encoding = text_encoder.encode(prompts)
    cond = encoding.embeddings.to(device=device, dtype=model_dtype)
    mask = encoding.mask.to(device=device)
    if noise is None:
        noise = torch.randn(
            len(prompts), model.cfg.in_channels, height, width, generator=generator, device=device
        )
    x = noise.to(device=device, dtype=torch.float32)
    sigma = torch.full((x.shape[0], 1, 1, 1), sigmas[0], device=x.device, dtype=torch.float32)
    for i in range(len(sigmas)):
        out = model(x.to(model_dtype), sigma.flatten() * num_train_timesteps, cond, y_mask=mask).x.float()
        clean = out if prediction == "x" else x - sigma * out
        if i + 1 < len(sigmas):
            implied_noise = (x - (1 - sigma) * clean) / sigma
            sigma = torch.full_like(sigma, sigmas[i + 1])
            x = (1 - sigma) * clean + sigma * implied_noise
    return clean.clamp(-1, 1)
