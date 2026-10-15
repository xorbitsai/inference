"""Rotary position embeddings.

Image tokens use axial 2D RoPE whose coordinates are normalized to a fixed
span ``[0, scale]`` regardless of grid size, so a 32x32 (512px) and a 64x64
(1024px) patch grid cover the same angular range - this is what lets one
set of weights train at 512px and fine-tune at 1024px. Text tokens use plain
1D RoPE over raw integer positions.

Two opt-in surfaces exist for decisions that must be made before a long
pretraining run, because RoPE geometry is baked into the learned weights:
``aspect`` (whether the encoding preserves aspect ratio at all) and
``frame_pairs`` (whether a third, frame axis is reserved for later
multi-image conditioning).
"""

import torch


def rope_2d(
    head_dim: int,
    height: int,
    width: int,
    theta: float = 10000.0,
    scale: float = 16.0,
    aspect: str = "square",
    frame_pairs: int = 0,
    frame_theta: float = 10.0,
    frame_index: int = 0,
) -> torch.Tensor:
    """Complex rotation factors for a row-major ``height x width`` token grid.

    Returns ``[height*width, head_dim//2]`` complex64. Frequencies come in
    ``head_dim//4`` pairs interleaved as (x, y) along the last axis.

    ``aspect`` selects the coordinate law:

    - ``"square"`` (default): each axis is normalized to ``[0, scale]``
      independently, so a 96x40 grid and a 40x96 grid are positionally
      identical and the per-token angular step differs between the axes.
    - ``"isotropic"``: one step ``scale / (max(height, width) - 1)`` shared by
      both axes. The longer axis spans ``[0, scale]``, the shorter one spans
      proportionally less, so aspect ratio survives and the step is the same
      in both directions. Positions still never leave ``[0, scale]``.

    ``frame_pairs`` hands the N *slowest* (x, y) frequency pairs to a third,
    frame axis carrying ``frame_index`` at its own ``frame_theta``. The tail is
    the right donor: at ``theta=1e4`` over a span of 16 those pairs rotate by
    <1e-2 rad across the entire image, so they carry no usable position today.
    ``frame_index=0`` rotates them by exactly zero, and a frame index shared by
    every token cancels in ``q . k``, so single-image training is unaffected by
    reserving them. That is what lets a model pretrained wholly at
    ``frame_index=0`` be finetuned later with a second image at
    ``frame_index=1`` without disturbing its learned single-image geometry.
    """
    n_pairs = head_dim // 4
    if not 0 <= frame_pairs < n_pairs:
        raise ValueError(f"frame_pairs must be in [0, {n_pairs}), got {frame_pairs}")
    xy_pairs = n_pairs - frame_pairs
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 4)[:n_pairs].float() / head_dim)
    freqs = freqs[:xy_pairs]
    if aspect == "square":
        x_pos = torch.linspace(0, scale, width)
        y_pos = torch.linspace(0, scale, height)
    elif aspect == "isotropic":
        step = scale / max(max(height, width) - 1, 1)
        x_pos = torch.arange(width).float() * step
        y_pos = torch.arange(height).float() * step
    else:
        raise ValueError(f"unknown rope aspect mode '{aspect}'")
    x_ang = torch.outer(x_pos, freqs)  # [W, xy_pairs]
    y_ang = torch.outer(y_pos, freqs)  # [H, xy_pairs]
    x_cis = torch.polar(torch.ones_like(x_ang), x_ang)
    y_cis = torch.polar(torch.ones_like(y_ang), y_ang)
    x_grid = x_cis[None, :, :].expand(height, width, -1)
    y_grid = y_cis[:, None, :].expand(height, width, -1)
    cis = torch.stack([x_grid, y_grid], dim=-1).reshape(height * width, 2 * xy_pairs)
    if frame_pairs:
        n_slots = 2 * frame_pairs
        f_freqs = 1.0 / frame_theta ** (torch.arange(n_slots).float() / n_slots)
        f_cis = torch.polar(torch.ones(n_slots), float(frame_index) * f_freqs)
        cis = torch.cat([cis, f_cis[None, :].expand(height * width, -1)], dim=-1)
    return cis


def rope_1d(head_dim: int, length: int, theta: float = 10000.0) -> torch.Tensor:
    """Standard 1D RoPE factors over integer positions: ``[length, head_dim//2]`` complex64."""
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim)
    ang = torch.outer(torch.arange(length).float(), freqs)
    return torch.polar(torch.ones_like(ang), ang)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``q``/``k`` of shape ``[B, N, H, head_dim]``; complex math runs in fp32."""
    q_c = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_c = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    fc = freqs_cis[None, :, None, :]  # broadcast over batch and heads
    q_out = torch.view_as_real(q_c * fc).flatten(3)
    k_out = torch.view_as_real(k_c * fc).flatten(3)
    return q_out.to(q.dtype), k_out.to(k.dtype)
