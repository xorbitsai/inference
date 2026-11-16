"""Inference-only adapters for the DiT velocity field predictor.

This module is strictly for inference: no training paths belong here. Anything
that depends on flex-attention block masks built from packed lengths, packed
pack_indices, dropout, or gradient-friendly variants lives in ``dit.py`` and
the training-side ``core.py``.

Layering (top → bottom):

    DiTSolver               — streaming decoder entry point
      ├── EagerDiTRunner    — reference path (optimize=False), no KV cache
      └── CachedDiTRunner   — compiled path (optimize=True), streaming KV cache
            ├── FirstPatchStep  — decode the first patch of a generation
            ├── NextPatchStep   — decode every subsequent patch, reuse/append cache
            └── KvPrefillStep   — warm the KV cache from a prefix in one shot

    DiTKvCache              — the KV cache tensor bundle
    DiTSolverState          — per-generation mutable state (cache, mods, schedule)

    FusedAdaLNDiT           — DiT rewired: fused adaLN + fused QKV
      └── FusedQKVMultiHeadAttention

    Free helpers: ``apply_cfg``, ``concat_branches``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
from loguru import logger
from torchdiffeq import odeint

from ...modules.backbone.dit import modulate
from ...modules.backbone.inference_utils import (
    build_causal_update_mask,
    build_fused_qkv_projection,
    build_rotary_cos_sin,
    compile_module_forward,
    project_attention,
)
from ...modules.backbone.layers import _compiled_create_block_mask

# --------------------------------------------------------------------------- #
# Section 1. Free math helpers shared across every inference path.
# --------------------------------------------------------------------------- #


def apply_cfg(
    vt: torch.Tensor,
    *,
    batch_size: int,
    guidance: torch.Tensor,
) -> torch.Tensor:
    """Combine a ``[cond; uncond]`` stacked batch into a CFG-guided prediction."""
    cond = vt[:batch_size]
    uncond = vt[batch_size:]
    g = guidance.to(device=cond.device, dtype=cond.dtype)
    return cond + g * (cond - uncond)


def concat_branches(
    cond_parts: Sequence[torch.Tensor],
    uncond_parts: Sequence[torch.Tensor] | None,
    *,
    dim: int = 1,
) -> torch.Tensor:
    """Concat ``cond_parts`` along ``dim``; if an uncond branch is given, stack it along batch."""
    cond = torch.cat(list(cond_parts), dim=dim)
    if uncond_parts is None:
        return cond
    return torch.cat([cond, torch.cat(list(uncond_parts), dim=dim)], dim=0)


def _maybe_compile(module: nn.Module, *, enabled: bool) -> Callable[..., Any]:
    return compile_module_forward(module) if enabled else module


# --------------------------------------------------------------------------- #
# Section 2. Fused DiT adapter (inference-only rewrite of DiT).
# --------------------------------------------------------------------------- #


class FusedQKVMultiHeadAttention(nn.Module):
    """``MultiHeadAttention`` with a fused Q/K/V projection.

    The base attention keeps three separate projections for training; here we
    concatenate their weights so inference only pays one Linear kernel per
    layer. Self-attention only.
    """

    _INHERITED_ATTRS = (
        "num_heads",
        "head_dim",
        "rotary_bias",
        "attn_backend",
        "q_norm",
        "k_norm",
        "attn_drop",
        "o_proj",
        "o_dropout",
    )

    def __init__(self, old_attn: nn.Module):
        super().__init__()
        for name in self._INHERITED_ATTRS:
            setattr(self, name, getattr(old_attn, name))
        self.qkv_proj = build_fused_qkv_projection(old_attn)
        if self.rotary_bias:
            self.rotary = old_attn.rotary

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pos_ids: torch.Tensor | None = None,
        block_mask: Any | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        if (k is not None and k is not q) or (v is not None and v is not q):
            raise ValueError("FusedQKVMultiHeadAttention only supports self attention.")

        _batch, q_len, _hidden = q.shape
        rotary_cos = rotary_sin = None
        if self.rotary_bias:
            if pos_ids is None:
                pos_ids = torch.arange(q_len, device=q.device, dtype=torch.float32)
            rotary = self.rotary(pos_ids)
            rotary_cos, rotary_sin = rotary.cos(), rotary.sin()

        sdpa_mask = None
        flex_block_mask = None
        if self.attn_backend == "flex":
            if block_mask is None:
                raise ValueError("Flex attention backend requires a BlockMask.")
            flex_block_mask = block_mask
        elif mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            sdpa_mask = mask

        out, _k, _v = project_attention(
            self,
            q,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            attn_mask=sdpa_mask,
            block_mask=flex_block_mask,
        )
        return out


class FusedAdaLNDiT(nn.Module):
    """DiT rewired so all per-block adaLN linears fuse into a single Linear.

    Inference-only. Requires every ``DiTBlock`` to have been trained with
    ``modulation=True``. Training-side switches (``flex_prefix_len`` /
    ``pack_indices`` / dropout on inference) are intentionally absent — callers
    supply their own ``attn_mask``/``pos_ids`` when needed.
    """

    def __init__(self, old_dit: nn.Module):
        super().__init__()
        if any(not block.modulation for block in old_dit.blocks):
            raise ValueError("FusedAdaLNDiT requires modulation=True for every block.")
        for block in old_dit.blocks:
            block.attn = FusedQKVMultiHeadAttention(block.attn)

        self.input_layer = old_dit.input_layer
        self.time_embedder = old_dit.time_embedder
        self.duration_embedder = getattr(old_dit, "duration_embedder", None)
        self.blocks = old_dit.blocks
        self.output_layer = old_dit.output_layer

        model_dim = int(self.input_layer.out_features)
        adaln_linears = [block.adaLN_modulation[-1] for block in self.blocks]
        adaln_linears.append(self.output_layer.adaLN_modulation[-1])
        first_linear = adaln_linears[0]

        out_dim = model_dim * (6 * len(self.blocks) + 2)
        self.fused_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, out_dim, bias=first_linear.bias is not None),
        )
        self.fused_adaln.to(
            device=first_linear.weight.device,
            dtype=first_linear.weight.dtype,
        )
        with torch.no_grad():
            self.fused_adaln[-1].weight.copy_(
                torch.cat([lin.weight.detach() for lin in adaln_linears], dim=0)
            )
            if self.fused_adaln[-1].bias is not None:
                self.fused_adaln[-1].bias.copy_(
                    torch.cat(
                        [
                            lin.bias.detach()
                            for lin in adaln_linears
                            if lin.bias is not None
                        ],
                        dim=0,
                    )
                )

        for block in self.blocks:
            block.adaLN_modulation = nn.Identity()
        self.output_layer.adaLN_modulation = nn.Identity()

    # ---- Modulation split (public: reused by mods-by-ODE prep) ---------- #

    def split_mods(
        self, all_mods: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor]]:
        """Split fused adaLN output into per-block (6-tuple) mods + final (shift, scale)."""
        model_dim = int(self.input_layer.out_features)
        block_width = 6 * model_dim
        final_start = block_width * len(self.blocks)
        block_mods = all_mods[:, :final_start].split(block_width, dim=1)
        shift, scale = all_mods[:, final_start : final_start + 2 * model_dim].chunk(
            2, dim=1
        )
        return block_mods, (shift, scale)

    # ---- Block iteration primitives ------------------------------------- #

    def run_modulated_blocks(
        self,
        *,
        x: torch.Tensor,
        all_mods: torch.Tensor,
        attention: Callable[[int, nn.Module, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run all DiT blocks with a caller-provided attention operator.

        The ``attention`` callback receives ``(layer_idx, block, attn_in)`` and
        returns the attention output. Different inference paths supply
        different callbacks: KV-cache warmup collects K/V, cached inference
        reads from cache, full-compute forwards through the wrapped attention.
        """
        block_mods, final_mod = self.split_mods(all_mods)
        x = self.input_layer(x)
        for layer_idx, (block, block_mod) in enumerate(
            zip(self.blocks, block_mods, strict=True)
        ):
            shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
                block_mod.chunk(6, dim=1)
            )
            attn_in = modulate(block.norm1(x), shift_attn, scale_attn)
            x = x + gate_attn.unsqueeze(1) * attention(layer_idx, block, attn_in)
            ffn_in = modulate(block.norm2(x), shift_ffn, scale_ffn)
            x = x + gate_ffn.unsqueeze(1) * block.ffn(ffn_in)
        return x, final_mod

    def apply_final_layer(
        self,
        x: torch.Tensor,
        final_mod: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        shift, scale = final_mod
        return self.output_layer.linear(
            modulate(self.output_layer.norm(x), shift, scale)
        )

    def run_blocks(
        self,
        *,
        x: torch.Tensor,
        all_mods: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        pos_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vanilla forward: sdpa attention, one shot, no cache."""

        def attention(_layer_idx: int, block: nn.Module, attn_in: torch.Tensor):
            return block.attn(attn_in, mask=attn_mask, pos_ids=pos_ids)

        x, final_mod = self.run_modulated_blocks(
            x=x, all_mods=all_mods, attention=attention
        )
        return self.apply_final_layer(x, final_mod)

    def build_condition(
        self,
        timesteps: torch.Tensor,
        *,
        duration: torch.Tensor | None = None,
        g_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.time_embedder(timesteps)
        if self.duration_embedder is not None and duration is not None:
            c = c + self.duration_embedder(duration)
        if g_cond is not None:
            c = c + g_cond
        return c

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        duration: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        pos_ids: torch.Tensor | None = None,
        g_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        all_mods = self.fused_adaln(
            self.build_condition(timesteps, duration=duration, g_cond=g_cond)
        )
        return self.run_blocks(
            x=x, all_mods=all_mods, attn_mask=attn_mask, pos_ids=pos_ids
        )


def fuse_dit_for_inference(core: Any) -> FusedAdaLNDiT:
    """Wrap ``core.velocity_field_predictor`` in ``FusedAdaLNDiT`` (idempotent).

    Mutates ``core`` so any later access — training or inference — goes through
    the fused module.
    """
    dit = core.velocity_field_predictor
    if not isinstance(dit, FusedAdaLNDiT):
        dit = FusedAdaLNDiT(dit)
        core.velocity_field_predictor = dit
    return dit


# --------------------------------------------------------------------------- #
# Section 3. Context / cache / schedule.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DiTInferenceContext:
    """Immutable bundle the inference solver reads from.

    ``dit`` is always a ``FusedAdaLNDiT``; construct via ``from_core`` which is
    the one place fusion happens.
    """

    dit: FusedAdaLNDiT
    coordinate_proj: nn.Module
    hidden_patch_size: int
    latent_patch_size: int
    latent_dim: int
    hidden_size: int
    fm_sigma: float

    @classmethod
    def from_core(cls, core: Any) -> DiTInferenceContext:
        return cls(
            dit=fuse_dit_for_inference(core),
            coordinate_proj=core.coordinate_proj,
            hidden_patch_size=int(core.hidden_patch_size),
            latent_patch_size=int(core.latent_patch_size),
            latent_dim=int(core.latent_dim),
            hidden_size=int(core.fm_hidden_size),
            fm_sigma=float(core.config.fm_sigma),
        )

    @property
    def unit_len(self) -> int:
        return self.hidden_patch_size + self.latent_patch_size

    @property
    def noise_latent_dim(self) -> int:
        return self.latent_dim


@dataclass
class DiTKvCache:
    """KV cache for a single (capacity, nfe) bucket, shared across ODE steps."""

    capacity_patches: int
    capacity_tokens: int
    nfe: int
    cache_k: torch.Tensor
    cache_v: torch.Tensor
    valid_tokens: int = 0

    def copy_prefix_from(self, other: DiTKvCache) -> None:
        if int(other.nfe) != int(self.nfe):
            raise ValueError(
                "Cannot copy KV cache prefixes across different nfe values: "
                f"source={other.nfe} target={self.nfe}."
            )
        valid_tokens = min(int(other.valid_tokens), self.capacity_tokens)
        if valid_tokens > 0:
            for target, source in (
                (self.cache_k, other.cache_k),
                (self.cache_v, other.cache_v),
            ):
                target[..., :valid_tokens, :].copy_(source[..., :valid_tokens, :])
        self.valid_tokens = valid_tokens


MEANFLOW_MODE = "meanflow"
FLOW_MATCHING_MODE = "flow_matching"


@dataclass
class ODESchedule:
    """Per-decode time grid.

    ``mode`` mirrors :class:`dots_tts.modules.backbone.dit.DiT.mode`:

    * ``"meanflow"``: ``durations`` is set; ``times[i]`` is the segment start.
    * ``"flow_matching"``: ``durations`` is ``None``; ``times[i] = i / nfe``.
    """

    mode: str
    times: torch.Tensor
    durations: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.mode not in (MEANFLOW_MODE, FLOW_MATCHING_MODE):
            raise ValueError(
                f"ODESchedule.mode must be {MEANFLOW_MODE!r} or {FLOW_MATCHING_MODE!r}, "
                f"got {self.mode!r}."
            )
        if self.mode == MEANFLOW_MODE and self.durations is None:
            raise ValueError("MeanFlow schedule requires per-step durations.")

    def step_kwargs(self, ode_idx: int) -> dict[str, torch.Tensor]:
        if self.mode == MEANFLOW_MODE:
            return {}
        return {"t": self.times[ode_idx]}

    def advance(
        self,
        z: torch.Tensor,
        vt: torch.Tensor,
        *,
        ode_idx: int,
        batch_size: int,
        flow_dt: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.mode == MEANFLOW_MODE:
            assert self.durations is not None  # enforced in __post_init__
            dt = self.durations[ode_idx].expand(batch_size)
            return (z + dt.view(-1, 1, 1) * vt).clone()
        if flow_dt is None:
            raise RuntimeError("FlowMatching schedule step is missing flow_dt.")
        return z + flow_dt * vt


@dataclass
class DiTSolverState:
    """Per-generation mutable state carried through streaming decode calls."""

    cache: DiTKvCache | None = None
    all_mods_by_ode: torch.Tensor | None = None
    schedule: ODESchedule | None = None


# --------------------------------------------------------------------------- #
# Section 4. Step modules (compiled hot loops).
# --------------------------------------------------------------------------- #


class FirstPatchStep(nn.Module):
    """First patch of a decode: no prev unit, no KV cache, no history."""

    def __init__(self, context: DiTInferenceContext):
        super().__init__()
        self.dit = context.dit
        self.coordinate_proj = context.coordinate_proj
        self.hidden_patch_size = int(context.hidden_patch_size)
        self.latent_patch_size = int(context.latent_patch_size)
        self.hidden_size = int(context.hidden_size)
        self.fm_sigma = float(context.fm_sigma)

    def forward(
        self,
        z: torch.Tensor,
        *,
        current_hidden: torch.Tensor,
        all_mods: torch.Tensor,
        t: torch.Tensor | None = None,
        cfg_current_hidden: torch.Tensor | None = None,
        guidance_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del t
        batch_size = current_hidden.size(0)
        z_proj = self.coordinate_proj(z)
        x = concat_branches(
            [current_hidden, z_proj],
            ([cfg_current_hidden, z_proj] if cfg_current_hidden is not None else None),
        )
        vt = self.dit.run_blocks(x=x, all_mods=all_mods)
        vt = vt[:, self.hidden_patch_size :]
        if cfg_current_hidden is None:
            return vt
        if guidance_scale is None:
            raise ValueError("FlowMatching current step requires guidance_scale.")
        return apply_cfg(vt, batch_size=batch_size, guidance=guidance_scale)


class NextPatchStep(nn.Module):
    """Tail patch: prev_unit (fresh KV) + current_hidden + noise → velocity, KV updates."""

    def __init__(self, context: DiTInferenceContext, *, attn_backend: str):
        super().__init__()
        if attn_backend not in {"sdpa", "flex"}:
            raise ValueError(
                f"Unsupported KV cache attention backend: {attn_backend!r}."
            )
        self.dit = context.dit
        self.coordinate_proj = context.coordinate_proj
        self.hidden_patch_size = int(context.hidden_patch_size)
        self.latent_patch_size = int(context.latent_patch_size)
        self.unit_len = self.hidden_patch_size + self.latent_patch_size
        self.hidden_size = int(context.hidden_size)
        self.attn_backend = attn_backend
        self.fm_sigma = float(context.fm_sigma)

        attn0 = self.dit.blocks[0].attn
        self.num_heads = int(attn0.num_heads)
        self.head_dim = int(attn0.head_dim)
        self.num_layers = len(self.dit.blocks)

        current_latent_start = self.unit_len + self.hidden_patch_size
        self.current_latent_slice = slice(
            current_latent_start, current_latent_start + self.latent_patch_size
        )

    def _run_cached_blocks(
        self,
        x: torch.Tensor,
        *,
        all_mods: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        block_mask: Any | None,
        sdpa_mask: torch.Tensor | None,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_batch = x.size(0)
        new_k = torch.empty(
            self.num_layers,
            branch_batch,
            self.num_heads,
            self.unit_len,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        new_v = torch.empty_like(new_k)
        use_flex = self.attn_backend == "flex"
        if use_flex and block_mask is None:
            raise ValueError("Flex attention backend requires a BlockMask.")

        def attention(layer_idx: int, block: nn.Module, attn_in: torch.Tensor):
            out, k, v = project_attention(
                block.attn,
                attn_in,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                key_prefix=cache_k[layer_idx],
                value_prefix=cache_v[layer_idx],
                block_mask=block_mask if use_flex else None,
                attn_mask=sdpa_mask,
                dropout_p=0.0,
            )
            new_k[layer_idx].copy_(k[:, :, : self.unit_len, :])
            new_v[layer_idx].copy_(v[:, :, : self.unit_len, :])
            return out

        x, final_mod = self.dit.run_modulated_blocks(
            x=x, all_mods=all_mods, attention=attention
        )
        velocity = self.dit.apply_final_layer(
            x[:, self.current_latent_slice], final_mod
        )
        return velocity, new_k, new_v

    def forward(
        self,
        z: torch.Tensor,
        *,
        prev_unit: torch.Tensor,
        current_hidden: torch.Tensor,
        all_mods: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        t: torch.Tensor | None = None,
        cfg_prev_unit: torch.Tensor | None = None,
        cfg_current_hidden: torch.Tensor | None = None,
        guidance_scale: torch.Tensor | None = None,
        block_mask: Any | None = None,
        sdpa_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        has_cfg = cfg_prev_unit is not None
        if has_cfg != (cfg_current_hidden is not None):
            raise ValueError(
                "NextPatchStep CFG requires both cfg_prev_unit and cfg_current_hidden."
            )
        batch_size = current_hidden.size(0)
        z_proj = self.coordinate_proj(z)
        x = concat_branches(
            [prev_unit, current_hidden, z_proj],
            ([cfg_prev_unit, cfg_current_hidden, z_proj] if has_cfg else None),
        )
        vt, new_k, new_v = self._run_cached_blocks(
            x,
            all_mods=all_mods,
            cache_k=cache_k,
            cache_v=cache_v,
            block_mask=block_mask,
            sdpa_mask=sdpa_mask,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
        )
        del t
        if not has_cfg:
            return vt, new_k, new_v
        if guidance_scale is None:
            raise ValueError("FlowMatching NextPatchStep requires guidance_scale.")
        velocity = apply_cfg(vt, batch_size=batch_size, guidance=guidance_scale)
        return velocity, new_k, new_v


class KvPrefillStep(nn.Module):
    """KV cache warmup: run the full modulated stack once, collect per-layer K/V.

    Wrapped as an ``nn.Module`` only because ``torch.compile`` needs one.
    """

    def __init__(self, context: DiTInferenceContext):
        super().__init__()
        self.dit = context.dit
        attn0 = self.dit.blocks[0].attn
        self.num_heads = int(attn0.num_heads)
        self.head_dim = int(attn0.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        all_mods: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_k: list[torch.Tensor] = []
        new_v: list[torch.Tensor] = []

        def attention(_layer_idx: int, block: nn.Module, attn_in: torch.Tensor):
            out, k, v = project_attention(
                block.attn,
                attn_in,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                is_causal=True,
                dropout_p=0.0,
            )
            new_k.append(k)
            new_v.append(v)
            return out

        self.dit.run_modulated_blocks(x=x, all_mods=all_mods, attention=attention)
        return torch.stack(new_k), torch.stack(new_v)


# --------------------------------------------------------------------------- #
# Section 5. Compiled/cached runner bundle (one per capacity bucket).
# --------------------------------------------------------------------------- #


class CachedDiTRunner:
    """Compiled step modules + memoized masks/rotaries for one capacity bucket.

    A bucket is identified by ``(capacity_patches, device, dtype, backend,
    compile_step)`` at the solver level; this class holds everything that
    depends on those and nothing that depends on the actual sequence being
    decoded.
    """

    PREFILL_COMPILE_BUCKET_PATCHES = 192

    def __init__(
        self,
        *,
        context: DiTInferenceContext,
        capacity_patches: int,
        capacity_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        attn_backend: str,
        compile_step: bool,
        block_mask_size: int = 64,
        branch_multiplier: int = 2,
    ):
        self.context = context
        self.capacity_patches = int(capacity_patches)
        self.capacity_tokens = int(capacity_tokens)
        self.dtype = dtype
        self.device = device
        self.attn_backend = attn_backend
        self.block_mask_size = int(block_mask_size)
        self.branch_multiplier = int(branch_multiplier)
        self.unit_len = int(context.unit_len)
        self.prefill_compile_bucket_tokens = (
            self.PREFILL_COMPILE_BUCKET_PATCHES * self.unit_len
        )

        attn0 = context.dit.blocks[0].attn
        self.num_layers = len(context.dit.blocks)
        self.num_heads = int(attn0.num_heads)
        self.head_dim = int(attn0.head_dim)

        self.step = _maybe_compile(
            NextPatchStep(context, attn_backend=attn_backend).eval(),
            enabled=compile_step,
        )
        self.current_step = _maybe_compile(
            FirstPatchStep(context).eval(),
            enabled=compile_step,
        )
        self.eager_prefill_step = KvPrefillStep(context).eval()
        self.compiled_prefill_step = (
            compile_module_forward(self.eager_prefill_step) if compile_step else None
        )

        self._mask_cache: dict[int, tuple[Any | None, torch.Tensor | None]] = {}
        self._rotary_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._prefill_rotary_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    # ---- Allocation ------------------------------------------------------ #

    def allocate_cache(self, *, batch_size: int, nfe: int) -> DiTKvCache:
        if int(nfe) <= 0:
            raise ValueError(f"nfe must be positive, got {nfe}.")
        cache_k = torch.zeros(
            (
                int(nfe),
                self.num_layers,
                self.branch_multiplier * int(batch_size),
                self.num_heads,
                self.capacity_tokens,
                self.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        return DiTKvCache(
            capacity_patches=self.capacity_patches,
            capacity_tokens=self.capacity_tokens,
            nfe=int(nfe),
            cache_k=cache_k,
            cache_v=torch.zeros_like(cache_k),
        )

    # ---- Memoized masks / rotaries -------------------------------------- #

    def masks_for(
        self, *, valid_persistent_tokens: int
    ) -> tuple[Any | None, torch.Tensor | None]:
        key = int(valid_persistent_tokens)
        if (cached := self._mask_cache.get(key)) is not None:
            return cached
        if self.attn_backend == "flex":
            masks = (self._build_flex_mask(key), None)
        else:
            sdpa_mask = build_causal_update_mask(
                capacity_tokens=self.capacity_tokens,
                valid_persistent_tokens=key,
                prev_len=self.unit_len,
                current_len=self.unit_len,
                device=self.device,
            )
            masks = (None, sdpa_mask)
        self._mask_cache[key] = masks
        return masks

    def _build_flex_mask(self, valid_persistent_tokens: int) -> Any:
        valid_tokens = torch.tensor(
            valid_persistent_tokens, device=self.device, dtype=torch.long
        ).reshape(())
        tail_start = int(self.capacity_tokens)
        unit_len = int(self.unit_len)

        def mask_mod(b, h, q_idx, kv_idx):
            del b, h
            tail_idx = kv_idx - tail_start
            prev_query = q_idx < unit_len
            prev_causal = (tail_idx >= 0) & (tail_idx < unit_len) & (tail_idx <= q_idx)
            return (kv_idx < valid_tokens) | torch.where(
                prev_query,
                prev_causal,
                tail_idx >= 0,
            )

        return _compiled_create_block_mask(
            mask_mod,
            1,
            None,
            2 * unit_len,
            self.capacity_tokens + 2 * unit_len,
            device=self.device,
            BLOCK_SIZE=self.block_mask_size,
        )

    def rotary_for(self, *, start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = int(start_pos)
        if (cached := self._rotary_cache.get(key)) is not None:
            return cached
        rotary = build_rotary_cos_sin(
            self.context.dit.blocks[0].attn.rotary,
            start_pos=key,
            seq_len=2 * self.unit_len,
            device=self.device,
            batched=True,
        )
        self._rotary_cache[key] = rotary
        return rotary

    def _prefill_rotary_for_compile(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._prefill_rotary_cache is not None:
            return self._prefill_rotary_cache
        self._prefill_rotary_cache = build_rotary_cos_sin(
            self.context.dit.blocks[0].attn.rotary,
            start_pos=0,
            seq_len=self.prefill_compile_bucket_tokens,
            device=self.device,
            batched=True,
        )
        return self._prefill_rotary_cache

    # ---- KV prefill from a prefix --------------------------------------- #

    def prefill(
        self,
        *,
        kv_cache: DiTKvCache,
        prefix_sequence: torch.Tensor,
        all_mods_by_ode: torch.Tensor,
        cfg_prefix_sequence: torch.Tensor | None = None,
    ) -> None:
        prefix_len = int(prefix_sequence.size(1))
        if prefix_len <= 0:
            kv_cache.valid_tokens = 0
            return
        if prefix_len > kv_cache.capacity_tokens:
            raise ValueError(
                "KV prefill exceeds cache capacity: "
                f"prefix_len={prefix_len} capacity={kv_cache.capacity_tokens}."
            )
        if not self._prefill_compiled(
            kv_cache=kv_cache,
            prefix_sequence=prefix_sequence,
            cfg_prefix_sequence=cfg_prefix_sequence,
            all_mods_by_ode=all_mods_by_ode,
        ):
            self._prefill_eager(
                kv_cache=kv_cache,
                prefix_sequence=prefix_sequence,
                cfg_prefix_sequence=cfg_prefix_sequence,
                all_mods_by_ode=all_mods_by_ode,
            )
        kv_cache.valid_tokens = prefix_len

    def _prefill_eager(
        self,
        *,
        kv_cache: DiTKvCache,
        prefix_sequence: torch.Tensor,
        all_mods_by_ode: torch.Tensor,
        cfg_prefix_sequence: torch.Tensor | None,
    ) -> None:
        prefix_len = int(prefix_sequence.size(1))
        rotary_cos, rotary_sin = build_rotary_cos_sin(
            self.context.dit.blocks[0].attn.rotary,
            start_pos=0,
            seq_len=prefix_len,
            device=prefix_sequence.device,
            batched=True,
        )
        x_base = (
            prefix_sequence
            if cfg_prefix_sequence is None
            else torch.cat([prefix_sequence, cfg_prefix_sequence], dim=0)
        )
        with torch.no_grad():
            for ode_idx in range(int(all_mods_by_ode.size(0))):
                new_k, new_v = self.eager_prefill_step(
                    x_base,
                    all_mods=all_mods_by_ode[ode_idx],
                    rotary_cos=rotary_cos,
                    rotary_sin=rotary_sin,
                )
                kv_cache.cache_k[ode_idx, :, :, :, :prefix_len, :].copy_(new_k)
                kv_cache.cache_v[ode_idx, :, :, :, :prefix_len, :].copy_(new_v)

    def _prefill_compiled(
        self,
        *,
        kv_cache: DiTKvCache,
        prefix_sequence: torch.Tensor,
        all_mods_by_ode: torch.Tensor,
        cfg_prefix_sequence: torch.Tensor | None,
    ) -> bool:
        if self.compiled_prefill_step is None:
            return False
        prefix_len = int(prefix_sequence.size(1))
        bucket_tokens = int(self.prefill_compile_bucket_tokens)
        if prefix_len > bucket_tokens:
            return False

        batch_size = int(prefix_sequence.size(0))
        branch_batch = batch_size if cfg_prefix_sequence is None else 2 * batch_size
        x_base = prefix_sequence.new_zeros(
            branch_batch, bucket_tokens, int(prefix_sequence.size(2))
        )
        x_base[:batch_size, :prefix_len].copy_(prefix_sequence)
        if cfg_prefix_sequence is not None:
            x_base[batch_size:, :prefix_len].copy_(cfg_prefix_sequence)

        rotary_cos, rotary_sin = self._prefill_rotary_for_compile()
        with torch.no_grad():
            for ode_idx in range(int(all_mods_by_ode.size(0))):
                new_k, new_v = self.compiled_prefill_step(
                    x_base,
                    all_mods=all_mods_by_ode[ode_idx],
                    rotary_cos=rotary_cos,
                    rotary_sin=rotary_sin,
                )
                kv_cache.cache_k[ode_idx, :, :, :, :prefix_len, :].copy_(
                    new_k[:, :, :, :prefix_len, :]
                )
                kv_cache.cache_v[ode_idx, :, :, :, :prefix_len, :].copy_(
                    new_v[:, :, :, :prefix_len, :]
                )
        return True


# --------------------------------------------------------------------------- #
# Section 6. Eager (non-optimize) runner.
# --------------------------------------------------------------------------- #


class EagerDiTRunner:
    """Reference decoder used when compilation/optimization is disabled.

    No KV cache, no bucketing — every ODE step reruns the full stack. This is
    what ``torchdiffeq.odeint`` drives for FlowMatching.
    """

    def __init__(self, *, context: DiTInferenceContext):
        self.context = context
        self.dit = context.dit
        self.coordinate_proj = context.coordinate_proj
        self.hidden_patch_size = int(context.hidden_patch_size)
        self.latent_patch_size = int(context.latent_patch_size)
        self.latent_dim = int(context.noise_latent_dim)
        self.hidden_size = int(context.hidden_size)
        self.fm_sigma = float(context.fm_sigma)

    # ---- Static per-decode inputs --------------------------------------- #

    def _build_decode_mask(
        self, *, total_len: int, fm_seq_len: int, device: torch.device
    ) -> torch.Tensor:
        mask = torch.zeros((1, total_len, total_len), dtype=torch.bool, device=device)
        latent_start = total_len - self.latent_patch_size
        block_start = int(fm_seq_len) - self.hidden_patch_size
        if block_start > 0:
            mask[:, :block_start, :block_start] = (
                torch.ones((block_start, block_start), dtype=torch.bool, device=device)
                .triu(1)
                .logical_not()
            )
        mask[:, block_start:fm_seq_len, :fm_seq_len] = True
        mask[:, block_start:fm_seq_len, latent_start:] = True
        mask[:, latent_start:, :fm_seq_len] = True
        mask[:, latent_start:, latent_start:] = True
        return mask

    def _build_pos_ids(
        self, *, total_len: int, fm_seq_len: int, device: torch.device
    ) -> torch.Tensor:
        pos_ids = torch.zeros((1, total_len), dtype=torch.float32, device=device)
        latent_start = total_len - self.latent_patch_size
        pos_ids[:, :fm_seq_len] = torch.arange(
            fm_seq_len, dtype=pos_ids.dtype, device=device
        )
        pos_ids[:, latent_start:] = torch.arange(
            fm_seq_len,
            fm_seq_len + self.latent_patch_size,
            dtype=pos_ids.dtype,
            device=device,
        )
        return pos_ids

    def _prepare_inputs(
        self,
        *,
        sequence: torch.Tensor,
        fm_seq_len: int,
        cfg_sequence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        fm_seq_len = int(fm_seq_len)
        total_len = fm_seq_len + self.latent_patch_size

        def _pad(src: torch.Tensor) -> torch.Tensor:
            padded = src.new_zeros(src.size(0), total_len, self.hidden_size)
            padded[:, :fm_seq_len].copy_(src[:, :fm_seq_len])
            return padded

        input_sequence = _pad(sequence)
        cfg_input_sequence = None if cfg_sequence is None else _pad(cfg_sequence)
        attn_mask = self._build_decode_mask(
            total_len=total_len, fm_seq_len=fm_seq_len, device=sequence.device
        )
        pos_ids = self._build_pos_ids(
            total_len=total_len, fm_seq_len=fm_seq_len, device=sequence.device
        )
        return input_sequence, cfg_input_sequence, attn_mask, pos_ids

    def _splat_noise(self, z_proj: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        latent_start = base.size(1) - self.latent_patch_size
        out = base.clone()
        out[:, latent_start:] = z_proj
        return out

    # ---- FlowMatching --------------------------------------------------- #

    def _flow_matching_velocity(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        input_sequence: torch.Tensor,
        cfg_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor,
        g_cond: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> torch.Tensor:
        if input_sequence.shape != cfg_sequence.shape:
            raise ValueError(
                "FlowMatching full-compute inputs require matching CFG shapes."
            )
        if input_sequence.size(1) < self.latent_patch_size:
            raise ValueError(
                "FlowMatching full-compute input must reserve a latent patch slot."
            )
        batch_size = input_sequence.size(0)
        latent_start = input_sequence.size(1) - self.latent_patch_size
        z_proj = self.coordinate_proj(z)
        x = torch.cat(
            [
                self._splat_noise(z_proj, input_sequence),
                self._splat_noise(z_proj, cfg_sequence),
            ],
            dim=0,
        )
        timesteps = t.reshape(1).expand(x.size(0))
        g_cond = g_cond.to(device=x.device, dtype=x.dtype)
        g_cond_branches = torch.cat([g_cond, torch.zeros_like(g_cond)], dim=0)
        pred = self.dit(
            x=x,
            timesteps=timesteps,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            g_cond=g_cond_branches,
        )[:, latent_start:]
        del t
        return apply_cfg(pred, batch_size=batch_size, guidance=guidance_scale)

    def _decode_flow_matching(
        self,
        *,
        sequence: torch.Tensor,
        cfg_sequence: torch.Tensor,
        fm_seq_len: int,
        g_cond: torch.Tensor,
        nfe: int,
        ode_method: str,
        guidance_scale: float,
    ) -> torch.Tensor:
        input_sequence, cfg_input_sequence, attn_mask, pos_ids = self._prepare_inputs(
            sequence=sequence, cfg_sequence=cfg_sequence, fm_seq_len=fm_seq_len
        )
        assert cfg_input_sequence is not None  # validated by decode_next
        guidance = sequence.new_tensor(float(guidance_scale))

        def solver(t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            return self._flow_matching_velocity(
                t,
                z,
                input_sequence=input_sequence,
                cfg_sequence=cfg_input_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                g_cond=g_cond,
                guidance_scale=guidance,
            )

        noise = torch.randn(
            (sequence.size(0), self.latent_patch_size, self.latent_dim),
            dtype=sequence.dtype,
            device=sequence.device,
        )
        times = torch.tensor([0.0, 1.0], dtype=sequence.dtype, device=sequence.device)
        if ode_method in {"euler", "midpoint", "rk4"}:
            options = {"step_size": 1.0 / int(nfe)}
        else:
            logger.warning(
                "Using adaptive step size ODE solver for FlowMatching full-compute "
                "DiT; NFE is not guaranteed: ode_method={}",
                ode_method,
            )
            options = {}
        trajectory = odeint(
            func=solver,
            y0=noise,
            t=times,
            atol=1e-5,
            rtol=1e-5,
            method=ode_method,
            options=options,
        )
        return trajectory[-1]

    # ---- MeanFlow ------------------------------------------------------- #

    def _meanflow_step(
        self,
        z: torch.Tensor,
        *,
        t: torch.Tensor,
        dt: torch.Tensor,
        input_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor,
        g_cond: torch.Tensor,
    ) -> torch.Tensor:
        if input_sequence.size(1) < self.latent_patch_size:
            raise ValueError(
                "MeanFlow full-compute input must reserve a latent patch slot."
            )
        latent_start = input_sequence.size(1) - self.latent_patch_size
        x = self._splat_noise(self.coordinate_proj(z), input_sequence)
        velocity = self.dit(
            x=x,
            timesteps=t,
            duration=dt,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            g_cond=g_cond.to(device=x.device, dtype=x.dtype),
        )[:, latent_start:]
        return z + velocity * dt.view(-1, 1, 1)

    def _decode_meanflow(
        self,
        *,
        sequence: torch.Tensor,
        fm_seq_len: int,
        g_cond: torch.Tensor,
        nfe: int,
    ) -> torch.Tensor:
        input_sequence, _cfg, attn_mask, pos_ids = self._prepare_inputs(
            sequence=sequence, fm_seq_len=fm_seq_len
        )
        z = torch.randn(
            (sequence.size(0), self.latent_patch_size, self.latent_dim),
            dtype=sequence.dtype,
            device=sequence.device,
        )
        times = torch.linspace(
            0.0, 1.0, int(nfe) + 1, device=sequence.device, dtype=sequence.dtype
        )
        for step in range(int(nfe)):
            t = times[step].expand(sequence.size(0))
            dt = (times[step + 1] - times[step]).expand(sequence.size(0))
            z = self._meanflow_step(
                z,
                t=t,
                dt=dt,
                input_sequence=input_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                g_cond=g_cond,
            ).clone()
        return z

    # ---- Entry ---------------------------------------------------------- #

    def decode_next(
        self,
        *,
        sequence: torch.Tensor,
        fm_seq_len: int,
        g_cond: torch.Tensor,
        nfe: int,
        meanflow: bool,
        cfg_sequence: torch.Tensor | None = None,
        guidance_scale: float | None = None,
        ode_method: str = "euler",
    ) -> torch.Tensor:
        if meanflow:
            return self._decode_meanflow(
                sequence=sequence,
                fm_seq_len=fm_seq_len,
                g_cond=g_cond,
                nfe=nfe,
            )
        if cfg_sequence is None:
            raise ValueError("FlowMatching full-compute DiT requires cfg_sequence.")
        if guidance_scale is None:
            raise ValueError("FlowMatching full-compute DiT requires guidance_scale.")
        return self._decode_flow_matching(
            sequence=sequence,
            cfg_sequence=cfg_sequence,
            fm_seq_len=fm_seq_len,
            g_cond=g_cond,
            nfe=nfe,
            ode_method=ode_method,
            guidance_scale=guidance_scale,
        )


# --------------------------------------------------------------------------- #
# Section 7. Solver: orchestrates streaming decode.
# --------------------------------------------------------------------------- #


def _resolve_kv_attention_backend(
    *, optimize: bool, on_cuda: bool, default_backend: str | None = None
) -> str:
    """Pick the KV-cache attention backend (flex vs sdpa) from env / device."""
    if not optimize:
        return "sdpa"
    env = os.environ.get("DOTS_TTS_DELAYED_DIT_BACKEND")
    if env is None:
        if default_backend is None:
            return "flex" if on_cuda else "sdpa"
        env = default_backend
    env = env.lower()
    if env not in {"sdpa", "flex"}:
        raise ValueError(
            f"DOTS_TTS_DELAYED_DIT_BACKEND must be 'sdpa' or 'flex', got {env!r}."
        )
    if env == "flex" and not on_cuda:
        raise ValueError("KV cache Flex attention backend requires CUDA.")
    return env


class DiTSolver:
    """Streaming DiT decoder with a KV cache across ODE steps (optimize mode).

    In non-optimize mode this delegates to :class:`EagerDiTRunner`. In
    optimize mode it maintains a per-bucket :class:`CachedDiTRunner` and
    incrementally warms a :class:`DiTKvCache`.
    """

    def __init__(
        self,
        context: DiTInferenceContext,
        *,
        optimize: bool,
        bucket_resolver: Callable[[int], int],
        meanflow: bool = False,
    ):
        self.context = context
        self.optimize = bool(optimize)
        self.bucket_resolver = bucket_resolver
        self.meanflow = bool(meanflow)
        self.branch_multiplier = 1 if self.meanflow else 2
        self._cached_runners: dict[tuple[Any, ...], CachedDiTRunner] = {}
        self._eager_runners: dict[tuple[Any, ...], EagerDiTRunner] = {}
        self._backend_by_device_type: dict[str, str] = {}

    # ---- Runner lookup -------------------------------------------------- #

    def _resolve_bucket_patches(self, fm_seq_len: int) -> int:
        requested = int(fm_seq_len)
        if requested <= 0:
            raise ValueError("fm_seq_len must be positive.")
        requested_patch_count = max(
            1, (requested + self.context.unit_len - 1) // self.context.unit_len
        )
        if not self.optimize:
            return requested_patch_count
        return int(self.bucket_resolver(requested_patch_count))

    def _backend_for(self, device: torch.device) -> str:
        key = device.type
        if key not in self._backend_by_device_type:
            self._backend_by_device_type[key] = _resolve_kv_attention_backend(
                optimize=self.optimize, on_cuda=(key == "cuda")
            )
        return self._backend_by_device_type[key]

    def _get_cached_runner(
        self, *, capacity_patches: int, device: torch.device, dtype: torch.dtype
    ) -> CachedDiTRunner:
        backend = self._backend_for(device)
        capacity_tokens = int(capacity_patches) * self.context.unit_len
        compile_step = bool(self.optimize and device.type == "cuda")
        key = (int(capacity_patches), str(device), dtype, backend, compile_step)
        runner = self._cached_runners.get(key)
        if runner is None:
            runner = CachedDiTRunner(
                context=self.context,
                capacity_patches=int(capacity_patches),
                capacity_tokens=capacity_tokens,
                dtype=dtype,
                device=device,
                attn_backend=backend,
                compile_step=compile_step,
                branch_multiplier=self.branch_multiplier,
            )
            self._cached_runners[key] = runner
        return runner

    def _get_eager_runner(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> EagerDiTRunner:
        key = (str(device), dtype)
        runner = self._eager_runners.get(key)
        if runner is None:
            runner = EagerDiTRunner(context=self.context)
            self._eager_runners[key] = runner
        return runner

    # ---- Cache management ----------------------------------------------- #

    def _ensure_cache(
        self,
        state: DiTSolverState,
        *,
        sequence: torch.Tensor,
        nfe: int,
    ) -> tuple[CachedDiTRunner, DiTKvCache]:
        capacity_patches = self._resolve_bucket_patches(sequence.size(1))
        runner = self._get_cached_runner(
            capacity_patches=capacity_patches,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        branch_batch = self.branch_multiplier * int(sequence.size(0))
        if (
            isinstance(state.cache, DiTKvCache)
            and state.cache.capacity_patches == runner.capacity_patches
            and state.cache.capacity_tokens == runner.capacity_tokens
            and state.cache.nfe == int(nfe)
            and int(state.cache.cache_k.size(2)) == branch_batch
        ):
            return runner, state.cache

        next_cache = runner.allocate_cache(batch_size=sequence.size(0), nfe=int(nfe))
        if (
            isinstance(state.cache, DiTKvCache)
            and state.cache.nfe == int(nfe)
            and int(state.cache.cache_k.size(2)) == branch_batch
        ):
            next_cache.copy_prefix_from(state.cache)
        state.cache = next_cache
        return runner, next_cache

    # ---- Schedule / mods preparation ------------------------------------ #

    def _prepare_schedule(
        self,
        state: DiTSolverState,
        *,
        g_cond: torch.Tensor,
        nfe: int,
    ) -> tuple[torch.Tensor, ODESchedule]:
        """Compute (and cache) the per-ODE-step AdaLN mods and the time grid.

        Cached across streaming decode calls as long as ``nfe`` doesn't change.
        ``g_cond`` is assumed constant within one generation (it's derived once
        from the prompt). The solver itself is per-mode so ``state`` never sees
        a schedule from the other mode.
        """
        cached_valid = (
            state.all_mods_by_ode is not None
            and state.schedule is not None
            and int(state.all_mods_by_ode.size(0)) == int(nfe)
        )
        if cached_valid:
            assert state.all_mods_by_ode is not None and state.schedule is not None
            return state.all_mods_by_ode, state.schedule

        if self.meanflow:
            all_mods, schedule = self._prepare_schedule_meanflow(g_cond=g_cond, nfe=nfe)
        else:
            all_mods, schedule = self._prepare_schedule_flow_matching(
                g_cond=g_cond, nfe=nfe
            )
        state.all_mods_by_ode = all_mods
        state.schedule = schedule
        return all_mods, schedule

    def _prepare_schedule_meanflow(
        self, *, g_cond: torch.Tensor, nfe: int
    ) -> tuple[torch.Tensor, ODESchedule]:
        dit = self.context.dit
        device, dtype = g_cond.device, g_cond.dtype
        nfe = int(nfe)
        batch_size = int(g_cond.size(0))

        times_grid = torch.linspace(0.0, 1.0, nfe + 1, device=device, dtype=dtype)
        times = times_grid[:-1]
        durations = times_grid[1:] - times

        with torch.no_grad():
            t_t = times[:, None].expand(nfe, batch_size).reshape(-1)
            c = dit.time_embedder(t_t)
            if dit.duration_embedder is not None:
                dt_t = durations[:, None].expand(nfe, batch_size).reshape(-1)
                c = c + dit.duration_embedder(dt_t)
            g_flat = g_cond[None].expand(nfe, -1, -1).reshape(nfe * batch_size, -1)
            mods = dit.fused_adaln(c + g_flat)
        return (
            mods.reshape(nfe, batch_size, -1),
            ODESchedule(mode=MEANFLOW_MODE, times=times, durations=durations),
        )

    def _prepare_schedule_flow_matching(
        self, *, g_cond: torch.Tensor, nfe: int
    ) -> tuple[torch.Tensor, ODESchedule]:
        dit = self.context.dit
        device, dtype = g_cond.device, g_cond.dtype
        nfe = int(nfe)
        g_branches = torch.cat([g_cond, torch.zeros_like(g_cond)], dim=0)
        branch_batch = int(g_branches.size(0))

        times = torch.arange(nfe, device=device, dtype=dtype) / nfe
        with torch.no_grad():
            t_t = times[:, None].expand(nfe, branch_batch).reshape(-1)
            g_flat = (
                g_branches[None].expand(nfe, -1, -1).reshape(nfe * branch_batch, -1)
            )
            mods = dit.fused_adaln(dit.time_embedder(t_t) + g_flat)
        return (
            mods.reshape(nfe, branch_batch, -1),
            ODESchedule(mode=FLOW_MATCHING_MODE, times=times),
        )

    # ---- Public entrypoint ---------------------------------------------- #

    def decode_next(
        self,
        state: DiTSolverState,
        *,
        sequence: torch.Tensor,
        fm_seq_len: int,
        null_g_cond: torch.Tensor,
        g_cond: torch.Tensor | None,
        nfe: int,
        cfg_sequence: torch.Tensor | None = None,
        ode_method: str = "euler",
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        self._validate_decode_args(
            fm_seq_len=fm_seq_len,
            nfe=nfe,
            cfg_sequence=cfg_sequence,
            guidance_scale=guidance_scale,
        )
        g_cond = (
            null_g_cond
            if g_cond is None
            else g_cond.to(device=null_g_cond.device, dtype=sequence.dtype)
        )

        if not self.optimize:
            runner = self._get_eager_runner(
                device=sequence.device, dtype=sequence.dtype
            )
            return runner.decode_next(
                sequence=sequence,
                cfg_sequence=cfg_sequence,
                fm_seq_len=int(fm_seq_len),
                g_cond=g_cond,
                nfe=int(nfe),
                meanflow=self.meanflow,
                ode_method=ode_method,
                guidance_scale=guidance_scale,
            )
        return self._decode_optimized(
            state,
            sequence=sequence,
            cfg_sequence=cfg_sequence,
            fm_seq_len=int(fm_seq_len),
            g_cond=g_cond,
            nfe=int(nfe),
            guidance_scale=guidance_scale,
        )

    # ---- Optimized (KV cache + compile) path ---------------------------- #

    def _validate_decode_args(
        self,
        *,
        fm_seq_len: int,
        nfe: int,
        cfg_sequence: torch.Tensor | None,
        guidance_scale: float | None,
    ) -> None:
        if int(nfe) <= 0:
            raise ValueError(f"nfe must be positive, got {nfe}.")
        if int(fm_seq_len) <= 0:
            raise RuntimeError(
                "Cannot decode audio before any conditioning state has been prefetched."
            )
        if self.meanflow:
            if cfg_sequence is not None:
                raise ValueError("MeanFlow DiTSolver does not accept cfg_sequence.")
        else:
            if cfg_sequence is None:
                raise ValueError("FlowMatching DiTSolver requires cfg_sequence.")
            if guidance_scale is None:
                raise ValueError("FlowMatching DiTSolver requires guidance_scale.")

    def _decode_optimized(
        self,
        state: DiTSolverState,
        *,
        sequence: torch.Tensor,
        cfg_sequence: torch.Tensor | None,
        fm_seq_len: int,
        g_cond: torch.Tensor,
        nfe: int,
        guidance_scale: float | None,
    ) -> torch.Tensor:
        hidden_patch_size = int(self.context.hidden_patch_size)
        unit_len = int(self.context.unit_len)
        prefix_len = fm_seq_len - hidden_patch_size
        if prefix_len < 0:
            raise RuntimeError(
                "FM sequence does not contain the current hidden patch: "
                f"fm_seq_len={fm_seq_len} hidden_patch_size={hidden_patch_size}."
            )
        if prefix_len % unit_len != 0:
            name = "MeanFlow DiTSolver" if self.meanflow else "FlowMatching DiTSolver"
            raise RuntimeError(
                f"{name} expects finalized history to be unit-aligned: "
                f"prefix_len={prefix_len} unit_len={unit_len}."
            )

        batch_size = int(sequence.size(0))
        current_hidden = sequence[:, prefix_len : prefix_len + hidden_patch_size]
        cfg_current_hidden = (
            None
            if cfg_sequence is None
            else cfg_sequence[:, prefix_len : prefix_len + hidden_patch_size]
        )
        z = torch.randn(
            (
                batch_size,
                int(self.context.latent_patch_size),
                self.context.noise_latent_dim,
            ),
            dtype=sequence.dtype,
            device=sequence.device,
        )
        all_mods_by_ode, schedule = self._prepare_schedule(
            state, g_cond=g_cond, nfe=nfe
        )
        guidance = (
            None if self.meanflow else sequence.new_tensor(float(guidance_scale))  # type: ignore[arg-type]
        )
        flow_dt = None if self.meanflow else sequence.new_tensor(1.0 / nfe)

        if prefix_len == 0:
            return self._decode_first_patch(
                z,
                current_hidden=current_hidden,
                cfg_current_hidden=cfg_current_hidden,
                fm_seq_len=fm_seq_len,
                all_mods_by_ode=all_mods_by_ode,
                schedule=schedule,
                sequence=sequence,
                nfe=nfe,
                batch_size=batch_size,
                guidance=guidance,
                flow_dt=flow_dt,
            )
        return self._decode_with_kv_cache(
            z,
            state=state,
            sequence=sequence,
            cfg_sequence=cfg_sequence,
            fm_seq_len=fm_seq_len,
            prefix_len=prefix_len,
            current_hidden=current_hidden,
            cfg_current_hidden=cfg_current_hidden,
            all_mods_by_ode=all_mods_by_ode,
            schedule=schedule,
            nfe=nfe,
            batch_size=batch_size,
            guidance=guidance,
            flow_dt=flow_dt,
        )

    def _decode_first_patch(
        self,
        z: torch.Tensor,
        *,
        current_hidden: torch.Tensor,
        cfg_current_hidden: torch.Tensor | None,
        fm_seq_len: int,
        all_mods_by_ode: torch.Tensor,
        schedule: ODESchedule,
        sequence: torch.Tensor,
        nfe: int,
        batch_size: int,
        guidance: torch.Tensor | None,
        flow_dt: torch.Tensor | None,
    ) -> torch.Tensor:
        runner = self._get_cached_runner(
            capacity_patches=self._resolve_bucket_patches(fm_seq_len),
            device=sequence.device,
            dtype=sequence.dtype,
        )
        for ode_idx in range(nfe):
            kwargs = {
                "current_hidden": current_hidden,
                "all_mods": all_mods_by_ode[ode_idx],
                **schedule.step_kwargs(ode_idx),
            }
            if not self.meanflow:
                kwargs.update(
                    cfg_current_hidden=cfg_current_hidden,
                    guidance_scale=guidance,
                )
            vt = runner.current_step(z, **kwargs)
            z = schedule.advance(
                z,
                vt,
                ode_idx=ode_idx,
                batch_size=batch_size,
                flow_dt=flow_dt,
            )
        return z

    def _decode_with_kv_cache(
        self,
        z: torch.Tensor,
        *,
        state: DiTSolverState,
        sequence: torch.Tensor,
        cfg_sequence: torch.Tensor | None,
        fm_seq_len: int,
        prefix_len: int,
        current_hidden: torch.Tensor,
        cfg_current_hidden: torch.Tensor | None,
        all_mods_by_ode: torch.Tensor,
        schedule: ODESchedule,
        nfe: int,
        batch_size: int,
        guidance: torch.Tensor | None,
        flow_dt: torch.Tensor | None,
    ) -> torch.Tensor:
        unit_len = int(self.context.unit_len)
        persistent_len = prefix_len - unit_len
        prev_unit = sequence[:, persistent_len:prefix_len]
        cfg_prev_unit = (
            None if cfg_sequence is None else cfg_sequence[:, persistent_len:prefix_len]
        )
        runner, kv_cache = self._ensure_cache(
            state, sequence=sequence[:, :fm_seq_len], nfe=nfe
        )
        if kv_cache.valid_tokens != persistent_len:
            runner.prefill(
                kv_cache=kv_cache,
                prefix_sequence=sequence[:, :persistent_len],
                cfg_prefix_sequence=(
                    None if cfg_sequence is None else cfg_sequence[:, :persistent_len]
                ),
                all_mods_by_ode=all_mods_by_ode,
            )
        block_mask, sdpa_mask = runner.masks_for(valid_persistent_tokens=persistent_len)
        rotary_cos, rotary_sin = runner.rotary_for(start_pos=persistent_len)
        dst = slice(persistent_len, prefix_len)

        for ode_idx in range(nfe):
            kwargs = {
                "prev_unit": prev_unit,
                "current_hidden": current_hidden,
                "all_mods": all_mods_by_ode[ode_idx],
                "cache_k": kv_cache.cache_k[ode_idx],
                "cache_v": kv_cache.cache_v[ode_idx],
                "block_mask": block_mask,
                "sdpa_mask": sdpa_mask,
                "rotary_cos": rotary_cos,
                "rotary_sin": rotary_sin,
                **schedule.step_kwargs(ode_idx),
            }
            if not self.meanflow:
                kwargs.update(
                    cfg_prev_unit=cfg_prev_unit,
                    cfg_current_hidden=cfg_current_hidden,
                    guidance_scale=guidance,
                )
            vt, new_k, new_v = runner.step(z, **kwargs)
            z = schedule.advance(
                z,
                vt,
                ode_idx=ode_idx,
                batch_size=batch_size,
                flow_dt=flow_dt,
            )
            kv_cache.cache_k[ode_idx, :, :, :, dst, :].copy_(new_k)
            kv_cache.cache_v[ode_idx, :, :, :, dst, :].copy_(new_v)
        kv_cache.valid_tokens = prefix_len
        return z
