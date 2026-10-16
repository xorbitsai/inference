from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from .config import ModelConfig
from .speaker_inversion import SPEAKER_INVERSION_UNCOND_MODES, SpeakerInversionEmbedding

DURATION_SPEAKER_FUSIONS = {
    "concat",
    "adarn",
    "adarn_zero",
    "speaker_cross_attn",
    "text_cross_attn",
}
DURATION_CAPTION_FUSIONS = {"adarn_zero"}
DURATION_CAPTION_POOLINGS = {"masked_mean"}
DURATION_ARCHITECTURES = {
    "pooled",
    "token_sum_adarn_zero_no_aux",
    "token_sum_dual_adarn_zero_no_aux",
}


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.complex(torch.cos(freqs), torch.sin(freqs))


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: (B, S, H, Dh), Dh must be even.
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:3], -1, 2))
    x_ = x_ * freqs_cis[None, :, None, :]
    x_ = torch.view_as_real(x_).reshape_as(x)
    return x_.type_as(x)


def get_timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    freqs = 1000.0 * torch.exp(
        -torch.log(torch.tensor(10000.0, device=timestep.device, dtype=torch.float32))
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / half
    )
    args = timestep[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(timestep.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int | tuple[int, ...], eps: float = 1e-6):
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(x_dtype)


class LowRankAdaLN(nn.Module):
    """
    Echo-style low-rank AdaLN that returns both modulated activations and a residual gate.
    """

    def __init__(self, model_dim: int, rank: int, eps: float):
        super().__init__()
        rank = max(1, min(int(rank), int(model_dim)))
        self.eps = eps
        self.shift_down = nn.Linear(model_dim, rank, bias=False)
        self.scale_down = nn.Linear(model_dim, rank, bias=False)
        self.gate_down = nn.Linear(model_dim, rank, bias=False)
        self.shift_up = nn.Linear(rank, model_dim, bias=True)
        self.scale_up = nn.Linear(rank, model_dim, bias=True)
        self.gate_up = nn.Linear(rank, model_dim, bias=True)
        # Match Echo/JAX AdaLN behavior: zero-init output projections.
        nn.init.zeros_(self.shift_up.weight)
        nn.init.zeros_(self.scale_up.weight)
        nn.init.zeros_(self.gate_up.weight)
        if self.shift_up.bias is not None:
            nn.init.zeros_(self.shift_up.bias)
        if self.scale_up.bias is not None:
            nn.init.zeros_(self.scale_up.bias)
        if self.gate_up.bias is not None:
            nn.init.zeros_(self.gate_up.bias)

    def forward(
        self, x: torch.Tensor, cond_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = cond_embed.chunk(3, dim=-1)
        shift = self.shift_up(self.shift_down(F.silu(shift))) + shift
        scale = self.scale_up(self.scale_down(F.silu(scale))) + scale
        gate = self.gate_up(self.gate_down(F.silu(gate))) + gate

        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        x = x * (1.0 + scale) + shift
        gate = torch.tanh(gate)
        return x.to(x_dtype), gate


def patch_sequence_with_mask(
    seq: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Patch along sequence axis:
      seq: (B, S, D) -> (B, S//patch, D*patch)
      mask: (B, S) -> (B, S//patch) with all() over patch window.

    Note:
      For speaker conditioning in this project, `seq` is already in
      latent-patched space (D = latent_dim * latent_patch_size).
      This helper applies an additional sequence patching for
      `speaker_patch_size`.
    """
    if patch_size <= 1:
        return seq, mask
    if seq.ndim != 3 or mask.ndim != 2:
        raise ValueError(
            f"Expected seq=(B,S,D), mask=(B,S), got seq={tuple(seq.shape)} mask={tuple(mask.shape)}"
        )
    if seq.shape[0] != mask.shape[0] or seq.shape[1] != mask.shape[1]:
        raise ValueError(
            f"Sequence/mask shape mismatch: seq={tuple(seq.shape)}, mask={tuple(mask.shape)}. "
            "Expected matching (B,S)."
        )
    bsz, seq_len, dim = seq.shape
    usable = (seq_len // patch_size) * patch_size
    if usable <= 0:
        raise ValueError(
            f"Reference sequence too short for speaker_patch_size={patch_size}: seq_len={seq_len}"
        )
    seq = seq[:, :usable].reshape(bsz, usable // patch_size, dim * patch_size)
    mask = mask[:, :usable].reshape(bsz, usable // patch_size, patch_size).all(dim=-1)
    return seq, mask


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, norm_eps: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if (dim // heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm((self.heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((self.heads, self.head_dim), eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.wq(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        k = self.wk(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        v = self.wv(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        gate = self.gate(x)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cis[:seq_len])
        k = apply_rotary_emb(k, freqs_cis[:seq_len])

        attn_mask = None
        if key_mask is not None:
            attn_mask = key_mask[:, None, None, :]

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            is_causal=False,
        ).transpose(1, 2)
        y = y.reshape(bsz, seq_len, self.dim)
        y = y * torch.sigmoid(gate)
        return self.wo(y)


class JointAttention(nn.Module):
    """
    Echo-style joint attention over latent self tokens + conditioning contexts.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        text_ctx_dim: int,
        speaker_ctx_dim: int | None,
        caption_ctx_dim: int | None,
        norm_eps: float,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if (dim // heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wk_text = nn.Linear(text_ctx_dim, dim, bias=False)
        self.wv_text = nn.Linear(text_ctx_dim, dim, bias=False)
        self.has_speaker_condition = speaker_ctx_dim is not None
        if self.has_speaker_condition:
            self.wk_speaker = nn.Linear(int(speaker_ctx_dim), dim, bias=False)
            self.wv_speaker = nn.Linear(int(speaker_ctx_dim), dim, bias=False)
        self.has_caption_condition = caption_ctx_dim is not None
        if self.has_caption_condition:
            self.wk_caption = nn.Linear(int(caption_ctx_dim), dim, bias=False)
            self.wv_caption = nn.Linear(int(caption_ctx_dim), dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm((self.heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((self.heads, self.head_dim), eps=norm_eps)

    def _apply_rotary_half(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x_rot, x_passthrough = x.chunk(2, dim=-2)
        x_rot = apply_rotary_emb(x_rot, freqs_cis)
        return torch.cat([x_rot, x_passthrough], dim=-2)

    def project_context_kv(
        self,
        text_context: torch.Tensor,
        speaker_context: torch.Tensor | None,
        caption_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        Precompute conditioning KV projections for static conditioning.
        """
        bsz = text_context.shape[0]
        k_text = self.wk_text(text_context).reshape(
            bsz, text_context.shape[1], self.heads, self.head_dim
        )
        v_text = self.wv_text(text_context).reshape(
            bsz, text_context.shape[1], self.heads, self.head_dim
        )
        k_text = self.k_norm(k_text)
        projected: list[torch.Tensor] = [k_text, v_text]
        if self.has_speaker_condition:
            if speaker_context is None:
                raise ValueError(
                    "speaker_context is required when speaker conditioning is enabled."
                )
            if speaker_context.shape[0] != bsz:
                raise ValueError(
                    "Batch mismatch for context projection: "
                    f"text={tuple(text_context.shape)} speaker={tuple(speaker_context.shape)}"
                )
            k_speaker = self.wk_speaker(speaker_context).reshape(
                bsz, speaker_context.shape[1], self.heads, self.head_dim
            )
            v_speaker = self.wv_speaker(speaker_context).reshape(
                bsz, speaker_context.shape[1], self.heads, self.head_dim
            )
            k_speaker = self.k_norm(k_speaker)
            projected.extend([k_speaker, v_speaker])
        elif speaker_context is not None and speaker_context.shape[0] != bsz:
            raise ValueError(
                "Batch mismatch for ignored speaker context: "
                f"text={tuple(text_context.shape)} speaker={tuple(speaker_context.shape)}"
            )
        if not self.has_caption_condition:
            return tuple(projected)
        if caption_context is None:
            raise ValueError("caption_context is required when caption conditioning is enabled.")
        if caption_context.shape[0] != bsz:
            raise ValueError(
                "Batch mismatch for caption context projection: "
                f"text={tuple(text_context.shape)} caption={tuple(caption_context.shape)}"
            )
        k_caption = self.wk_caption(caption_context).reshape(
            bsz, caption_context.shape[1], self.heads, self.head_dim
        )
        v_caption = self.wv_caption(caption_context).reshape(
            bsz, caption_context.shape[1], self.heads, self.head_dim
        )
        k_caption = self.k_norm(k_caption)
        projected.extend([k_caption, v_caption])
        return tuple(projected)

    def forward(
        self,
        x: torch.Tensor,
        text_context: torch.Tensor,
        text_mask: torch.Tensor | None,
        speaker_context: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        caption_context: torch.Tensor | None,
        caption_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        context_kv: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.wq(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        k_self = self.wk(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        v_self = self.wv(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        if context_kv is None:
            projected = self.project_context_kv(
                text_context=text_context,
                speaker_context=speaker_context,
                caption_context=caption_context,
            )
        else:
            projected = context_kv
        if projected is None:
            raise RuntimeError("JointAttention projected context unexpectedly missing.")
        offset = 0
        k_text, v_text = projected[offset], projected[offset + 1]
        offset += 2
        k_speaker = None
        v_speaker = None
        if self.has_speaker_condition:
            k_speaker, v_speaker = projected[offset], projected[offset + 1]
            offset += 2
        k_caption = None
        v_caption = None
        if self.has_caption_condition:
            k_caption, v_caption = projected[offset], projected[offset + 1]

        q = self.q_norm(q)
        k_self = self.k_norm(k_self)
        q = self._apply_rotary_half(q, freqs_cis[:seq_len])
        k_self = self._apply_rotary_half(k_self, freqs_cis[:seq_len])

        if self_mask is None:
            self_mask = torch.ones((bsz, seq_len), dtype=torch.bool, device=x.device)
        if text_mask is None:
            text_mask = torch.ones(
                (bsz, text_context.shape[1]),
                dtype=torch.bool,
                device=x.device,
            )
        context_k = [k_self, k_text]
        context_v = [v_self, v_text]
        context_masks = [self_mask, text_mask]
        if self.has_speaker_condition:
            if speaker_context is None or k_speaker is None or v_speaker is None:
                raise ValueError(
                    "speaker_context is required when speaker conditioning is enabled."
                )
            if speaker_mask is None:
                speaker_mask = torch.ones(
                    (bsz, speaker_context.shape[1]),
                    dtype=torch.bool,
                    device=x.device,
                )
            context_k.append(k_speaker)
            context_v.append(v_speaker)
            context_masks.append(speaker_mask)
        if self.has_caption_condition:
            if caption_context is None:
                raise ValueError(
                    "caption_context is required when caption conditioning is enabled."
                )
            if caption_mask is None:
                caption_mask = torch.ones(
                    (bsz, caption_context.shape[1]),
                    dtype=torch.bool,
                    device=x.device,
                )
            if k_caption is None or v_caption is None:
                raise RuntimeError(
                    "Caption projections are missing despite enabled caption conditioning."
                )
            context_k.append(k_caption)
            context_v.append(v_caption)
            context_masks.append(caption_mask)

        k = torch.cat(context_k, dim=1)
        v = torch.cat(context_v, dim=1)
        attn_mask = torch.cat(context_masks, dim=1)
        attn_mask = attn_mask[:, None, None, :]

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            is_causal=False,
        ).transpose(1, 2)
        y = y.reshape(bsz, seq_len, self.dim)
        y = y * torch.sigmoid(self.gate(x))
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _safe_attention_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.ndim != 2 or mask.shape[0] != x.shape[0] or mask.shape[1] != x.shape[1]:
        raise ValueError(
            f"mask must have shape (B, S) matching x, got x={tuple(x.shape)} "
            f"mask={tuple(mask.shape)}"
        )
    mask = mask.to(device=x.device, dtype=torch.bool)
    has_any = mask.any(dim=1)
    if bool(has_any.all()):
        return x, mask
    if x.shape[1] <= 0:
        raise ValueError("Cannot attention-pool an empty sequence.")
    x = x.clone()
    mask = mask.clone()
    x[~has_any] = 0
    mask[~has_any, 0] = True
    return x, mask


class AttentionPooling(nn.Module):
    def __init__(self, dim: int, heads: int, norm_eps: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = int(dim) // int(heads)
        self.query = nn.Parameter(torch.empty(1, 1, int(dim)))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.q_norm = RMSNorm(dim, eps=norm_eps)
        self.k_norm = RMSNorm(dim, eps=norm_eps)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape (B, S, {self.dim}), got {tuple(x.shape)}")
        x, mask = _safe_attention_mask(x, mask)
        bsz, seq_len, _ = x.shape
        q = self.query.to(dtype=x.dtype).expand(bsz, -1, -1)
        q = self.wq(self.q_norm(q)).reshape(bsz, 1, self.heads, self.head_dim)
        k = self.wk(self.k_norm(x)).reshape(bsz, seq_len, self.heads, self.head_dim)
        v = self.wv(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask[:, None, None, :],
            is_causal=False,
        )
        y = y.transpose(1, 2).reshape(bsz, 1, self.dim)
        return self.wo(y).squeeze(1)


class CrossAttentionPooling(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: int,
        output_dim: int,
        heads: int,
        norm_eps: float,
    ):
        super().__init__()
        if output_dim % heads != 0:
            raise ValueError(f"output_dim={output_dim} must be divisible by heads={heads}")
        self.query_dim = int(query_dim)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.heads = int(heads)
        self.head_dim = int(output_dim) // int(heads)
        self.q_norm = RMSNorm(query_dim, eps=norm_eps)
        self.k_norm = RMSNorm(context_dim, eps=norm_eps)
        self.wq = nn.Linear(query_dim, output_dim, bias=False)
        self.wk = nn.Linear(context_dim, output_dim, bias=False)
        self.wv = nn.Linear(context_dim, output_dim, bias=False)
        self.wo = nn.Linear(output_dim, output_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 2 or query.shape[-1] != self.query_dim:
            raise ValueError(
                f"query must have shape (B, {self.query_dim}), got {tuple(query.shape)}"
            )
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must have shape (B, S, {self.context_dim}), got {tuple(context.shape)}"
            )
        context, context_mask = _safe_attention_mask(context, context_mask)
        bsz, seq_len, _ = context.shape
        q = query[:, None, :]
        q = self.wq(self.q_norm(q)).reshape(bsz, 1, self.heads, self.head_dim)
        k = self.wk(self.k_norm(context)).reshape(bsz, seq_len, self.heads, self.head_dim)
        v = self.wv(context).reshape(bsz, seq_len, self.heads, self.head_dim)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=context_mask[:, None, None, :],
            is_causal=False,
        )
        y = y.transpose(1, 2).reshape(bsz, 1, self.output_dim)
        return self.wo(y).squeeze(1)


class DurationSwiGLUBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        dropout: float,
        norm_eps: float,
        cond_dim: int | None = None,
        caption_cond_dim: int | None = None,
    ):
        super().__init__()
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.mlp = SwiGLU(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.cond_dim = cond_dim
        self.modulation = None
        if cond_dim is not None:
            self.modulation = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)
        self.caption_cond_dim = caption_cond_dim
        self.caption_modulation = None
        if caption_cond_dim is not None:
            self.caption_modulation = nn.Linear(caption_cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.caption_modulation.weight)
            nn.init.zeros_(self.caption_modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        caption_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm(x)
        if self.modulation is not None or self.caption_modulation is not None:
            shift = scale = gate = None
            if self.modulation is not None:
                if cond is None:
                    raise ValueError("cond is required for AdaRN-Zero duration blocks.")
                shift, scale, gate = self.modulation(F.silu(cond)).chunk(3, dim=-1)
            if self.caption_modulation is not None:
                if caption_cond is None:
                    raise ValueError(
                        "caption_cond is required for caption AdaRN-Zero duration blocks."
                    )
                caption_shift, caption_scale, caption_gate = self.caption_modulation(
                    F.silu(caption_cond)
                ).chunk(3, dim=-1)
                if shift is None:
                    shift, scale, gate = caption_shift, caption_scale, caption_gate
                else:
                    shift = shift + caption_shift
                    scale = scale + caption_scale
                    gate = gate + caption_gate
            if shift is None or scale is None or gate is None:
                raise RuntimeError("Duration block modulation state is incomplete.")
            if h.ndim == 3 and shift.ndim == 2:
                shift = shift.unsqueeze(1)
                scale = scale.unsqueeze(1)
                gate = gate.unsqueeze(1)
            h = h * (1.0 + scale) + shift
            return x + self.dropout(torch.tanh(gate) * self.mlp(h))
        return x + self.dropout(self.mlp(h))


class TextBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, norm_eps: float, dropout: float):
        super().__init__()
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = SelfAttention(dim, heads, norm_eps=norm_eps)
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(
            self.attention(self.attention_norm(x), key_mask=mask, freqs_cis=freqs_cis)
        )
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        norm_eps: float,
        dropout: float,
    ):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            TextBlock(
                dim=dim,
                heads=heads,
                mlp_ratio=mlp_ratio,
                norm_eps=norm_eps,
                dropout=dropout,
            )
            for _ in range(layers)
        )
        self.head_dim = dim // heads
        self.register_buffer(
            "_freqs_cis_cache", torch.empty(0, 0, dtype=torch.complex64), persistent=False
        )

    def _rope_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        cache = self._freqs_cis_cache
        if cache.device != device or cache.shape[0] < seq_len:
            cache = precompute_freqs_cis(self.head_dim, seq_len).to(device)
            self._freqs_cis_cache = cache
        return cache[:seq_len]

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.text_embedding(input_ids)
        # Hard-mask invalid tokens so fully-masked conditioning becomes truly unconditional.
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x * mask_f
        freqs = self._rope_freqs(input_ids.shape[1], x.device)
        for block in self.blocks:
            x = block(x, mask=mask, freqs_cis=freqs)
            x = x * mask_f
        return x * mask_f


class PretrainedTextBackbone(nn.Module):
    """Hugging Face backbone shared by text and caption conditioning."""

    def __init__(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        config_dict: dict[str, object] | None = None,
        load_pretrained_weights: bool = True,
    ):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
            from transformers.initialization import no_init_weights
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required when text_encoder_type='pretrained'."
            ) from exc

        if config_dict is None:
            config = AutoConfig.from_pretrained(
                repo_id,
                trust_remote_code=False,
                revision=revision,
            )
        else:
            raw_config = dict(config_dict)
            model_type = raw_config.pop("model_type", None)
            if not isinstance(model_type, str) or not model_type:
                raise ValueError("Embedded pretrained text encoder config has no model_type.")
            config = AutoConfig.for_model(model_type, **raw_config)
        if str(getattr(config, "model_type", "")).lower() == "t5gemma2":
            # Loading AutoModel would materialize the decoder and vision tower before
            # discarding them. Load only the bidirectional text encoder weights.
            try:
                from transformers.models.t5gemma2.modeling_t5gemma2 import (
                    T5Gemma2TextEncoder,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "The installed transformers version does not expose T5Gemma2TextEncoder."
                ) from exc
            if load_pretrained_weights:
                backbone = T5Gemma2TextEncoder.from_pretrained(
                    repo_id,
                    config=config.encoder.text_config,
                    eoi_token_index=config.eoi_token_index,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                    revision=revision,
                    key_mapping={r"^model\.encoder\.(.*)": r"\1"},
                )
            else:
                with no_init_weights():
                    backbone = T5Gemma2TextEncoder(
                        config.encoder.text_config,
                        eoi_token_index=config.eoi_token_index,
                    )
        else:
            if load_pretrained_weights:
                loaded_model = AutoModel.from_pretrained(
                    repo_id,
                    config=config,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                    revision=revision,
                )
            else:
                with no_init_weights():
                    loaded_model = AutoModel.from_config(
                        config,
                        trust_remote_code=False,
                    )
            if bool(getattr(config, "is_encoder_decoder", False)):
                get_encoder = getattr(loaded_model, "get_encoder", None)
                if get_encoder is None:
                    raise ValueError(
                        f"Encoder-decoder model does not expose get_encoder(): {repo_id}"
                    )
                encoder = get_encoder()
                backbone = getattr(encoder, "text_model", encoder)
            else:
                backbone = loaded_model

        hidden_size = _pretrained_hidden_size(backbone.config)
        # Register only the selected backbone. In the encoder-decoder case this
        # drops the decoder immediately after loading.
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.repo_id = str(repo_id)
        self.revision = revision
        self.config_dict = config.to_dict()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=mask,
            return_dict=True,
        )
        state = outputs.last_hidden_state
        return state * mask.unsqueeze(-1).to(dtype=state.dtype)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        method_name = (
            "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
        )
        method = getattr(self.backbone, method_name, None)
        if callable(method):
            method()


def _pretrained_hidden_size(config: object) -> int:
    candidates = (
        config,
        getattr(config, "text_config", None),
        getattr(config, "encoder", None),
        getattr(getattr(config, "encoder", None), "text_config", None),
    )
    for candidate in candidates:
        hidden_size = getattr(candidate, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
    raise ValueError(f"Could not determine pretrained encoder hidden size from {config!r}.")


class PretrainedConditionProjector(nn.Module):
    """Trainable projection from a shared pretrained backbone into a condition space."""

    def __init__(
        self,
        backbone_dim: int,
        output_dim: int,
        *,
        projector_type: str = "linear",
        hidden_ratio: float = 2.0,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        projector_type = str(projector_type).strip().lower()
        if projector_type not in {"linear", "residual_mlp"}:
            raise ValueError(
                "pretrained projector type must be 'linear' or 'residual_mlp', "
                f"got {projector_type!r}."
            )
        if hidden_ratio <= 0:
            raise ValueError(f"projector hidden_ratio must be > 0, got {hidden_ratio}.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"projector dropout must be in [0, 1], got {dropout}.")

        self.projector_type = projector_type
        self.projector = nn.Linear(backbone_dim, output_dim, bias=True)
        if backbone_dim == output_dim:
            nn.init.eye_(self.projector.weight)
        else:
            nn.init.xavier_uniform_(self.projector.weight)
        if self.projector.bias is not None:
            nn.init.zeros_(self.projector.bias)

        self.residual_norm = None
        self.residual_up = None
        self.residual_down = None
        self.residual_dropout = None
        if projector_type == "residual_mlp":
            hidden_dim = max(1, int(round(output_dim * float(hidden_ratio))))
            self.residual_norm = RMSNorm(backbone_dim, eps=norm_eps)
            self.residual_up = nn.Linear(backbone_dim, hidden_dim, bias=True)
            self.residual_down = nn.Linear(hidden_dim, output_dim, bias=True)
            self.residual_dropout = nn.Dropout(float(dropout))
            # Start exactly at the base Linear mapping. The residual branch
            # gradually comes online during projector-only warmup.
            nn.init.zeros_(self.residual_down.weight)
            nn.init.zeros_(self.residual_down.bias)

    def forward(
        self,
        backbone: PretrainedTextBackbone,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        state = backbone(input_ids, mask)
        if (
            not torch.is_autocast_enabled(state.device.type)
            and state.dtype != self.projector.weight.dtype
        ):
            state = state.to(dtype=self.projector.weight.dtype)
        projected = self.projector(state)
        if self.projector_type == "residual_mlp":
            if (
                self.residual_norm is None
                or self.residual_up is None
                or self.residual_down is None
                or self.residual_dropout is None
            ):
                raise RuntimeError("Residual pretrained projector modules are missing.")
            residual = self.residual_norm(state)
            residual = F.silu(self.residual_up(residual))
            residual = self.residual_dropout(residual)
            projected = projected + self.residual_down(residual)
        return projected * mask.unsqueeze(-1).to(dtype=projected.dtype)


class ReferenceLatentEncoder(nn.Module):
    """
    Encoder for reference latents used as speaker/style conditioning.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.in_proj = nn.Linear(cfg.speaker_patched_latent_dim, cfg.speaker_dim, bias=True)
        speaker_mlp_ratio = cfg.speaker_mlp_ratio_resolved
        self.blocks = nn.ModuleList(
            TextBlock(
                dim=cfg.speaker_dim,
                heads=cfg.speaker_heads,
                mlp_ratio=speaker_mlp_ratio,
                norm_eps=cfg.norm_eps,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.speaker_layers)
        )
        self.head_dim = cfg.speaker_dim // cfg.speaker_heads
        self.register_buffer(
            "_freqs_cis_cache", torch.empty(0, 0, dtype=torch.complex64), persistent=False
        )

    def _rope_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        cache = self._freqs_cis_cache
        if cache.device != device or cache.shape[0] < seq_len:
            cache = precompute_freqs_cis(self.head_dim, seq_len).to(device)
            self._freqs_cis_cache = cache
        return cache[:seq_len]

    def forward(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(latent)
        x = x / 6.0
        # Keep masked reference positions strictly zero across residual/MLP paths.
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x * mask_f
        freqs = self._rope_freqs(x.shape[1], x.device)
        for block in self.blocks:
            x = block(x, mask=mask, freqs_cis=freqs)
            x = x * mask_f
        return x * mask_f


class DiffusionBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attention = JointAttention(
            cfg.model_dim,
            cfg.num_heads,
            cfg.text_dim,
            cfg.speaker_dim if cfg.use_speaker_condition_resolved else None,
            cfg.caption_dim_resolved if cfg.use_caption_condition else None,
            norm_eps=cfg.norm_eps,
        )
        self.mlp = SwiGLU(cfg.model_dim, int(cfg.model_dim * cfg.mlp_ratio))
        adaln_rank = max(1, min(int(cfg.adaln_rank), int(cfg.model_dim)))
        self.attention_adaln = LowRankAdaLN(
            model_dim=cfg.model_dim,
            rank=adaln_rank,
            eps=cfg.norm_eps,
        )
        self.mlp_adaln = LowRankAdaLN(
            model_dim=cfg.model_dim,
            rank=adaln_rank,
            eps=cfg.norm_eps,
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        caption_state: torch.Tensor | None,
        caption_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        context_kv: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        h, attention_gate = self.attention_adaln(x, cond_embed)
        x = x + self.dropout(
            attention_gate
            * self.attention(
                x=h,
                text_context=text_state,
                text_mask=text_mask,
                speaker_context=speaker_state,
                speaker_mask=speaker_mask,
                caption_context=caption_state,
                caption_mask=caption_mask,
                freqs_cis=freqs_cis,
                self_mask=self_mask,
                context_kv=context_kv,
            )
        )

        h, mlp_gate = self.mlp_adaln(x, cond_embed)
        x = x + self.dropout(mlp_gate * self.mlp(h))
        return x


class DurationPredictor(nn.Module):
    def __init__(
        self,
        *,
        text_dim: int,
        aux_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        speaker_dim: int | None = None,
        speaker_fusion: str = "concat",
        caption_dim: int | None = None,
        caption_fusion: str = "adarn_zero",
        caption_pooling: str = "masked_mean",
        attention_heads: int = 8,
        norm_eps: float = 1e-5,
        architecture: str = "pooled",
        token_init_frames: float = 6.3,
    ):
        super().__init__()
        if text_dim <= 0:
            raise ValueError(f"duration predictor text_dim must be > 0, got {text_dim}")
        if aux_dim <= 0:
            raise ValueError(f"duration predictor aux_dim must be > 0, got {aux_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"duration predictor hidden_dim must be > 0, got {hidden_dim}")
        if layers <= 0:
            raise ValueError(f"duration predictor layers must be > 0, got {layers}")
        if speaker_dim is not None and speaker_dim <= 0:
            raise ValueError(f"duration predictor speaker_dim must be > 0, got {speaker_dim}")
        if caption_dim is not None and caption_dim <= 0:
            raise ValueError(f"duration predictor caption_dim must be > 0, got {caption_dim}")
        speaker_fusion = str(speaker_fusion).strip().lower()
        if speaker_fusion not in DURATION_SPEAKER_FUSIONS:
            raise ValueError(
                f"duration speaker fusion must be one of {sorted(DURATION_SPEAKER_FUSIONS)}, "
                f"got {speaker_fusion!r}"
            )
        caption_fusion = str(caption_fusion).strip().lower()
        if caption_fusion not in DURATION_CAPTION_FUSIONS:
            raise ValueError(
                f"duration caption fusion must be one of {sorted(DURATION_CAPTION_FUSIONS)}, "
                f"got {caption_fusion!r}"
            )
        caption_pooling = str(caption_pooling).strip().lower()
        if caption_pooling not in DURATION_CAPTION_POOLINGS:
            raise ValueError(
                f"duration caption pooling must be one of {sorted(DURATION_CAPTION_POOLINGS)}, "
                f"got {caption_pooling!r}"
            )
        architecture = str(architecture).strip().lower()
        if architecture not in DURATION_ARCHITECTURES:
            raise ValueError(
                "duration architecture must be one of "
                f"{sorted(DURATION_ARCHITECTURES)}, got {architecture!r}"
            )
        if attention_heads <= 0:
            raise ValueError(
                f"duration predictor attention_heads must be > 0, got {attention_heads}"
            )
        if token_init_frames <= 0:
            raise ValueError(f"duration token_init_frames must be > 0, got {token_init_frames}")
        if speaker_dim is None and speaker_fusion != "concat":
            raise ValueError(f"duration speaker fusion {speaker_fusion!r} requires speaker_dim.")
        if architecture == "token_sum_adarn_zero_no_aux" and speaker_dim is None:
            raise ValueError("token_sum_adarn_zero_no_aux requires speaker_dim.")
        if architecture == "token_sum_adarn_zero_no_aux" and speaker_fusion != "adarn_zero":
            raise ValueError(
                "token_sum_adarn_zero_no_aux uses block-level speaker AdaRN-Zero and "
                "requires speaker_fusion='adarn_zero'."
            )
        if architecture == "token_sum_dual_adarn_zero_no_aux" and (
            speaker_dim is None or caption_dim is None
        ):
            raise ValueError(
                "token_sum_dual_adarn_zero_no_aux requires speaker_dim and caption_dim."
            )
        if architecture == "token_sum_dual_adarn_zero_no_aux" and speaker_fusion != "adarn_zero":
            raise ValueError(
                "token_sum_dual_adarn_zero_no_aux uses block-level speaker AdaRN-Zero and "
                "requires speaker_fusion='adarn_zero'."
            )
        if architecture == "token_sum_dual_adarn_zero_no_aux" and caption_fusion != "adarn_zero":
            raise ValueError(
                "token_sum_dual_adarn_zero_no_aux uses block-level caption AdaRN-Zero and "
                "requires caption_fusion='adarn_zero'."
            )

        self.text_dim = int(text_dim)
        self.aux_dim = int(aux_dim)
        self.hidden_dim = int(hidden_dim)
        self.speaker_dim = None if speaker_dim is None else int(speaker_dim)
        self.speaker_fusion = speaker_fusion
        self.caption_dim = None if caption_dim is None else int(caption_dim)
        self.caption_fusion = caption_fusion
        self.caption_pooling = caption_pooling
        self.duration_architecture = architecture
        self.text_pool = None
        self.null_speaker = (
            nn.Parameter(torch.zeros(int(speaker_dim))) if speaker_dim is not None else None
        )
        self.null_caption = (
            nn.Parameter(torch.zeros(int(caption_dim))) if caption_dim is not None else None
        )
        self.text_adarn_norm = None
        self.text_adarn = None
        self.speaker_cross_attn = None
        self.text_cross_attn = None
        self.token_input_proj = None
        self.token_blocks = None
        self.token_out_norm = None
        self.token_out_proj = None

        if architecture in {
            "token_sum_adarn_zero_no_aux",
            "token_sum_dual_adarn_zero_no_aux",
        }:
            self.token_input_proj = nn.Linear(int(text_dim), int(hidden_dim))
            self.token_blocks = nn.ModuleList(
                DurationSwiGLUBlock(
                    dim=int(hidden_dim),
                    hidden_dim=int(hidden_dim),
                    dropout=float(dropout),
                    norm_eps=float(norm_eps),
                    cond_dim=int(speaker_dim),
                    caption_cond_dim=(
                        int(caption_dim)
                        if architecture == "token_sum_dual_adarn_zero_no_aux"
                        else None
                    ),
                )
                for _ in range(int(layers))
            )
            self.token_out_norm = RMSNorm(int(hidden_dim), eps=float(norm_eps))
            self.token_out_proj = nn.Linear(int(hidden_dim), 1)
            nn.init.zeros_(self.token_out_proj.weight)
            nn.init.constant_(
                self.token_out_proj.bias,
                float(math.log(math.expm1(float(token_init_frames)))),
            )
            return

        self.text_pool = AttentionPooling(
            dim=int(text_dim),
            heads=int(attention_heads),
            norm_eps=float(norm_eps),
        )

        if speaker_dim is not None:
            if speaker_fusion == "concat":
                input_dim = int(text_dim) + int(speaker_dim) + int(aux_dim)
            elif speaker_fusion == "adarn":
                input_dim = int(text_dim) + int(aux_dim)
                self.text_adarn_norm = RMSNorm(int(text_dim), eps=float(norm_eps))
                self.text_adarn = nn.Linear(int(speaker_dim), int(text_dim) * 2)
                nn.init.zeros_(self.text_adarn.weight)
                nn.init.zeros_(self.text_adarn.bias)
            elif speaker_fusion == "adarn_zero":
                input_dim = int(text_dim) + int(aux_dim)
            elif speaker_fusion == "speaker_cross_attn":
                input_dim = int(text_dim) * 2 + int(aux_dim)
                self.speaker_cross_attn = CrossAttentionPooling(
                    query_dim=int(text_dim),
                    context_dim=int(speaker_dim),
                    output_dim=int(text_dim),
                    heads=int(attention_heads),
                    norm_eps=float(norm_eps),
                )
            elif speaker_fusion == "text_cross_attn":
                input_dim = int(text_dim) + int(speaker_dim) + int(aux_dim)
                self.text_cross_attn = CrossAttentionPooling(
                    query_dim=int(speaker_dim),
                    context_dim=int(text_dim),
                    output_dim=int(text_dim),
                    heads=int(attention_heads),
                    norm_eps=float(norm_eps),
                )
            else:
                raise RuntimeError(f"Unsupported duration speaker fusion: {speaker_fusion!r}")
        else:
            input_dim = int(text_dim) + int(aux_dim)

        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        block_cond_dim = int(speaker_dim) if speaker_fusion == "adarn_zero" else None
        self.blocks = nn.ModuleList(
            DurationSwiGLUBlock(
                dim=int(hidden_dim),
                hidden_dim=int(hidden_dim),
                dropout=float(dropout),
                norm_eps=float(norm_eps),
                cond_dim=block_cond_dim,
            )
            for _ in range(int(layers))
        )
        self.out_norm = RMSNorm(int(hidden_dim), eps=float(norm_eps))
        self.out_proj = nn.Linear(int(hidden_dim), 1)

    def _speaker_vec(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        speaker_state: torch.Tensor | None,
        has_speaker: torch.Tensor,
    ) -> torch.Tensor:
        if self.null_speaker is None or self.speaker_dim is None:
            raise RuntimeError("Duration speaker modules are missing.")
        null_vec = self.null_speaker.to(device=device, dtype=dtype)[None, :].expand(batch_size, -1)
        if speaker_state is None:
            return null_vec
        if speaker_state.ndim != 3 or speaker_state.shape[0] != batch_size:
            raise ValueError(
                f"speaker_state must have shape (B, S, D), got {tuple(speaker_state.shape)}"
            )
        if speaker_state.shape[-1] != self.speaker_dim:
            raise ValueError(
                f"speaker_state last dim must be {self.speaker_dim}, got {speaker_state.shape[-1]}"
            )
        speaker_vec = speaker_state[:, 0].to(device=device, dtype=dtype)
        return torch.where(has_speaker[:, None], speaker_vec, null_vec)

    def _caption_vec(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        caption_state: torch.Tensor | None,
        caption_mask: torch.Tensor | None,
        has_caption: torch.Tensor,
    ) -> torch.Tensor:
        if self.null_caption is None or self.caption_dim is None:
            raise RuntimeError("Duration caption modules are missing.")
        null_vec = self.null_caption.to(device=device, dtype=dtype)[None, :].expand(batch_size, -1)
        if caption_state is None:
            return null_vec
        if caption_state.ndim != 3 or caption_state.shape[0] != batch_size:
            raise ValueError(
                f"caption_state must have shape (B, S, D), got {tuple(caption_state.shape)}"
            )
        if caption_state.shape[-1] != self.caption_dim:
            raise ValueError(
                f"caption_state last dim must be {self.caption_dim}, got {caption_state.shape[-1]}"
            )
        caption_state = caption_state.to(device=device, dtype=dtype)
        if caption_mask is None:
            caption_mask = torch.ones(
                (batch_size, caption_state.shape[1]), dtype=torch.bool, device=device
            )
        elif caption_mask.ndim != 2 or caption_mask.shape[:2] != caption_state.shape[:2]:
            raise ValueError(
                "caption_mask must have shape matching caption_state (B, S), "
                f"got caption_state={tuple(caption_state.shape)} mask={tuple(caption_mask.shape)}"
            )
        caption_mask = caption_mask.to(device=device, dtype=torch.bool) & has_caption[:, None]
        caption_mask_f = caption_mask.unsqueeze(-1).to(dtype=caption_state.dtype)
        denom = caption_mask_f.sum(dim=1).clamp_min(1.0)
        caption_vec = (caption_state * caption_mask_f).sum(dim=1) / denom
        return torch.where(caption_mask.any(dim=1, keepdim=True), caption_vec, null_vec)

    def _speaker_sequence(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        speaker_state: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        has_speaker: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.null_speaker is None or self.speaker_dim is None:
            raise RuntimeError("Duration speaker modules are missing.")
        null_token = self.null_speaker.to(device=device, dtype=dtype)[None, None, :].expand(
            batch_size, 1, -1
        )
        if speaker_state is None:
            return null_token, torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        if speaker_state.ndim != 3 or speaker_state.shape[0] != batch_size:
            raise ValueError(
                f"speaker_state must have shape (B, S, D), got {tuple(speaker_state.shape)}"
            )
        if speaker_state.shape[-1] != self.speaker_dim:
            raise ValueError(
                f"speaker_state last dim must be {self.speaker_dim}, got {speaker_state.shape[-1]}"
            )
        speaker_state = speaker_state.to(device=device, dtype=dtype)
        if speaker_mask is None:
            speaker_mask = torch.ones(
                (batch_size, speaker_state.shape[1]), dtype=torch.bool, device=device
            )
        elif speaker_mask.ndim != 2 or speaker_mask.shape[:2] != speaker_state.shape[:2]:
            raise ValueError(
                "speaker_mask must have shape matching speaker_state (B, S), "
                f"got speaker_state={tuple(speaker_state.shape)} mask={tuple(speaker_mask.shape)}"
            )
        speaker_mask = speaker_mask.to(device=device, dtype=torch.bool)
        real_mask = speaker_mask & has_speaker[:, None]
        fallback_mask = ~real_mask.any(dim=1, keepdim=True)
        context = torch.cat([speaker_state, null_token], dim=1)
        context_mask = torch.cat([real_mask, fallback_mask], dim=1)
        return context, context_mask

    def forward(
        self,
        text_state: torch.Tensor,
        *,
        text_mask: torch.Tensor,
        aux_features: torch.Tensor,
        speaker_state: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        has_speaker: torch.Tensor | None = None,
        caption_state: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
        has_caption: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_state.ndim != 3 or text_state.shape[-1] != self.text_dim:
            raise ValueError(
                f"text_state must have shape (B, S, {self.text_dim}), got {tuple(text_state.shape)}"
            )
        if aux_features.ndim != 2 or aux_features.shape[1] != self.aux_dim:
            raise ValueError(
                f"aux_features must have shape (B, {self.aux_dim}), got {tuple(aux_features.shape)}"
            )
        if aux_features.shape[0] != text_state.shape[0]:
            raise ValueError(
                "Batch mismatch for duration predictor: "
                f"text_state={tuple(text_state.shape)} aux_features={tuple(aux_features.shape)}"
            )
        text_state, text_mask = _safe_attention_mask(text_state, text_mask)
        aux_features = aux_features.to(device=text_state.device, dtype=text_state.dtype)

        if self.duration_architecture in {
            "token_sum_adarn_zero_no_aux",
            "token_sum_dual_adarn_zero_no_aux",
        }:
            if self.speaker_dim is None:
                raise RuntimeError("Token-sum duration architecture requires speaker modules.")
            if has_speaker is None:
                raise ValueError(
                    "has_speaker is required for speaker-conditioned duration prediction."
                )
            has_speaker = has_speaker.to(device=text_state.device, dtype=torch.bool)
            if has_speaker.ndim != 1 or has_speaker.shape[0] != text_state.shape[0]:
                raise ValueError(
                    f"has_speaker must have shape (B,), got {tuple(has_speaker.shape)}"
                )
            speaker_vec = self._speaker_vec(
                batch_size=text_state.shape[0],
                device=text_state.device,
                dtype=text_state.dtype,
                speaker_state=speaker_state,
                has_speaker=has_speaker,
            )
            caption_vec = None
            if self.duration_architecture == "token_sum_dual_adarn_zero_no_aux":
                if self.caption_dim is None:
                    raise RuntimeError(
                        "Dual token-sum duration architecture requires caption modules."
                    )
                if has_caption is None:
                    raise ValueError(
                        "has_caption is required for caption-conditioned duration prediction."
                    )
                has_caption = has_caption.to(device=text_state.device, dtype=torch.bool)
                if has_caption.ndim != 1 or has_caption.shape[0] != text_state.shape[0]:
                    raise ValueError(
                        f"has_caption must have shape (B,), got {tuple(has_caption.shape)}"
                    )
                caption_vec = self._caption_vec(
                    batch_size=text_state.shape[0],
                    device=text_state.device,
                    dtype=text_state.dtype,
                    caption_state=caption_state,
                    caption_mask=caption_mask,
                    has_caption=has_caption,
                )
            if (
                self.token_input_proj is None
                or self.token_blocks is None
                or self.token_out_norm is None
                or self.token_out_proj is None
            ):
                raise RuntimeError("Token-sum duration modules are missing.")
            h = self.token_input_proj(text_state)
            for block in self.token_blocks:
                h = block(h, cond=speaker_vec, caption_cond=caption_vec)
            token_logits = self.token_out_proj(self.token_out_norm(h)).squeeze(-1)
            token_frames = F.softplus(token_logits.float())
            total_frames = (token_frames * text_mask.to(dtype=token_frames.dtype)).sum(dim=1)
            return torch.log1p(total_frames.clamp_min(0.0))

        if self.text_pool is None:
            raise RuntimeError("Pooled duration modules are missing.")
        text_vec = self.text_pool(text_state, text_mask)
        if self.speaker_dim is None:
            x = torch.cat([text_vec, aux_features], dim=-1)
            h = self.input_proj(x)
            for block in self.blocks:
                h = block(h)
            return self.out_proj(self.out_norm(h)).squeeze(-1)

        if has_speaker is None:
            raise ValueError("has_speaker is required for speaker-conditioned duration prediction.")
        has_speaker = has_speaker.to(device=text_vec.device, dtype=torch.bool)
        if has_speaker.ndim != 1 or has_speaker.shape[0] != text_vec.shape[0]:
            raise ValueError(f"has_speaker must have shape (B,), got {tuple(has_speaker.shape)}")
        speaker_vec = self._speaker_vec(
            batch_size=text_vec.shape[0],
            device=text_vec.device,
            dtype=text_vec.dtype,
            speaker_state=speaker_state,
            has_speaker=has_speaker,
        )

        if self.speaker_fusion == "concat":
            x = torch.cat([text_vec, speaker_vec, aux_features], dim=-1)
            cond = None
        elif self.speaker_fusion == "adarn":
            if self.text_adarn_norm is None or self.text_adarn is None:
                raise RuntimeError("AdaRN duration speaker modules are missing.")
            scale, shift = self.text_adarn(speaker_vec).chunk(2, dim=-1)
            text_vec = (self.text_adarn_norm(text_vec) * (1.0 + scale)) + shift
            x = torch.cat([text_vec, aux_features], dim=-1)
            cond = None
        elif self.speaker_fusion == "adarn_zero":
            x = torch.cat([text_vec, aux_features], dim=-1)
            cond = speaker_vec
        elif self.speaker_fusion == "speaker_cross_attn":
            if self.speaker_cross_attn is None:
                raise RuntimeError("speaker_cross_attn duration module is missing.")
            speaker_context, speaker_context_mask = self._speaker_sequence(
                batch_size=text_vec.shape[0],
                device=text_vec.device,
                dtype=text_vec.dtype,
                speaker_state=speaker_state,
                speaker_mask=speaker_mask,
                has_speaker=has_speaker,
            )
            context_vec = self.speaker_cross_attn(
                query=text_vec,
                context=speaker_context,
                context_mask=speaker_context_mask,
            )
            x = torch.cat([text_vec, context_vec, aux_features], dim=-1)
            cond = None
        elif self.speaker_fusion == "text_cross_attn":
            if self.text_cross_attn is None:
                raise RuntimeError("text_cross_attn duration module is missing.")
            context_vec = self.text_cross_attn(
                query=speaker_vec,
                context=text_state,
                context_mask=text_mask,
            )
            x = torch.cat([context_vec, speaker_vec, aux_features], dim=-1)
            cond = None
        else:
            raise RuntimeError(f"Unsupported duration speaker fusion: {self.speaker_fusion!r}")

        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, cond=cond)
        return self.out_proj(self.out_norm(h)).squeeze(-1)


class TextToLatentRFDiT(nn.Module):
    """
    Text + reference-latent conditioned RF diffusion model over patched DACVAE latent sequences.

    Input x_t shape: (B, S, latent_dim * latent_patch_size)
    Output v_pred shape: same as input.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        pretrained_backbone_config: dict[str, object] | None = None,
        load_pretrained_backbone_weights: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.pretrained_text_backbone = None
        if cfg.use_pretrained_text_encoder:
            self.pretrained_text_backbone = PretrainedTextBackbone(
                cfg.text_tokenizer_repo,
                revision=cfg.text_encoder_revision,
                config_dict=pretrained_backbone_config,
                load_pretrained_weights=load_pretrained_backbone_weights,
            )
            self.text_encoder = PretrainedConditionProjector(
                self.pretrained_text_backbone.hidden_size,
                cfg.text_dim,
                projector_type=cfg.pretrained_projector_type,
                hidden_ratio=cfg.pretrained_projector_hidden_ratio,
                dropout=cfg.pretrained_projector_dropout,
                norm_eps=cfg.norm_eps,
            )
        else:
            self.text_encoder = TextEncoder(
                vocab_size=cfg.text_vocab_size,
                dim=cfg.text_dim,
                layers=cfg.text_layers,
                heads=cfg.text_heads,
                mlp_ratio=cfg.text_mlp_ratio_resolved,
                norm_eps=cfg.norm_eps,
                dropout=cfg.dropout,
            )
        self.caption_encoder = None
        self.caption_norm = None
        if cfg.use_caption_condition:
            if cfg.use_pretrained_text_encoder:
                if cfg.caption_tokenizer_repo_resolved != cfg.text_tokenizer_repo:
                    raise ValueError(
                        "A shared pretrained text/caption encoder requires identical tokenizer "
                        "repositories: "
                        f"text={cfg.text_tokenizer_repo!r}, "
                        f"caption={cfg.caption_tokenizer_repo_resolved!r}."
                    )
                self.caption_encoder = PretrainedConditionProjector(
                    self.pretrained_text_backbone.hidden_size,
                    cfg.caption_dim_resolved,
                    projector_type=cfg.pretrained_projector_type,
                    hidden_ratio=cfg.pretrained_projector_hidden_ratio,
                    dropout=cfg.pretrained_projector_dropout,
                    norm_eps=cfg.norm_eps,
                )
            else:
                self.caption_encoder = TextEncoder(
                    vocab_size=cfg.caption_vocab_size_resolved,
                    dim=cfg.caption_dim_resolved,
                    layers=cfg.caption_layers_resolved,
                    heads=cfg.caption_heads_resolved,
                    mlp_ratio=cfg.caption_mlp_ratio_resolved,
                    norm_eps=cfg.norm_eps,
                    dropout=cfg.dropout,
                )
            self.caption_norm = RMSNorm(cfg.caption_dim_resolved, eps=cfg.norm_eps)
        self.speaker_encoder = None
        if cfg.use_speaker_condition_resolved:
            self.speaker_encoder = ReferenceLatentEncoder(cfg)
        self.text_norm = RMSNorm(cfg.text_dim, eps=cfg.norm_eps)
        self.speaker_norm = None
        if cfg.use_speaker_condition_resolved:
            self.speaker_norm = RMSNorm(cfg.speaker_dim, eps=cfg.norm_eps)
        self.duration_predictor = None
        if cfg.use_duration_predictor:
            duration_speaker_dim = None
            if cfg.use_speaker_condition_resolved:
                duration_speaker_dim = int(cfg.speaker_dim)
            duration_caption_dim = None
            if cfg.use_caption_condition:
                duration_caption_dim = int(cfg.caption_dim_resolved)
            self.duration_predictor = DurationPredictor(
                text_dim=cfg.text_dim,
                aux_dim=cfg.duration_aux_dim,
                hidden_dim=cfg.duration_hidden_dim,
                layers=cfg.duration_layers,
                dropout=cfg.duration_dropout,
                speaker_dim=duration_speaker_dim,
                speaker_fusion=cfg.duration_speaker_fusion,
                caption_dim=duration_caption_dim,
                caption_fusion=cfg.duration_caption_fusion,
                caption_pooling=cfg.duration_caption_pooling,
                attention_heads=cfg.duration_attention_heads,
                norm_eps=cfg.norm_eps,
                architecture=cfg.duration_architecture,
                token_init_frames=cfg.duration_token_init_frames,
            )

        self.cond_module = nn.Sequential(
            nn.Linear(cfg.timestep_embed_dim, cfg.model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.model_dim, cfg.model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.model_dim, cfg.model_dim * 3, bias=False),
        )

        self.in_proj = nn.Linear(cfg.patched_latent_dim, cfg.model_dim)
        self.blocks = nn.ModuleList(DiffusionBlock(cfg) for _ in range(cfg.num_layers))
        self.gradient_checkpointing = False
        self.out_norm = RMSNorm(cfg.model_dim, eps=cfg.norm_eps)
        self.out_proj = nn.Linear(cfg.model_dim, cfg.patched_latent_dim)
        # Echo/JAX training initializes decoder out projection to zero for stable early training.
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        self.head_dim = cfg.model_dim // cfg.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("model head_dim must be even for RoPE")
        self.register_buffer(
            "_freqs_cis_cache", torch.empty(0, 0, dtype=torch.complex64), persistent=False
        )

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)
        if self.pretrained_text_backbone is not None:
            self.pretrained_text_backbone.set_gradient_checkpointing(enabled)

    def enable_speaker_inversion(
        self,
        *,
        num_tokens: int,
        init_std: float,
        init_embedding: torch.Tensor | None = None,
    ) -> SpeakerInversionEmbedding:
        if not self.cfg.use_speaker_condition_resolved:
            raise ValueError("Speaker inversion requires model speaker conditioning to be enabled.")
        module = SpeakerInversionEmbedding(
            num_tokens=int(num_tokens),
            speaker_dim=int(self.cfg.speaker_dim),
            init_std=float(init_std),
            init_embedding=init_embedding,
        )
        self.speaker_inversion = module
        return module

    def _rope_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        cache = self._freqs_cis_cache
        if cache.device != device or cache.shape[0] < seq_len:
            cache = precompute_freqs_cis(self.head_dim, seq_len).to(device)
            self._freqs_cis_cache = cache
        return cache[:seq_len]

    @staticmethod
    def _prepend_masked_mean_token(
        state: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepend one global summary token computed as masked mean over time.
        """
        mask_f = mask.unsqueeze(-1).to(dtype=state.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_token = (state * mask_f).sum(dim=1, keepdim=True) / denom
        has_any = mask.any(dim=1, keepdim=True)
        state = torch.cat([mean_token, state], dim=1)
        mask = torch.cat([has_any, mask], dim=1)
        return state, mask

    @staticmethod
    def _expand_speaker_condition_batch(
        state: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        batch_size: int,
        speaker_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim == 2:
            state = state.unsqueeze(0)
        if state.ndim != 3:
            raise ValueError(
                f"speaker_state must have shape (B,S,D) or (S,D), got {tuple(state.shape)}"
            )
        if int(state.shape[-1]) != int(speaker_dim):
            raise ValueError(
                f"speaker_state last dim must be {int(speaker_dim)}, got {int(state.shape[-1])}"
            )
        if state.shape[0] == 1 and batch_size != 1:
            state = state.expand(batch_size, -1, -1)
        elif int(state.shape[0]) != int(batch_size):
            raise ValueError(
                f"speaker_state batch mismatch: expected {int(batch_size)}, got {int(state.shape[0])}"
            )

        if mask is None:
            mask = torch.ones(state.shape[:2], dtype=torch.bool, device=state.device)
        else:
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.ndim != 2:
                raise ValueError(
                    f"speaker_mask must have shape (B,S) or (S,), got {tuple(mask.shape)}"
                )
            if mask.shape[0] == 1 and batch_size != 1:
                mask = mask.expand(batch_size, -1)
            elif int(mask.shape[0]) != int(batch_size):
                raise ValueError(
                    f"speaker_mask batch mismatch: expected {int(batch_size)}, got {int(mask.shape[0])}"
                )
            if int(mask.shape[1]) != int(state.shape[1]):
                raise ValueError(
                    "speaker_mask token mismatch: "
                    f"state={tuple(state.shape)} mask={tuple(mask.shape)}"
                )
            mask = mask.to(device=state.device, dtype=torch.bool)
        return state, mask

    def _apply_speaker_condition_dropout(
        self,
        *,
        speaker_state: torch.Tensor,
        speaker_mask: torch.Tensor,
        dropout_mask: torch.Tensor | None,
        uncond_state: torch.Tensor | None,
        uncond_mask: torch.Tensor | None,
        uncond_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dropout_mask is None:
            return speaker_state, speaker_mask
        dropout_mask = dropout_mask.to(device=speaker_state.device, dtype=torch.bool)
        if dropout_mask.ndim != 1 or dropout_mask.shape[0] != speaker_state.shape[0]:
            raise ValueError(
                "speaker_condition_dropout must have shape (B,), "
                f"got {tuple(dropout_mask.shape)} for speaker_state={tuple(speaker_state.shape)}"
            )
        mode = str(uncond_mode).strip().lower()
        if mode not in SPEAKER_INVERSION_UNCOND_MODES:
            raise ValueError(
                f"speaker_uncond_mode must be one of {sorted(SPEAKER_INVERSION_UNCOND_MODES)}, "
                f"got {uncond_mode!r}"
            )
        if mode == "noise":
            if uncond_state is None:
                scale = speaker_state.detach().std().clamp_min(1e-6)
                uncond_state = torch.randn_like(speaker_state) * scale
            if uncond_mask is None:
                uncond_mask = torch.ones_like(speaker_mask)
            uncond_state, uncond_mask = self._expand_speaker_condition_batch(
                uncond_state,
                uncond_mask,
                batch_size=speaker_state.shape[0],
                speaker_dim=self.cfg.speaker_dim,
            )
            uncond_state = uncond_state.to(device=speaker_state.device, dtype=speaker_state.dtype)
            uncond_mask = uncond_mask.to(device=speaker_state.device, dtype=torch.bool)
            speaker_state = torch.where(dropout_mask[:, None, None], uncond_state, speaker_state)
            speaker_mask = torch.where(dropout_mask[:, None], uncond_mask, speaker_mask)
            return speaker_state, speaker_mask

        speaker_mask = speaker_mask.clone()
        speaker_mask[dropout_mask] = False
        return speaker_state, speaker_mask

    def encode_conditions(
        self,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        ref_latent: torch.Tensor | None,
        ref_mask: torch.Tensor | None,
        caption_input_ids: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
        speaker_state_override: torch.Tensor | None = None,
        speaker_mask_override: torch.Tensor | None = None,
        speaker_uncond_mode: str = "mask",
        text_condition_dropout: torch.Tensor | None = None,
        speaker_condition_dropout: torch.Tensor | None = None,
        caption_condition_dropout: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if text_condition_dropout is not None:
            text_mask = text_mask.clone()
            text_mask[text_condition_dropout] = False
        if self.cfg.use_speaker_condition_resolved:
            speaker_inversion = getattr(self, "speaker_inversion", None)
            has_direct_speaker = speaker_state_override is not None or isinstance(
                speaker_inversion, SpeakerInversionEmbedding
            )
            if not has_direct_speaker and (
                self.speaker_encoder is None or self.speaker_norm is None
            ):
                raise RuntimeError(
                    "Speaker conditioning is enabled but speaker modules are missing."
                )
            if not has_direct_speaker and (ref_latent is None or ref_mask is None):
                raise ValueError(
                    "ref_latent and ref_mask are required when speaker conditioning is enabled."
                )
        elif speaker_state_override is not None:
            raise ValueError(
                "speaker_state_override was provided but speaker conditioning is disabled."
            )
        if self.cfg.use_caption_condition:
            if self.caption_encoder is None or self.caption_norm is None:
                raise RuntimeError(
                    "Caption conditioning is enabled but caption modules are missing."
                )
            if caption_input_ids is None or caption_mask is None:
                raise ValueError(
                    "caption_input_ids and caption_mask are required when caption conditioning is enabled."
                )
            if caption_condition_dropout is not None:
                caption_mask = caption_mask.clone()
                caption_mask[caption_condition_dropout] = False

        if self.pretrained_text_backbone is None:
            text_state = self.text_encoder(text_input_ids, text_mask)
        else:
            text_state = self.text_encoder(self.pretrained_text_backbone, text_input_ids, text_mask)
        text_state = self.text_norm(text_state)
        ref_state = None
        if self.cfg.use_speaker_condition_resolved:
            if speaker_state_override is not None:
                ref_state, ref_mask = self._expand_speaker_condition_batch(
                    speaker_state_override,
                    speaker_mask_override,
                    batch_size=text_input_ids.shape[0],
                    speaker_dim=self.cfg.speaker_dim,
                )
                ref_state = ref_state.to(device=text_state.device, dtype=text_state.dtype)
                ref_mask = ref_mask.to(device=text_state.device, dtype=torch.bool)
            else:
                speaker_inversion = getattr(self, "speaker_inversion", None)
                if isinstance(speaker_inversion, SpeakerInversionEmbedding):
                    ref_state, ref_mask = speaker_inversion(
                        batch_size=text_input_ids.shape[0],
                        device=text_state.device,
                        dtype=text_state.dtype,
                    )
                else:
                    ref_latent, ref_mask = patch_sequence_with_mask(
                        seq=ref_latent,
                        mask=ref_mask,
                        patch_size=self.cfg.speaker_patch_size,
                    )
                    ref_state = self.speaker_encoder(ref_latent, ref_mask)
                    ref_state = self.speaker_norm(ref_state)
                    ref_state, ref_mask = self._prepend_masked_mean_token(ref_state, ref_mask)
            ref_state, ref_mask = self._apply_speaker_condition_dropout(
                speaker_state=ref_state,
                speaker_mask=ref_mask,
                dropout_mask=speaker_condition_dropout,
                uncond_state=None,
                uncond_mask=None,
                uncond_mode=speaker_uncond_mode,
            )
        caption_state = None
        if self.cfg.use_caption_condition:
            if self.pretrained_text_backbone is None:
                caption_state = self.caption_encoder(caption_input_ids, caption_mask)
            else:
                caption_state = self.caption_encoder(
                    self.pretrained_text_backbone, caption_input_ids, caption_mask
                )
            caption_state = self.caption_norm(caption_state)
        return text_state, text_mask, ref_state, ref_mask, caption_state, caption_mask

    def forward_with_encoded_conditions(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        caption_state: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
        latent_mask: torch.Tensor | None = None,
        context_kv_cache: list[tuple[torch.Tensor, ...]] | None = None,
    ) -> torch.Tensor:
        t_embed = get_timestep_embedding(t, self.cfg.timestep_embed_dim).to(dtype=x_t.dtype)
        cond_embed = self.cond_module(t_embed)
        cond_embed = cond_embed[:, None, :]

        x = self.in_proj(x_t)
        freqs = self._rope_freqs(x.shape[1], x.device)
        use_checkpoint = self.gradient_checkpointing and self.training and context_kv_cache is None
        for i, block in enumerate(self.blocks):
            context_kv = context_kv_cache[i] if context_kv_cache is not None else None
            if use_checkpoint:
                x = _torch_checkpoint(
                    block,
                    x,
                    cond_embed,
                    text_state,
                    text_mask,
                    speaker_state,
                    speaker_mask,
                    caption_state,
                    caption_mask,
                    freqs,
                    latent_mask,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x=x,
                    cond_embed=cond_embed,
                    text_state=text_state,
                    text_mask=text_mask,
                    speaker_state=speaker_state,
                    speaker_mask=speaker_mask,
                    caption_state=caption_state,
                    caption_mask=caption_mask,
                    freqs_cis=freqs,
                    self_mask=latent_mask,
                    context_kv=context_kv,
                )

        x = self.out_norm(x)
        x = self.out_proj(x)
        return x.to(dtype=x_t.dtype)

    def forward(
        self,
        x_t: torch.Tensor | None,
        t: torch.Tensor | None,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        ref_latent: torch.Tensor | None,
        ref_mask: torch.Tensor | None,
        caption_input_ids: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
        latent_mask: torch.Tensor | None = None,
        text_condition_dropout: torch.Tensor | None = None,
        speaker_condition_dropout: torch.Tensor | None = None,
        caption_condition_dropout: torch.Tensor | None = None,
        duration_features: torch.Tensor | None = None,
        duration_has_speaker: torch.Tensor | None = None,
        duration_has_caption: torch.Tensor | None = None,
        duration_only: bool = False,
        duration_backprop_to_condition: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if duration_features is not None:
            (
                text_state,
                text_mask_full,
                speaker_state,
                speaker_mask_full,
                caption_state,
                caption_mask_full,
            ) = self.encode_conditions(
                text_input_ids=text_input_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                caption_input_ids=caption_input_ids,
                caption_mask=caption_mask,
            )
            if duration_only:
                return self.predict_duration_log_frames(
                    text_state=text_state,
                    text_mask=text_mask_full,
                    speaker_state=speaker_state,
                    speaker_mask=speaker_mask_full,
                    caption_state=caption_state,
                    caption_mask=caption_mask_full,
                    duration_features=duration_features,
                    has_speaker=duration_has_speaker,
                    has_caption=duration_has_caption,
                    detach_condition=True,
                )

            if x_t is None or t is None:
                raise ValueError("x_t and t are required unless duration_only=True.")
            text_mask_dit = text_mask_full
            speaker_state_dit = speaker_state
            speaker_mask_dit = speaker_mask_full
            caption_mask_dit = caption_mask_full
            if text_condition_dropout is not None:
                text_mask_dit = text_mask_dit.clone()
                text_mask_dit[text_condition_dropout] = False
            if (
                speaker_condition_dropout is not None
                and speaker_state_dit is not None
                and speaker_mask_dit is not None
            ):
                speaker_state_dit, speaker_mask_dit = self._apply_speaker_condition_dropout(
                    speaker_state=speaker_state_dit,
                    speaker_mask=speaker_mask_dit,
                    dropout_mask=speaker_condition_dropout,
                    uncond_state=None,
                    uncond_mask=None,
                    uncond_mode="mask",
                )
            if caption_condition_dropout is not None and caption_mask_dit is not None:
                caption_mask_dit = caption_mask_dit.clone()
                caption_mask_dit[caption_condition_dropout] = False

            v_pred = self.forward_with_encoded_conditions(
                x_t=x_t,
                t=t,
                text_state=text_state,
                text_mask=text_mask_dit,
                speaker_state=speaker_state_dit,
                speaker_mask=speaker_mask_dit,
                caption_state=caption_state,
                caption_mask=caption_mask_dit,
                latent_mask=latent_mask,
            )
            duration_pred = self.predict_duration_log_frames(
                text_state=text_state,
                text_mask=text_mask_full,
                speaker_state=speaker_state,
                speaker_mask=speaker_mask_full,
                caption_state=caption_state,
                caption_mask=caption_mask_full,
                duration_features=duration_features,
                has_speaker=duration_has_speaker,
                has_caption=duration_has_caption,
                detach_condition=not duration_backprop_to_condition,
            )
            return v_pred, duration_pred

        if duration_only:
            raise ValueError("duration_features is required when duration_only=True.")
        if x_t is None or t is None:
            raise ValueError("x_t and t are required for RF forward.")

        (
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
        ) = self.encode_conditions(
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_input_ids,
            caption_mask=caption_mask,
            text_condition_dropout=text_condition_dropout,
            speaker_condition_dropout=speaker_condition_dropout,
            caption_condition_dropout=caption_condition_dropout,
        )
        return self.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=caption_state,
            caption_mask=caption_mask,
            latent_mask=latent_mask,
        )

    def build_context_kv_cache(
        self,
        text_state: torch.Tensor,
        speaker_state: torch.Tensor | None,
        caption_state: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, ...]]:
        """
        Build per-layer projected conditioning KV tensors for faster repeated sampling steps.
        """
        return [
            block.attention.project_context_kv(
                text_context=text_state,
                speaker_context=speaker_state,
                caption_context=caption_state,
            )
            for block in self.blocks
        ]

    @staticmethod
    def masked_mean(state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=state.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (state * mask_f).sum(dim=1) / denom

    def predict_duration_log_frames(
        self,
        *,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        duration_features: torch.Tensor,
        has_speaker: torch.Tensor | None,
        caption_state: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
        has_caption: torch.Tensor | None = None,
        detach_condition: bool = True,
    ) -> torch.Tensor:
        if self.duration_predictor is None:
            raise RuntimeError("Duration predictor is disabled for this model.")
        if duration_features.ndim != 2:
            raise ValueError(
                f"duration_features must have shape (B, D), got {tuple(duration_features.shape)}"
            )
        if duration_features.shape[1] != self.cfg.duration_aux_dim:
            raise ValueError(
                "duration_features dim mismatch: "
                f"expected {self.cfg.duration_aux_dim}, got {duration_features.shape[1]}"
            )

        duration_text_state = text_state
        duration_speaker_state = speaker_state
        duration_caption_state = caption_state
        if detach_condition:
            duration_text_state = text_state.detach()
            if speaker_state is not None:
                duration_speaker_state = speaker_state.detach()
            if caption_state is not None:
                duration_caption_state = caption_state.detach()
        pred = self.duration_predictor(
            duration_text_state,
            text_mask=text_mask,
            aux_features=duration_features,
            speaker_state=duration_speaker_state,
            speaker_mask=speaker_mask,
            has_speaker=has_speaker,
            caption_state=duration_caption_state,
            caption_mask=caption_mask,
            has_caption=has_caption,
        )
        return pred.float()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def as_dict(self) -> dict:
        return asdict(self.cfg)
