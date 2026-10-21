from typing import Iterable, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.cache_utils import DynamicCache

from .rope import RotaryPositionalEmbeddings

try:
    from flash_attn import flash_attn_func as _flash_attn_func
except (ImportError, RuntimeError, OSError):
    _flash_attn_func = None

# flash-attn ships CUDA kernels only, and only for half precision. Being
# installed is therefore not enough to be usable: dispatching on the import
# alone raises NotImplementedError on CPU tensors and asserts on fp32.
_FLASH_DTYPES = (torch.float16, torch.bfloat16)


def _flash_attn_usable(q: Tensor) -> bool:
    return (
        _flash_attn_func is not None
        and q.is_cuda
        and q.dtype in _FLASH_DTYPES
    )


def causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Causal attention over ``[B, H, T, D]`` inputs, returning ``[B, T, H*D]``."""
    if _flash_attn_usable(q):
        a = _flash_attn_func(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
            causal=True,
        )
        return a.flatten(start_dim=2)

    # flash-attn anchors the causal mask to the bottom right of the score
    # matrix, which is what a populated KV cache needs. SDPA's ``is_causal``
    # anchors to the top left and would hide the entire cache once the query
    # is a single step, so the mask is built explicitly.
    q_len, k_len = q.size(-2), k.size(-2)
    mask = torch.ones(q_len, k_len, dtype=torch.bool, device=q.device).tril(
        diagonal=k_len - q_len
    )
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return a.permute(0, 2, 1, 3).flatten(start_dim=2)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class MultiHeadAttention(nn.Module):

    def __init__(self, n_state: int, n_head: int, layer_idx: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.layer_idx = layer_idx
        self.rotary_embed = RotaryPositionalEmbeddings(dim=n_state // n_head)

    def forward(
        self,
        x: Tensor,
        past_key_values=None
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk, past_key_values = self.qkv_attention(q, k, v, past_key_values=past_key_values)
        return self.out(wv), qk, past_key_values

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, past_key_values=None
    ):
        q = q.view(*q.shape[:2], self.n_head, -1)  # [B, T, nhead, dm]
        k = k.view(*k.shape[:2], self.n_head, -1)  # [B, T, nhead, dm]
        v = v.view(*v.shape[:2], self.n_head, -1)  # [B, T, nhead, dm]

        if past_key_values is not None:
            past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + q.size(1), device=q.device
            )
            cache_position = cache_position.unsqueeze(0)
        else:
            cache_position = None

        q = self.rotary_embed(q, input_pos=cache_position)
        k = self.rotary_embed(k, input_pos=cache_position)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx, {"cache_position": cache_position})

        out = causal_attention(q, k, v)
        qk = None

        return out, qk, past_key_values


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, layer_idx: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head, layer_idx)
        self.attn_ln = LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
        self.layer_idx = layer_idx

    def forward(
        self,
        x: Tensor,
        past_key_values=None
    ):
        attn_out, _, past_key_values = self.attn(self.attn_ln(x), past_key_values=past_key_values)
        x = x + attn_out
        x = x + self.mlp(self.mlp_ln(x))
        return x, past_key_values


class WhisperAudioEncoder(nn.Module):
    def __init__(self, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, layer_idx=i) for i in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, whipser_feats: Tensor, use_cache=False, past_key_values=None, **kwargs):
        if past_key_values is None and use_cache:
            past_key_values = DynamicCache()

        x = whipser_feats

        for block in self.blocks:
            x, past_key_values = block(x, past_key_values=past_key_values)

        x = self.ln_post(x)

        return x, past_key_values

    @classmethod
    def from_pretrained(cls, dims):
        audio_encoder = cls(
            dims['n_state'],
            dims['n_head'],
            dims['n_layer'],
        )

        audio_encoder.audio_emb_dim = dims['n_state']
        return audio_encoder
