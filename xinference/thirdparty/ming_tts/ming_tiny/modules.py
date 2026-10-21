from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb


_FLASH_ATTN_OPS = None


def _load_flash_attn_ops():
    """Load FlashAttention only when a model explicitly selects that backend.

    The production Ming-3.0-tts-tiny contract uses the PyTorch SDPA backend.  Some
    evaluation images nevertheless contain a FlashAttention wheel compiled
    against a different PyTorch ABI.  Importing that optional wheel while
    importing this module would make the SDPA path fail before model loading.
    """
    global _FLASH_ATTN_OPS
    if _FLASH_ATTN_OPS is None:
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            from flash_attn.bert_padding import pad_input, unpad_input
        except (ImportError, OSError) as error:
            raise ImportError(
                "FlashAttention backend was requested, but flash_attn could "
                "not be imported with the current PyTorch runtime."
            ) from error
        _FLASH_ATTN_OPS = (
            pad_input,
            unpad_input,
            flash_attn_varlen_func,
            flash_attn_func,
        )
    return _FLASH_ATTN_OPS


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x):
        if self.native_rms_norm:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        else:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight

        return x



class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)




class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        qk_norm: Optional[str] = None,
        pe_attn_head: int | None = None,  # number of attention head to apply rope, None for all
        attn_backend: str = "torch",  # "torch" or "flash_attn"
        attn_mask_enabled: bool = True,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")


        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout


        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")


        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if attn_backend == "flash_attn":
            _load_flash_attn_ops()
        elif attn_backend != "torch":
            raise ValueError(
                f"Unsupported attention backend: {attn_backend!r}"
            )

        self.pe_attn_head = pe_attn_head
        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled



    def forward(
        self,
        x: float,  # noised input x
        mask=None,
        rope=None,  # rotary position embedding for x
    ) -> torch.Tensor:

        batch_size = x.shape[0]

        # `sample` projections
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # qk norm
        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(query[:, :pn, :, :], freqs, q_xpos_scale)
                key[:, :pn, :, :] = apply_rotary_pos_emb(key[:, :pn, :, :], freqs, k_xpos_scale)
            else:
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        if self.attn_backend == "torch":
            # mask. e.g. inference got a batch with different target durations, mask out the padding
            if self.attn_mask_enabled and mask is not None:
                valid_sample_indices = mask.any(dim=1)
                final_output = torch.zeros_like(query).to(query.device)

                attn_mask = mask[valid_sample_indices]
                query = query[valid_sample_indices]
                key = key[valid_sample_indices]
                value = value[valid_sample_indices]
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
                attn_mask = attn_mask.expand(valid_sample_indices.sum().item(), self.heads, query.shape[-2], key.shape[-2])
            else:
                attn_mask = None

            x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            if self.attn_mask_enabled and mask is not None:
                final_output[valid_sample_indices] = x
                x = final_output

            x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)

        elif self.attn_backend == "flash_attn":
            (
                pad_input,
                unpad_input,
                flash_attn_varlen_func,
                flash_attn_func,
            ) = _load_flash_attn_ops()
            query = query.transpose(1, 2)  # [b, h, n, d] -> [b, n, h, d]
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if self.attn_mask_enabled and mask is not None:
                query, indices, q_cu_seqlens, q_max_seqlen_in_batch = unpad_input(query, mask)
                key, _, k_cu_seqlens, k_max_seqlen_in_batch = unpad_input(key, mask)
                value, _, _, _ = unpad_input(value, mask)
                x = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    q_cu_seqlens,
                    k_cu_seqlens,
                    q_max_seqlen_in_batch,
                    k_max_seqlen_in_batch,
                )
                x = pad_input(x, indices, batch_size, q_max_seqlen_in_batch)
                x = x.reshape(batch_size, -1, self.heads * head_dim)
            else:
                x = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
                x = x.reshape(batch_size, -1, self.heads * head_dim)

        x = x.to(query.dtype)

        # linear proj
        x = self.to_out[0](x)
        # dropout
        x = self.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x



class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="flash_attn",  # "torch" or "flash_attn"
        attn_mask_enabled=True,
        **kwargs
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            pe_attn_head=pe_attn_head,
            attn_backend=attn_backend,
            attn_mask_enabled=attn_mask_enabled,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate="tanh")

    def forward(self, x, mask, rope):
        x = x + self.attn(self.norm1(x), mask=mask, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)


    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x
