import math

import torch
import torch.nn as nn

from ...modules.backbone.layers import Mlp, MultiHeadAttention


def modulate(x, shift, scale, **_kwargs):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-5)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x, c, **_kwargs):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        ffn: nn.Module,
        hidden_size: int = 1024,
        modulation: bool = False,
        eps: float = 1e-5,
        **_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=not modulation, eps=eps
        )
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=not modulation, eps=eps
        )
        self.attn = attention
        self.ffn = ffn
        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )

    def forward(self, x, condition=None, mask=None, **kwargs):
        if condition is None:
            assert not self.modulation, (
                "Without global condition, must set modulation to False"
            )
        else:
            assert self.modulation, "With global condition, must set modulation to True"
            shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
                self.adaLN_modulation(condition).chunk(6, dim=1)
            )

        if condition is not None:
            pack_indices = kwargs.get("pack_indices")
            if pack_indices is not None:
                gate_attn = gate_attn[pack_indices]
                gate_ffn = gate_ffn[pack_indices]
            else:
                gate_attn = gate_attn.unsqueeze(1)
                gate_ffn = gate_ffn.unsqueeze(1)

        if condition is not None:
            x = x + gate_attn * self.attn(
                modulate(self.norm1(x), shift_attn, scale_attn, **kwargs),
                mask=mask,
                **kwargs,
            )
        else:
            x = x + self.attn(self.norm1(x), mask=mask, **kwargs)

        if condition is not None:
            x = x + gate_ffn * self.ffn(
                modulate(self.norm2(x), shift_ffn, scale_ffn, **kwargs)
            )
        else:
            x = x + self.ffn(self.norm2(x), mask=mask)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        transformer_config,
        *,
        mode: str = "flow_matching",
    ):
        super().__init__()
        if mode not in {"flow_matching", "meanflow"}:
            raise ValueError(
                f"DiT mode must be 'flow_matching' or 'meanflow', got {mode!r}."
            )

        transformer_kwargs = transformer_config.to_dict()
        model_dim = transformer_config.hidden_size
        self.mode = mode
        self.num_layers = transformer_config.num_layers

        self.input_layer = nn.Linear(in_dim, model_dim)
        self.time_embedder = TimestepEmbedder(model_dim)
        if mode == "meanflow":
            self.duration_embedder = TimestepEmbedder(model_dim)

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            attn_block = MultiHeadAttention(**transformer_kwargs, name=f"layer_{i}")
            ffn_block = Mlp(
                act_layer=lambda: nn.GELU(approximate="tanh"), **transformer_kwargs
            )
            self.blocks.append(
                DiTBlock(attention=attn_block, ffn=ffn_block, **transformer_kwargs)
            )

        self.output_layer = FinalLayer(model_dim, out_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            if hasattr(block, "adaLN_modulation"):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.output_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.output_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.output_layer.linear.weight, 0)
        nn.init.constant_(self.output_layer.linear.bias, 0)

    def forward(
        self,
        x,
        timesteps,
        duration: torch.Tensor | None = None,
        mask=None,
        attn_mask=None,
        g_cond: torch.Tensor | None = None,
        **kwargs,
    ):
        t = self.time_embedder(timesteps)
        c = t
        duration_embedder = getattr(self, "duration_embedder", None)
        if duration_embedder is not None and duration is not None:
            c = c + duration_embedder(duration)
        if g_cond is not None:
            c = c + g_cond

        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x, c, mask=attn_mask, **kwargs)
        return self.output_layer(x, c, **kwargs)
