# Author: wanren
# Date: 2025/9/15
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math




# 保留原有代码的其他类和函数(SimpleMLPAdaLN, TimestepEmbedder, ResBlock, FinalLayer等)
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self._freqs = None
        self._dim_for_freqs = None
        self._max_period_for_freqs = None

    def freqs(self, device, dim, max_period=10000):
        if self._freqs is None or max_period != self._max_period_for_freqs or dim != self._dim_for_freqs:
            # print("Building freqs")
            self._dim_for_freqs = dim
            self._max_period_for_freqs = max_period

            half = dim // 2
            self._freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
            ).to(device=device)
        return self._freqs

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        # freqs = torch.exp(
        #     -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        # ).to(device=t.device)
        args = t[:, None].float() * self.freqs(t.device, dim, max_period)[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.drop = nn.Dropout(drop)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.drop(hidden)
        return self.w3(hidden)


class SwiGLUFFNFused(SwiGLUFFN):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            drop=drop,
            bias=bias,
        )


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
            self,
            channels,
            mlp_mult=1.,
            drop=0.,
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        # print(f"hidden feature for ResBlock: {channels * mlp_mult}")
        self.mlp = SwiGLUFFNFused(
            in_features=channels,
            hidden_features=int(channels * mlp_mult),
            drop=drop
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
            self,
            in_channels=64,
            model_channels=3072,
            out_channels=64,
            z_channels=896,
            num_res_blocks=12,
            mlp_mult=4,
            cfg_prob=0.1,
            grad_checkpointing=False,
            is_mf=False,
            drop=0.,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.cfg_prob = cfg_prob
        self.grad_checkpointing = grad_checkpointing
        self.is_mf = is_mf

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels*2 if is_mf else model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        self.c_proj = nn.Sequential(
            nn.Linear(model_channels*2, model_channels, bias=True),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels, bias=True),
            nn.SiLU()) if is_mf else None


        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
                mlp_mult,
                drop=drop,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # def forward(self, x, t, c):
    #     """
    #     Apply the model to an input batch.
    #     :param x: an [N x C] Tensor of inputs.
    #     :param t: a 1-D batch of timesteps.
    #     :param c: conditioning from AR transformer.
    #     :return: an [N x C] Tensor of outputs.
    #     """
    #     # import ipdb;ipdb.set_trace()
    #     x = self.input_proj(x)
    #     t = self.time_embed(t)
    #     c = self.cond_embed(c)

    #     y = t + c

    #     if self.grad_checkpointing and not torch.jit.is_scripting():
    #         for block in self.res_blocks:
    #             x = checkpoint(block, x, y)
    #     else:
    #         for block in self.res_blocks:
    #             x = block(x, y)

    #     return self.final_layer(x, y)

    # 以下是SimpleMLPAdaLN的forward方法，需要适应时间步t是标量而非整数
    def forward(self, x, t, c, latent_history=None, mask=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps (0 to 1).
        :param c: conditioning from AR transformer.
        :return: velocity field prediction
        """
        x = self.input_proj(x)
        x_his = self.input_proj(latent_history)

        if self.training and self.cfg_prob > 0:
            drop_cfg = torch.rand(c.shape[0]) >= self.cfg_prob
            drop_cfg = drop_cfg[:, None, None].to(dtype=c.dtype, device=c.device)
            c = c * drop_cfg
        c = self.cond_embed(c)

        if self.is_mf:
            t = torch.cat([self.time_embed(t[0] * 1000), self.time_embed(t[1] * 1000)], dim=-1)[:, None]
            y = self.c_proj(c + t)
        else:
            t = self.time_embed(t * 1000).unsqueeze(1)
            y = c + t

        y = y + x_his

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y, use_reentrant=True)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, latent_history, spk_emb=None, **kwargs):
        x = torch.cat([x, x], dim=0)
        latent_history = torch.cat([latent_history, latent_history], dim=0)
        fake_latent = torch.zeros(c.shape).to(c.device)
        c = torch.cat([c, fake_latent], dim=0)
        if self.is_mf:
            if t[0].ndim == 0:
                t = (t[0].repeat(x.shape[0]), t[1].repeat(x.shape[0]))
        else:
            if t.ndim == 0:
                t = t.repeat(x.shape[0])
        if spk_emb is not None:
            spk_emb = torch.cat([spk_emb, spk_emb], dim=0)
        model_out = self.forward(x, t, c, latent_history, spk_emb)
        return model_out[:, -x.shape[1]:, :]


if __name__ == "__main__":
    # a = RectifiedFlowLoss(
    #     target_channels=32,
    #     z_channels=3584,
    #     depth=12,
    #     width=3072,
    #     num_sampling_steps="16",
    #     mlp_mult=4.,
    #     grad_checkpointing=False)
    # num = 0
    # for k, v in a.named_parameters():
    #     num += v.numel()
    # print(num)
    a = TimestepEmbedder(1024)
    print("running b")
    b = a(torch.rand(8, device="cpu"))
    print("running c")
    c = a(torch.rand(8, device="cpu"))
    print("running d")
    d = a(torch.rand(8, device="cpu"))
    print("ok")
