# Copyright © 2024 Apple Inc.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

_SHARED_REPLACEMENT_PATTERNS = [
    (".block.", ".layers."),
    (".k.", ".key_proj."),
    (".o.", ".out_proj."),
    (".q.", ".query_proj."),
    (".v.", ".value_proj."),
    ("shared.", "wte."),
    ("lm_head.", "lm_head.linear."),
    (".layer.0.layer_norm.", ".ln1."),
    (".layer.1.layer_norm.", ".ln2."),
    (".layer.2.layer_norm.", ".ln3."),
    (".final_layer_norm.", ".ln."),
    (
        "layers.0.layer.0.SelfAttention.relative_attention_bias.",
        "relative_attention_bias.embeddings.",
    ),
]

_ENCODER_REPLACEMENT_PATTERNS = [
    (".layer.0.SelfAttention.", ".attention."),
    (".layer.1.DenseReluDense.", ".dense."),
]


@dataclass
class T5Config:
    vocab_size: int
    num_layers: int
    num_heads: int
    relative_attention_num_buckets: int
    d_kv: int
    d_model: int
    feed_forward_proj: str
    tie_word_embeddings: bool

    d_ff: Optional[int] = None
    num_decoder_layers: Optional[int] = None
    relative_attention_max_distance: int = 128
    layer_norm_epsilon: float = 1e-6

    @classmethod
    def from_dict(cls, config):
        return cls(
            vocab_size=config["vocab_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            relative_attention_num_buckets=config["relative_attention_num_buckets"],
            d_kv=config["d_kv"],
            d_model=config["d_model"],
            feed_forward_proj=config["feed_forward_proj"],
            tie_word_embeddings=config["tie_word_embeddings"],
            d_ff=config.get("d_ff", 4 * config["d_model"]),
            num_decoder_layers=config.get("num_decoder_layers", config["num_layers"]),
            relative_attention_max_distance=config.get(
                "relative_attention_max_distance", 128
            ),
            layer_norm_epsilon=config.get("layer_norm_epsilon", 1e-6),
        )


class RelativePositionBias(nn.Module):
    def __init__(self, config: T5Config, bidirectional: bool):
        self.bidirectional = bidirectional
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.n_heads = config.num_heads
        self.embeddings = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(rpos, bidirectional, num_buckets, max_distance):
        num_buckets = num_buckets // 2 if bidirectional else num_buckets
        max_exact = num_buckets // 2

        abspos = rpos.abs()
        is_small = abspos < max_exact

        scale = (num_buckets - max_exact) / math.log(max_distance / max_exact)
        buckets_large = (mx.log(abspos / max_exact) * scale).astype(mx.int16)
        buckets_large = mx.minimum(max_exact + buckets_large, num_buckets - 1)

        buckets = mx.where(is_small, abspos, buckets_large)
        if bidirectional:
            buckets = buckets + (rpos > 0) * num_buckets
        else:
            buckets = buckets * (rpos < 0)

        return buckets

    def __call__(self, query_length: int, key_length: int, offset: int = 0):
        """Compute binned relative position bias"""
        context_position = mx.arange(offset, query_length)[:, None]
        memory_position = mx.arange(key_length)[None, :]

        # shape (query_length, key_length)
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )

        # shape (query_length, key_length, num_heads)
        values = self.embeddings(relative_position_bucket)

        # shape (num_heads, query_length, key_length)
        return values.transpose(2, 0, 1)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        inner_dim = config.d_kv * config.num_heads
        self.num_heads = config.num_heads
        self.query_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.key_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.value_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, config.d_model, bias=False)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> [mx.array, Tuple[mx.array, mx.array]]:
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=3)
            values = mx.concatenate([value_cache, values], axis=2)

        values_hat = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=1.0, mask=mask.astype(queries.dtype)
        )
        values_hat = values_hat.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat), (keys, values)


class DenseActivation(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        mlp_dims = config.d_ff or config.d_model * 4
        self.gated = config.feed_forward_proj.startswith("gated")
        if self.gated:
            self.wi_0 = nn.Linear(config.d_model, mlp_dims, bias=False)
            self.wi_1 = nn.Linear(config.d_model, mlp_dims, bias=False)
        else:
            self.wi = nn.Linear(config.d_model, mlp_dims, bias=False)
        self.wo = nn.Linear(mlp_dims, config.d_model, bias=False)
        activation = config.feed_forward_proj.removeprefix("gated-")
        if activation == "relu":
            self.act = nn.relu
        elif activation == "gelu":
            self.act = nn.gelu
        elif activation == "silu":
            self.act = nn.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def __call__(self, x):
        if self.gated:
            hidden_act = self.act(self.wi_0(x))
            hidden_linear = self.wi_1(x)
            x = hidden_act * hidden_linear
        else:
            x = self.act(self.wi(x))
        return self.wo(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ln1 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dense = DenseActivation(config)

    def __call__(self, x, mask):
        y = self.ln1(x)
        y, _ = self.attention(y, y, y, mask=mask)
        x = x + y

        y = self.ln2(x)
        y = self.dense(y)
        return x + y


class TransformerEncoder(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(config) for i in range(config.num_layers)
        ]
        self.ln = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=True)

    def __call__(self, x: mx.array):
        pos_bias = self.relative_attention_bias(x.shape[1], x.shape[1])
        pos_bias = pos_bias.astype(x.dtype)
        for layer in self.layers:
            x = layer(x, mask=pos_bias)
        return self.ln(x)


class T5Encoder(nn.Module):
    def __init__(self, config: T5Config):
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = TransformerEncoder(config)

    def sanitize(self, weights):
        new_weights = {}
        for k, w in weights.items():
            for old, new in _SHARED_REPLACEMENT_PATTERNS:
                k = k.replace(old, new)
            if k.startswith("encoder."):
                for old, new in _ENCODER_REPLACEMENT_PATTERNS:
                    k = k.replace(old, new)
            new_weights[k] = w
        return new_weights

    def __call__(self, inputs: mx.array):
        return self.encoder(self.wte(inputs))
