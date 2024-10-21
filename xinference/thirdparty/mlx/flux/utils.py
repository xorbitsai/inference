# Copyright © 2024 Apple Inc.

import json
import os
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from .autoencoder import AutoEncoder, AutoEncoderParams
from .clip import CLIPTextModel, CLIPTextModelConfig
from .model import Flux, FluxParams
from .t5 import T5Config, T5Encoder
from .tokenizers import CLIPTokenizer, T5Tokenizer


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: Optional[str]
    ae_path: Optional[str]
    repo_id: Optional[str]
    repo_flow: Optional[str]
    repo_ae: Optional[str]


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def load_flow_model(name: str, ckpt_path: str):
    # Make the model
    model = Flux(configs[name].params)

    # Load the checkpoint if needed
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, configs[name].repo_flow)
    weights = mx.load(ckpt_path)
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))

    return model


def load_ae(name: str, ckpt_path: str):
    # Make the autoencoder
    ae = AutoEncoder(configs[name].ae_params)

    # Load the checkpoint if needed
    ckpt_path = os.path.join(ckpt_path, "ae.safetensors")
    weights = mx.load(ckpt_path)
    weights = ae.sanitize(weights)
    ae.load_weights(list(weights.items()))

    return ae


def load_clip(name: str, ckpt_path: str):
    config_path = os.path.join(ckpt_path, "text_encoder/config.json")
    with open(config_path) as f:
        config = CLIPTextModelConfig.from_dict(json.load(f))

    # Make the clip text encoder
    clip = CLIPTextModel(config)

    ckpt_path = os.path.join(ckpt_path, "text_encoder/model.safetensors")
    weights = mx.load(ckpt_path)
    weights = clip.sanitize(weights)
    clip.load_weights(list(weights.items()))

    return clip


def load_t5(name: str, ckpt_path: str):
    config_path = os.path.join(ckpt_path, "text_encoder_2/config.json")
    with open(config_path) as f:
        config = T5Config.from_dict(json.load(f))

    # Make the T5 model
    t5 = T5Encoder(config)

    model_index = os.path.join(ckpt_path, "text_encoder_2/model.safetensors.index.json")
    weight_files = set()
    with open(model_index) as f:
        for _, w in json.load(f)["weight_map"].items():
            weight_files.add(w)
    weights = {}
    for w in weight_files:
        w = f"text_encoder_2/{w}"
        w = os.path.join(ckpt_path, w)
        weights.update(mx.load(w))
    weights = t5.sanitize(weights)
    t5.load_weights(list(weights.items()))

    return t5


def load_clip_tokenizer(name: str, ckpt_path: str):
    vocab_file = os.path.join(ckpt_path, "tokenizer/vocab.json")
    with open(vocab_file, encoding="utf-8") as f:
        vocab = json.load(f)

    merges_file = os.path.join(ckpt_path, "tokenizer/merges.txt")
    with open(merges_file, encoding="utf-8") as f:
        bpe_merges = f.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
    bpe_merges = [tuple(m.split()) for m in bpe_merges]
    bpe_ranks = dict(map(reversed, enumerate(bpe_merges)))

    return CLIPTokenizer(bpe_ranks, vocab, max_length=77)


def load_t5_tokenizer(name: str, ckpt_path: str, pad: bool = True):
    model_file = os.path.join(ckpt_path, "tokenizer_2/spiece.model")
    return T5Tokenizer(model_file, 256 if "schnell" in name else 512)
