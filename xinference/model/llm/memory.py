# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE:
#
#   The algorithm is ported from https://github.com/RahulSChand/gpu_poor
#
#   Improvement:
#
#      The original js code only calculate kv_cache_dtype by float32, instead of most case we run model with float16.
#
#   Known Issue:
#
#       * On vllm, some MHA model use smaller memory than calculation (qwen1.5-7B-chat-gptq-int4,
#       qwen1.5-14B-chat-gptq-int4 with large activation_mem).
#
#       * On vllm, gemma-it-7B pytorch format model use larger gpu mem than calculation

import json
import math
from dataclasses import dataclass
from logging import getLogger
from math import ceil
from typing import List, Optional, Union

from .llm_family import convert_model_size_to_float
from .model_metadata import ModelMetadata, VisionMetadata

logger = getLogger(__name__)


def unsupported_memory_reason(metadata: ModelMetadata) -> Optional[str]:
    """Keep this estimator's support policy separate from model facts."""
    if metadata.architecture_type not in (
        None,
        "dense",
        "moe",
        "multimodal",
        "hybrid_attention",
    ):
        return metadata.architecture_type
    if metadata.architecture_type == "moe" and metadata.num_experts is None:
        return "incomplete_expert_layout"
    types = metadata.attention_layers()
    if types is not None:
        if len(types) != metadata.num_hidden_layers or any(
            t not in ("full_attention", "sliding_attention", "linear_attention")
            for t in types
        ):
            return "unknown_attention_layout"
        if "linear_attention" in types and any(
            getattr(metadata, name) is None
            for name in (
                "linear_num_key_heads",
                "linear_num_value_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
            )
        ):
            return "incomplete_linear_attention"
    elif metadata.architecture_type in ("multimodal", "hybrid_attention"):
        return "unknown_attention_layout"
    return None


@dataclass
class ModelLayersInfo:
    vocab_size: int
    heads: int  # num_attention_heads, num_heads or n_head
    hidden_dim: int  # hidden_size, d_model, or n_embd
    inter_dim: int  # intermediate_size, n_inner or d_ff
    num_layers: int  # num_layers, num_hidden_layers or n_layer
    kv_heads: Optional[int] = None  # defaults to attention heads (MHA)
    head_dim: Optional[int] = None  # explicit config value takes precedence
    layer_types: Optional[List[str]] = None
    linear_num_key_heads: Optional[int] = None
    linear_num_value_heads: Optional[int] = None
    linear_key_head_dim: Optional[int] = None
    linear_value_head_dim: Optional[int] = None
    linear_conv_kernel_dim: Optional[int] = None
    active_experts: int = 1
    shared_inter_dim: int = 0
    vision: Optional[VisionMetadata] = None
    weights_bytes: Optional[int] = None
    architecture_type: Optional[str] = None

    @classmethod
    def from_metadata(cls, metadata: ModelMetadata) -> "ModelLayersInfo":
        reason = unsupported_memory_reason(metadata)
        if reason:
            raise ValueError(f"Unsupported memory architecture: {reason}")
        return cls(
            vocab_size=metadata.vocab_size,
            heads=metadata.num_attention_heads,
            hidden_dim=metadata.hidden_size,
            inter_dim=metadata.intermediate_size,
            num_layers=metadata.num_hidden_layers,
            kv_heads=metadata.num_key_value_heads,
            head_dim=metadata.head_dim,
            layer_types=metadata.attention_layers(),
            linear_num_key_heads=metadata.linear_num_key_heads,
            linear_num_value_heads=metadata.linear_num_value_heads,
            linear_key_head_dim=metadata.linear_key_head_dim,
            linear_value_head_dim=metadata.linear_value_head_dim,
            linear_conv_kernel_dim=metadata.linear_conv_kernel_dim,
            active_experts=metadata.num_experts_per_tok or metadata.num_experts or 1,
            shared_inter_dim=metadata.shared_expert_intermediate_size or 0,
            vision=metadata.vision,
            weights_bytes=metadata.weights_bytes,
            architecture_type=metadata.architecture_type,
        )


@dataclass
class ModelMemInfo:
    """Memory required by model, unit in MiB."""

    model_mem: int
    kv_cache_mem: int
    activation_mem: int
    overhead: int
    total: int
    recurrent_state_mem: int = 0
    vision_mem: int = 0
    prefix_cache_mem: int = 0


QUANT_NORMALIZE = {
    "2bit": "2-bit",
    "3bit": "3-bit",
    "4bit": "4-bit",
    "5bit": "5-bit",
    "6bit": "6-bit",
    "8bit": "8-bit",
    "8bits": "8-bit",
    "2-bit": "2-bit",
    "3-bit": "3-bit",
    "5-bit": "5-bit",
    "6-bit": "6-bit",
    "int4": "4-bit",
    "int8": "8-bit",
    "fp4": "4-bit",
    "mxfp4": "4-bit",
    "nvfp4": "4-bit",
    "fp8": "8-bit",
    "4-bit": "4-bit",
    "8-bit": "8-bit",
}

GGUF_MULTI_FACTOR_DICT = {
    "q4_0": 18,
    "q4_1": 20,
    "q5_0": 22,
    "q5_1": 24,
    "q8_0": 34,
    "q8_1": 40,
}

GGUF_MULTI_FACTOR_DICT_64 = {
    "q6_K": 54.0,
    "q3": 26.0,
    "q4": 38.0,
    "q5": 46.0,
}

GGUF_MULTI_FACTOR_DICT_COMBINE = {
    "q3_K_L": [38.0, 26.0],
    "q3_K_M": [46.0, 26.0],
    "q4_K_S": [46.0, 38.0],
    "q4_K_M": [54.0, 38.0],
    "q5_K_M": [54.0, 46.0],
    "q2_K": [26.0, 22.0],
}


# Return gpu memory in MB
def estimate_llm_gpu_memory(
    model_size_in_billions: Union[str, int],
    quantization: Optional[str],
    context_length: int,  # input+output
    model_format: str,
    model_name: Optional[str] = None,
    kv_cache_dtype: int = 16,
    *,
    allow_download: bool = True,
) -> Optional[ModelMemInfo]:
    """
    model_size_in_billions: must be str like 1_8 or 46_7, to match llm.

    With allow_download=False, require catalog metadata and never download a
    config or infer architecture from parameter count. Missing metadata returns
    None. This remains an estimate, not a check of available device memory.
    """
    info = get_model_layers_info(
        model_size_in_billions,
        model_name,
        model_format,
        quantization,
        allow_download=allow_download,
    )
    if info is None:
        return None
    size_in_billions = convert_model_size_to_float(model_size_in_billions)
    return estimate_llm_gpu_memory_details(
        info,
        size_in_billions,
        quantization,
        context_length,
        model_format,
        kv_cache_dtype,
    )


def estimate_kv_cache_memory(
    info: ModelLayersInfo, num_tokens: int, kv_cache_dtype: int = 16
) -> float:
    """Full-attention K+V payload in MiB, across all layers (not per GPU).

    num_tokens is the total cached tokens across sequences. This excludes block
    padding, quantization scales, prefix sharing and engine preallocation. It
    assumes equal K/V head dimensions and MHA/GQA/MQA full-attention layers.
    An explicit layout counts full-attention layers and sliding-attention layers
    at the conservative full-cache upper bound. MLA needs different accounting.
    """
    if kv_cache_dtype not in (8, 16, 32):
        raise ValueError(f"Invalid kv_cache_dtype {kv_cache_dtype}")
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    kv_heads = info.heads if info.kv_heads is None else info.kv_heads
    if info.heads <= 0 or kv_heads <= 0 or info.num_layers <= 0 or info.hidden_dim <= 0:
        raise ValueError("Model dimensions must be positive")
    # Use true division to preserve the old unnamed-model heuristic, whose
    # inferred hidden size is not necessarily divisible by its inferred heads.
    head_dim = info.hidden_dim / info.heads if info.head_dim is None else info.head_dim
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    layers = (
        sum(t in ("full_attention", "sliding_attention") for t in info.layer_types)
        if info.layer_types is not None
        else info.num_layers
    )
    return (
        2 * num_tokens * layers * kv_heads * head_dim * (kv_cache_dtype / 8) / (1024**2)
    )


def estimate_llm_gpu_memory_details(
    info: ModelLayersInfo,
    size_in_billions: float,
    quantization: Optional[str],
    context_length: int,  # input+output
    model_format: str,
    kv_cache_dtype: int = 16,
    *,
    engine: Optional[str] = None,
    concurrency: int = 1,
    vision_tokens: int = 0,
    vision_tokens_per_image: int = 0,
    prefix_cache_bytes: int = 0,
) -> ModelMemInfo:
    """return model_mem, kv_cache, overhead, activation_mem"""
    if type(context_length) is not int or context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    if type(concurrency) is not int or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")
    if type(vision_tokens) is not int or vision_tokens < 0:
        raise ValueError("vision_tokens must be a non-negative integer")
    if type(prefix_cache_bytes) is not int or prefix_cache_bytes < 0:
        raise ValueError("prefix_cache_bytes must be a non-negative integer")
    if type(vision_tokens_per_image) is not int or vision_tokens_per_image < 0:
        raise ValueError("vision_tokens_per_image must be a non-negative integer")
    tokens = context_length + vision_tokens
    inference_mem = estimate_kv_cache_memory(info, tokens * concurrency, kv_cache_dtype)
    recurrent_mem = estimate_recurrent_state_memory(info, concurrency, engine)
    vision_mem = 0.0
    if vision_tokens:
        if info.vision is None:
            raise ValueError("Vision workspace metadata is missing")
        v = info.vision
        patches = vision_tokens * v.spatial_merge_size**2
        image_patches = (
            min(vision_tokens, vision_tokens_per_image or vision_tokens)
            * v.spatial_merge_size**2
        )
        # Bound a batch of independent images, without cross-image scores.
        # Without a declared per-image limit, use one image as the upper bound.
        vision_mem = (
            patches * (5 * v.hidden_size + 3 * v.intermediate_size) * 2
            + patches * image_patches * v.num_attention_heads * 4
        ) / 1024**2
    overhead = 650.0
    complex_weights = (
        info.architecture_type in ("moe", "multimodal", "hybrid_attention")
        or info.active_experts > 1
        or "linear_attention" in (info.layer_types or [])
        or info.vision is not None
    )
    if info.weights_bytes is not None and model_format != "bnb":
        model_size_in_mb = info.weights_bytes / 1024**2
        if model_format == "ggufv2":
            overhead += context_length * 0.1
    elif model_format == "ggufv2":
        if complex_weights:
            raise ValueError(
                "Hybrid/MoE/multimodal GGUF requires checkpoint size metadata"
            )
        assert quantization is not None and quantization != "none"
        model_size_in_mb = _compute_model_size_gguf(info, quantization)
        overhead = overhead + context_length * 0.1
    else:
        if (
            complex_weights
            and quantization is not None
            and quantization.lower() not in ("none", "bf16", "fp16", "f16")
        ):
            raise ValueError(
                "Quantized hybrid/MoE/multimodal models require checkpoint size metadata"
            )
        if quantization is not None and quantization.lower() in (
            "none",
            "bf16",
            "fp16",
            "f16",
        ):
            quantization = None
        if quantization is not None:
            assert isinstance(quantization, str)
            quantization = QUANT_NORMALIZE[quantization.lower()]
            assert quantization is not None

        model_size = size_in_billions * 1000000000.0
        model_size_in_mb = _convert_to_mb_model_size(model_size, quantization)
    serving_engine = (engine or "").lower() in ("vllm", "sglang")
    if engine is None:
        activation_mem = _compute_inference_only_activation_memory(tokens, info)
    elif serving_engine:
        # Serving engines share a bounded prefill batch and fused attention
        # buffers. The 16K-token profile covers larger hardware defaults;
        # custom batching and attention settings require manual configuration.
        batch = min(tokens * concurrency, 16384)
        head_dim = info.head_dim or info.hidden_dim / info.heads
        activation_mem = (
            batch
            * (
                5 * info.hidden_dim
                + 3 * (info.inter_dim * info.active_experts + info.shared_inter_dim)
            )
            * 2
            + batch * info.heads * (head_dim + 4) * 4
            + min(concurrency, batch) * info.vocab_size * 4
        ) / 1024**2
    else:
        # Transformers can use eager attention. Local engines use their
        # default prefill bound; these are estimates, not calibrated peaks.
        chunk = min(tokens, 2048) if engine.lower() in ("mlx", "llama.cpp") else tokens
        activation_mem = (
            chunk
            * (
                5 * info.hidden_dim
                + 3 * (info.inter_dim * info.active_experts + info.shared_inter_dim)
            )
            * 2
            + chunk * tokens * info.heads * 2
            + chunk * info.vocab_size * 4
        ) / 1024**2
    workspace_multiplier = 1 if serving_engine else concurrency
    workspace = max(activation_mem, vision_mem) * workspace_multiplier
    prefix_mem = prefix_cache_bytes / 1024**2

    total_mem = ceil(
        inference_mem
        + recurrent_mem
        + model_size_in_mb
        + overhead
        + workspace
        + prefix_mem
    )
    return ModelMemInfo(
        model_mem=ceil(model_size_in_mb),
        kv_cache_mem=ceil(inference_mem),
        activation_mem=ceil(workspace),
        overhead=ceil(overhead),
        total=total_mem,
        recurrent_state_mem=ceil(recurrent_mem),
        vision_mem=ceil(vision_mem * workspace_multiplier),
        prefix_cache_mem=ceil(prefix_mem),
    )


def estimate_recurrent_state_memory(
    info: ModelLayersInfo, concurrency: int, engine: Optional[str] = None
) -> float:
    layers = (info.layer_types or []).count("linear_attention")
    if not layers:
        return 0.0
    dimensions = (
        info.linear_num_key_heads,
        info.linear_num_value_heads,
        info.linear_key_head_dim,
        info.linear_value_head_dim,
        info.linear_conv_kernel_dim,
    )
    if any(value is None or value <= 0 for value in dimensions):
        raise ValueError("Linear attention state dimensions are missing")
    key_heads, value_heads, key_dim, value_dim, kernel = (
        int(value) for value in dimensions if value is not None
    )
    # Gated DeltaNet's recurrent accumulator is FP32. MLX retains kernel-1
    # convolution steps; other backends may retain a full kernel buffer.
    conv_steps = kernel - 1 if (engine or "").lower() == "mlx" else kernel
    state = value_heads * key_dim * value_dim * 4
    conv = conv_steps * (2 * key_heads * key_dim + value_heads * value_dim) * 2
    return layers * concurrency * (state + conv) / 1024**2


def load_model_config_json(config_path: str) -> ModelLayersInfo:
    with open(config_path, "r") as f:
        config_data = json.load(f)
        return ModelLayersInfo.from_metadata(ModelMetadata.from_config(config_data))


def get_model_layers_info(
    model_size_in_billions: Union[str, int],
    model_name: Optional[str],
    model_format: Optional[str],
    quantization: Optional[str],
    *,
    allow_download: bool = True,
) -> Optional[ModelLayersInfo]:
    from . import match_llm
    from .llm_family import cache_model_config

    if not model_name:
        if not allow_download:
            return None
        logger.debug("get_model_layers_info by default size=%s", model_size_in_billions)
        size_in_billions = convert_model_size_to_float(model_size_in_billions)
        return _get_default_layers_from_size(size_in_billions)
    llm_family = match_llm(
        model_name=model_name,
        model_format=model_format,
        model_size_in_billions=model_size_in_billions,
        quantization=quantization,
    )
    if not llm_family:
        return None
    metadata = llm_family.model_specs[0].model_metadata
    if metadata is not None:
        if unsupported_memory_reason(metadata):
            return None
        return ModelLayersInfo.from_metadata(metadata)
    if not allow_download:
        return None
    config_path = cache_model_config(llm_family)
    return load_model_config_json(config_path)


def _get_default_layers_from_size(size_in_billion: float) -> ModelLayersInfo:
    if size_in_billion < 5:
        vocab_size = 32000
        heads = 32
        num_layers = 24
    elif size_in_billion < 10:
        vocab_size = 32000
        heads = 32
        num_layers = 32
    elif size_in_billion < 24:
        vocab_size = 32000
        heads = 40
        num_layers = 40
    elif size_in_billion < 55:
        vocab_size = 32000
        heads = 60
        num_layers = 48
    else:
        vocab_size = 32000
        heads = 64
        num_layers = 80

    model_size = int(size_in_billion * 1000000000)
    A = num_layers * 4 + 3 * 4 * num_layers
    B = 2 * vocab_size
    C = -1 * model_size
    h = (-B + math.sqrt(B**2 - 4 * A * C)) / (2 * A)
    h = math.ceil(h)
    return ModelLayersInfo(
        vocab_size=vocab_size,
        heads=heads,
        hidden_dim=h,
        inter_dim=4 * h,
        num_layers=num_layers,
    )


def _convert_to_mb_model_size(model_size: float, quantization: Optional[str]) -> float:
    extra = 0.0
    fB = 2.0
    size = (model_size * fB) / (1024.0 * 1024.0)
    # bnb_q4 == 4-bit ?
    if quantization in ("2-bit", "3-bit", "4-bit", "5-bit", "6-bit", "8-bit"):
        extra = 0.06 * size
        size = size * int(quantization.split("-")[0]) / 16
    return size + extra


def _compute_inference_only_activation_memory(
    context_length: int, info: ModelLayersInfo
) -> float:
    hidden_dim = info.hidden_dim
    heads = info.heads
    ret = (
        (context_length * hidden_dim * 5 * 2 + (context_length**2) * heads * 2)
        / 1024
        / 1024
    )
    return ret


def _compute_model_size_gguf(info: ModelLayersInfo, quantization: str) -> float:
    assert quantization is not None
    names = {
        key.lower(): key
        for table in (
            GGUF_MULTI_FACTOR_DICT,
            GGUF_MULTI_FACTOR_DICT_64,
            GGUF_MULTI_FACTOR_DICT_COMBINE,
        )
        for key in table
    }
    if quantization.lower() not in names:
        raise ValueError(f"Unsupported GGUF memory quantization: {quantization}")
    quantization = names[quantization.lower()]
    vocab_size = info.vocab_size
    num_layers = info.num_layers
    hidden_dim = info.hidden_dim
    inter_dim = info.inter_dim
    total_params = int(
        vocab_size * hidden_dim * 2
        + num_layers * 4 * (hidden_dim**2)
        + num_layers * 3 * inter_dim * hidden_dim
    )
    other_v_down_params = (
        num_layers * (hidden_dim**2) + num_layers * hidden_dim * inter_dim
    )
    other_param_q2k = (
        total_params - (hidden_dim**2) * num_layers * 2 + 2 * vocab_size * hidden_dim
    )

    total = 0.0
    v1 = GGUF_MULTI_FACTOR_DICT.get(quantization)
    if v1 is not None:
        total = (v1 * total_params) / (32 * 1024 * 1024)
    v2 = GGUF_MULTI_FACTOR_DICT_64.get(quantization)
    if v2 is not None:
        total = (v2 * total_params) / (64 * 1024 * 1024)
    v3 = GGUF_MULTI_FACTOR_DICT_COMBINE.get(quantization)
    if v3 is not None:
        factors = v3
        if quantization == "q2_K":
            total = (
                (total_params - other_param_q2k) * factors[1]
                + other_param_q2k * factors[0]
            ) / (64 * 1024 * 1024)
        else:
            total = (
                (total_params - other_v_down_params) * factors[1]
                + other_v_down_params * factors[0]
            ) / (64 * 1024 * 1024)
    return total
