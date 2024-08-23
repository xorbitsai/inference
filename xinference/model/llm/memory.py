# Copyright 2022-2023 XProbe Inc.
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
#   The algorithum is ported from https://github.com/RahulSChand/gpu_poor
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
from typing import Any, Optional, Union

from .llm_family import convert_model_size_to_float

logger = getLogger(__name__)


@dataclass
class ModelLayersInfo:
    vocab_size: int
    heads: int  # num_attention_heads, num_heads or n_head
    hidden_dim: int  # hidden_size, d_model, or n_embd
    inter_dim: int  # intermediate_size, n_inner or d_ff
    num_layers: int  # num_layers, num_hidden_layers or n_layer


@dataclass
class ModelMemInfo:
    """Memory required by model, unit in MB"""

    model_mem: int
    kv_cache_mem: int
    activation_mem: int
    overhead: int
    total: int


QUANT_NORMALIZE = {"int4": "4-bit", "int8": "8-bit", "4-bit": "4-bit", "8-bit": "8-bit"}

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
) -> Optional[ModelMemInfo]:
    """
    model_size_in_billions: must be str like 1_8 or 46_7, to match llm.
    """
    info = get_model_layers_info(
        model_size_in_billions,
        model_name,
        model_format,
        quantization,
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


def estimate_llm_gpu_memory_details(
    info: ModelLayersInfo,
    size_in_billions: float,
    quantization: Optional[str],
    context_length: int,  # input+output
    model_format: str,
    kv_cache_dtype: int = 16,
) -> ModelMemInfo:
    """return model_mem, kv_cache, overhead, activation_mem"""
    if kv_cache_dtype not in [8, 16, 32]:
        raise ValueError(f"Invalid kv_cache_dtype {kv_cache_dtype}")
    if kv_cache_dtype == 8:
        kv_dtype_size = 1
    elif kv_cache_dtype == 16:
        kv_dtype_size = 2
    else:
        kv_dtype_size = 4
    overhead = 650.0
    if model_format == "ggufv2":
        assert quantization is not None and quantization != "none"
        model_size_in_mb = _compute_model_size_gguf(info, quantization)
        inference_mem = float(
            context_length * kv_dtype_size * info.hidden_dim * info.num_layers
        )
        inference_mem = inference_mem / 1024.0 / 1024.0
        activation_mem = _compute_inference_only_activation_memory(context_length, info)
        overhead = overhead + context_length * 0.1
    else:
        if quantization is not None:
            assert isinstance(quantization, str)
            quantization = QUANT_NORMALIZE[quantization.lower()]
            assert quantization is not None

        model_size = size_in_billions * 1000000000.0
        model_size_in_mb = _convert_to_mb_model_size(model_size, quantization)
        # KV cache
        inference_mem = float(
            context_length * 2 * kv_dtype_size * info.hidden_dim * info.num_layers
        )
        inference_mem = inference_mem / 1024.0 / 1024.0
        activation_mem = _compute_inference_only_activation_memory(context_length, info)

    total_mem = ceil(inference_mem + model_size_in_mb + overhead + activation_mem)
    return ModelMemInfo(
        model_mem=ceil(model_size_in_mb),
        kv_cache_mem=ceil(inference_mem),
        activation_mem=ceil(activation_mem),
        overhead=ceil(overhead),
        total=total_mem,
    )


def _load_item_from_json(config_data: Any, *keys: str) -> str:
    assert len(keys) > 0
    for key in keys:
        v = config_data.get(key)
        if v is not None:
            return v
    raise ValueError("load ModelLayersInfo: missing %s" % (keys[0]))


def load_model_config_json(config_path: str) -> ModelLayersInfo:
    with open(config_path, "r") as f:
        config_data = json.load(f)
        return ModelLayersInfo(
            vocab_size=int(_load_item_from_json(config_data, "vocab_size")),
            heads=int(
                _load_item_from_json(
                    config_data, "num_key_value_heads", "num_attention_heads"
                )
            ),
            hidden_dim=int(
                _load_item_from_json(config_data, "hidden_size", "d_model", "n_embd")
            ),
            inter_dim=int(_load_item_from_json(config_data, "intermediate_size")),
            num_layers=int(
                _load_item_from_json(
                    config_data, "num_hidden_layers", "num_layers", "n_layer"
                )
            ),
        )


def get_model_layers_info(
    model_size_in_billions: Union[str, int],
    model_name: Optional[str],
    model_format: Optional[str],
    quantization: Optional[str],
) -> Optional[ModelLayersInfo]:
    from . import match_llm
    from .llm_family import cache_model_config

    if not model_name:
        logger.debug("get_model_layers_info by default size=%s", model_size_in_billions)
        size_in_billions = convert_model_size_to_float(model_size_in_billions)
        return _get_default_layers_from_size(size_in_billions)
    match_result = match_llm(
        model_name=model_name,
        model_format=model_format,
        model_size_in_billions=model_size_in_billions,
        quantization=quantization,
    )
    if not match_result:
        return None
    llm_family, llm_spec, _quant = match_result
    config_path = cache_model_config(llm_family, llm_spec)
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
    if quantization == "8-bit" or quantization == "4-bit":
        extra = 0.06 * size
    if quantization == "8-bit":
        size = size / 2
    if quantization == "4-bit":
        size = size / 4
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
