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

import json
from dataclasses import replace

import pytest

from ..memory import (
    ModelLayersInfo,
    estimate_kv_cache_memory,
    estimate_llm_gpu_memory,
    estimate_llm_gpu_memory_details,
    get_model_layers_info,
)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_mlx_quantization_aliases(bits):
    info = ModelLayersInfo(32000, 32, 4096, 14336, 32, kv_heads=8)
    mlx = estimate_llm_gpu_memory_details(info, 8, f"{bits}bit", 2048, "mlx")
    standard = estimate_llm_gpu_memory_details(info, 8, f"{bits}-bit", 2048, "mlx")
    assert mlx == standard
    assert mlx.model_mem > 0


@pytest.mark.parametrize("quant", ["bf16", "BF16", "fp16", "F16"])
def test_mlx_unquantized_aliases(quant):
    info = ModelLayersInfo(32000, 32, 4096, 14336, 32)
    assert estimate_llm_gpu_memory_details(
        info, 8, quant, 2048, "mlx"
    ) == estimate_llm_gpu_memory_details(info, 8, None, 2048, "mlx")


def test_gguf_quantization_case_and_unknown():
    info = ModelLayersInfo(32000, 32, 4096, 14336, 32)
    assert estimate_llm_gpu_memory_details(
        info, 8, "Q4_K_M", 2048, "ggufv2"
    ) == estimate_llm_gpu_memory_details(info, 8, "q4_k_m", 2048, "ggufv2")
    with pytest.raises(ValueError, match="Unsupported GGUF"):
        estimate_llm_gpu_memory_details(info, 8, "unknown", 2048, "ggufv2")


@pytest.mark.parametrize("kv_heads,expected_mib", [(32, 1024), (8, 256), (1, 32)])
@pytest.mark.parametrize("dtype", [8, 16, 32])
def test_kv_cache_mha_gqa_mqa(kv_heads, expected_mib, dtype):
    info = ModelLayersInfo(32000, 32, 4096, 14336, 32, kv_heads=kv_heads)
    assert estimate_kv_cache_memory(info, 2048, dtype) == expected_mib * dtype / 16
    assert estimate_kv_cache_memory(info, 4096, dtype) == 2 * expected_mib * dtype / 16
    assert estimate_kv_cache_memory(info, 0, dtype) == 0


def test_kv_cache_explicit_head_dim_and_mha_default():
    info = ModelLayersInfo(32000, 16, 1024, 3072, 28, kv_heads=8, head_dim=128)
    assert estimate_kv_cache_memory(info, 2048) == 224
    assert estimate_kv_cache_memory(replace(info, head_dim=None), 2048) == 112
    assert estimate_kv_cache_memory(replace(info, kv_heads=None), 2048) == 448


@pytest.mark.parametrize(
    "model_format,quantization",
    [("pytorch", None), ("gptq", "Int4"), ("ggufv2", "q4_0")],
)
def test_kv_cache_independent_of_weight_format(model_format, quantization):
    info = ModelLayersInfo(32000, 32, 4096, 14336, 32, kv_heads=8)
    gqa = estimate_llm_gpu_memory_details(info, 7, quantization, 2048, model_format)
    mha = estimate_llm_gpu_memory_details(
        replace(info, kv_heads=32), 7, quantization, 2048, model_format
    )
    assert gqa.kv_cache_mem == 256
    assert mha.kv_cache_mem == 1024
    assert gqa.activation_mem == mha.activation_mem == 336
    assert mha.total - gqa.total == 768


def test_bundled_qwen3_kv_cache():
    # 36 layers, 8 KV heads, head_dim=128, FP16, 8192 cached tokens.
    result = estimate_llm_gpu_memory(
        8, "none", 8192, "pytorch", "qwen3", allow_download=False
    )
    assert result.kv_cache_mem == 1152
    info = get_model_layers_info(8, "qwen3", "pytorch", "none", allow_download=False)
    assert info.heads == 32
    assert info.kv_heads == 8
    assert estimate_kv_cache_memory(info, 4 * 8192) == 4608


@pytest.mark.parametrize("tokens,dtype", [(-1, 16), (1, 4)])
def test_kv_cache_rejects_invalid_inputs(tokens, dtype):
    with pytest.raises(ValueError):
        estimate_kv_cache_memory(
            ModelLayersInfo(32000, 32, 4096, 14336, 32), tokens, dtype
        )


def test_llm_estimate_memory(monkeypatch, tmp_path):
    # Dimensions from the published Qwen1.5 checkpoint config.json files.
    # Keep formula tests independent of downloads and stale local cache symlinks.
    dimensions = {
        72: (152064, 64, 64, 8192, 24576, 80),
        32: (152064, 40, 8, 5120, 27648, 64),
        1.8: (151936, 16, 16, 2048, 5504, 24),
    }

    def cached_config(family):
        size = float(
            str(family.model_specs[0].model_size_in_billions).replace("_", ".")
        )
        config = dict(
            zip(
                (
                    "vocab_size",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "hidden_size",
                    "intermediate_size",
                    "num_hidden_layers",
                ),
                dimensions[size],
            )
        )
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        return str(path)

    monkeypatch.setattr(
        "xinference.model.llm.llm_family.cache_model_config", cached_config
    )

    # without model_name, use default ModelLayersInfo
    mem_info = estimate_llm_gpu_memory(1.8, None, 2048, "pytorch", kv_cache_dtype=32)
    assert mem_info.total == 5162
    mem_info = estimate_llm_gpu_memory(
        72, "8-bit", 1024 + 2048, "pytorch", kv_cache_dtype=32
    )
    assert mem_info.total == 92943
    mem_info = estimate_llm_gpu_memory(
        72, "4-bit", 1024 + 2048, "pytorch", kv_cache_dtype=32
    )
    assert mem_info.total == 58611
    mem_info = estimate_llm_gpu_memory(7, "Int4", 32768, "gptq", kv_cache_dtype=32)
    assert mem_info.total == 100550

    # with model_name to match_llm
    mem_info = estimate_llm_gpu_memory(
        72, "Int4", 32768, "gptq", kv_cache_dtype=16, model_name="qwen1.5-chat"
    )
    assert mem_info.total == 258775

    # model_size_in_billions use int
    mem_info = estimate_llm_gpu_memory(
        32, "Int4", 32768, "gptq", kv_cache_dtype=8, model_name="qwen1.5-chat"
    )
    # model_size_in_billions use str
    mem_info = estimate_llm_gpu_memory(
        "32", "Int4", 32768, "gptq", kv_cache_dtype=8, model_name="qwen1.5-chat"
    )
    # GQA: KV cache uses 8 heads, activation uses all 40 attention heads.
    assert mem_info.kv_cache_mem == 4096
    assert mem_info.total == 107187

    # model_size_in_billions use float
    mem_info = estimate_llm_gpu_memory(
        "1.8", None, 2048, "pytorch", kv_cache_dtype=32, model_name="qwen1.5-chat"
    )
    assert mem_info.total == 5020
