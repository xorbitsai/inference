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

import io
import json
import struct
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from ..memory import (
    ModelLayersInfo,
    estimate_llm_gpu_memory_details,
    estimate_recurrent_state_memory,
)
from ..model_metadata import ModelMetadata
from ..weights_metadata import collect_weights_size, read_safetensors_size


@pytest.fixture
def hybrid():
    return ModelMetadata.from_config(
        {
            "text_config": {
                "vocab_size": 248320,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "hidden_size": 2048,
                "intermediate_size": 512,
                "num_hidden_layers": 40,
                "full_attention_interval": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "num_experts": 256,
                "num_experts_per_tok": 8,
            },
            "vision_config": {
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_heads": 16,
                "spatial_merge_size": 2,
            },
        }
    )


@pytest.mark.parametrize(
    "engine,model_format,quantization",
    [
        ("MLX", "mlx", "4bit"),
        ("Transformers", "pytorch", "none"),
        ("vLLM", "gptq", "Int4"),
        ("SGLang", "fp8", "FP8"),
        ("llama.cpp", "ggufv2", "Q4_K_M"),
    ],
)
def test_hybrid_cache_for_every_engine(hybrid, engine, model_format, quantization):
    hybrid.weights_bytes = 20 * 1024**3
    info = ModelLayersInfo.from_metadata(hybrid)
    result = estimate_llm_gpu_memory_details(
        info, 35, quantization, 32768, model_format, engine=engine
    )
    assert result.model_mem == 20 * 1024
    assert result.kv_cache_mem == 640
    assert result.recurrent_state_mem >= 60
    assert (
        result.total
        >= result.model_mem + result.kv_cache_mem + result.recurrent_state_mem
    )
    doubled = estimate_llm_gpu_memory_details(
        info, 35, quantization, 32768, model_format, engine=engine, concurrency=2
    )
    assert doubled.model_mem == result.model_mem
    assert doubled.kv_cache_mem == 2 * result.kv_cache_mem
    assert doubled.recurrent_state_mem >= 2 * result.recurrent_state_mem - 1


def test_recurrent_state_does_not_grow_with_context(hybrid):
    info = ModelLayersInfo.from_metadata(hybrid)
    assert estimate_recurrent_state_memory(info, 1, "MLX") == pytest.approx(61.40625)
    assert estimate_recurrent_state_memory(info, 1, "Transformers") > 61.40625
    first = estimate_llm_gpu_memory_details(info, 35, "4bit", 2048, "mlx", engine="MLX")
    second = estimate_llm_gpu_memory_details(
        info, 35, "4bit", 4096, "mlx", engine="MLX"
    )
    assert first.recurrent_state_mem == second.recurrent_state_mem
    assert second.kv_cache_mem == 2 * first.kv_cache_mem


def test_checkpoint_size_handles_mixed_quantization(hybrid):
    hybrid.weights_bytes = 123456789
    result = estimate_llm_gpu_memory_details(
        ModelLayersInfo.from_metadata(hybrid), 35, "mixed-future-quant", 2048, "mlx"
    )
    assert result.model_mem == 118


@pytest.mark.parametrize("interval", [0, -1, True, "4"])
def test_invalid_attention_interval_is_rejected(hybrid, interval):
    config = hybrid.dict()
    config["full_attention_interval"] = interval
    with pytest.raises(ValueError, match="full_attention_interval"):
        ModelMetadata.from_config(config)


def test_mla_does_not_borrow_gated_delta_accounting(hybrid):
    config = hybrid.dict()
    config["kv_lora_rank"] = 512
    metadata = ModelMetadata.from_config(config)
    assert metadata.architecture_type == "mla"
    with pytest.raises(ValueError, match="mla"):
        ModelLayersInfo.from_metadata(metadata)


def test_old_moe_metadata_remains_unverified(hybrid):
    old = hybrid.copy(update={"architecture_type": "moe", "num_experts": None})
    with pytest.raises(ValueError, match="incomplete_expert_layout"):
        ModelLayersInfo.from_metadata(old)


def test_multimodal_gguf_needs_checkpoint_size(hybrid):
    multimodal = hybrid.copy(
        update={
            "architecture_type": "multimodal",
            "num_experts": None,
            "num_experts_per_tok": None,
            "full_attention_interval": None,
            "layer_types": ["full_attention"] * hybrid.num_hidden_layers,
        }
    )
    with pytest.raises(ValueError, match="requires checkpoint size"):
        estimate_llm_gpu_memory_details(
            ModelLayersInfo.from_metadata(multimodal), 35, "Q4_K_M", 2048, "ggufv2"
        )


def test_moe_gguf_does_not_use_a_dense_weight_formula(hybrid):
    with pytest.raises(ValueError, match="requires checkpoint size"):
        estimate_llm_gpu_memory_details(
            ModelLayersInfo.from_metadata(hybrid), 35, "Q4_K_M", 2048, "ggufv2"
        )


def test_engine_workspace_bound_changes_for_long_context(hybrid):
    info = ModelLayersInfo.from_metadata(hybrid)
    mlx = estimate_llm_gpu_memory_details(info, 35, "4bit", 8192, "mlx", engine="MLX")
    transformers = estimate_llm_gpu_memory_details(
        info, 35, "4bit", 8192, "mlx", engine="Transformers"
    )
    assert transformers.activation_mem > mlx.activation_mem
    assert transformers.model_mem == mlx.model_mem


def test_vision_and_prefix_cache_are_separate(hybrid):
    info = ModelLayersInfo.from_metadata(hybrid)
    text = estimate_llm_gpu_memory_details(info, 35, "4bit", 2048, "mlx", engine="MLX")
    image = estimate_llm_gpu_memory_details(
        info,
        35,
        "4bit",
        2048,
        "mlx",
        engine="MLX",
        vision_tokens=256,
        prefix_cache_bytes=1024**3,
    )
    assert image.vision_mem > 0
    assert image.prefix_cache_mem == 1024
    assert image.kv_cache_mem > text.kv_cache_mem
    assert image.total >= text.total + 1024
    with pytest.raises(ValueError, match="Vision workspace"):
        estimate_llm_gpu_memory_details(
            replace(info, vision=None), 35, "4bit", 2048, "mlx", vision_tokens=256
        )


def test_incomplete_or_unknown_layout_is_not_estimated(hybrid):
    hybrid.linear_key_head_dim = None
    with pytest.raises(ValueError, match="incomplete_linear_attention"):
        ModelLayersInfo.from_metadata(hybrid)
    hybrid.layer_types = ["unknown"] * 40
    with pytest.raises(ValueError, match="unknown_attention_layout"):
        ModelLayersInfo.from_metadata(hybrid)


def test_safetensors_reader_stops_before_payload():
    header = json.dumps(
        {"weight": {"dtype": "F16", "shape": [10], "data_offsets": [0, 20]}}
    ).encode()
    stream = io.BytesIO(
        struct.pack("<Q", len(header)) + header + b"DO NOT READ WEIGHTS"
    )
    size, digest = read_safetensors_size(stream)
    assert size == 20
    assert len(digest) == 64
    assert stream.tell() == 8 + len(header)
    with pytest.raises(ValueError, match="collection limit"):
        read_safetensors_size(io.BytesIO(struct.pack("<Q", 1024)), limit=100)


def test_weight_collection_is_bounded_and_cached(tmp_path, monkeypatch):
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = 200
    response.iter_content.return_value = [b'{"metadata":{"total_size":1234}}']
    get = MagicMock(return_value=response)
    monkeypatch.setattr("requests.get", get)
    job = ("huggingface", "test/quant", "pinned-revision")
    result = collect_weights_size(job, tmp_path, 1)
    assert result["weights_bytes"] == 1234
    assert "pinned-revision" in result["weights_source"]
    assert collect_weights_size(job, tmp_path, 1) == result
    assert get.call_count == 1
    response.iter_content.return_value = [b"x" * (8 * 1024**2 + 1)]
    assert (
        "collection limit"
        in collect_weights_size(("huggingface", "oversized", "main"), tmp_path, 1)[
            "error"
        ]
    )


def test_gguf_sizes_sum_shards_and_only_one_projector(tmp_path, monkeypatch):
    sizes = {
        "a.gguf": 1000,
        "b.gguf": 2000,
        "vision-f16.gguf": 200,
        "vision-f32.gguf": 400,
    }

    def head(url, **kwargs):
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {"Content-Length": str(sizes[url.rsplit("/", 1)[-1]])}
        return response

    monkeypatch.setattr("requests.head", head)
    job = (
        "huggingface",
        "test/gguf",
        "pinned",
        ("a.gguf", "b.gguf"),
        ("vision-f16.gguf", "vision-f32.gguf"),
    )
    assert collect_weights_size(job, tmp_path, 1)["weights_bytes"] == 3400
