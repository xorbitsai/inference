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
    hybrid.weights_bytes = 20 * 1024**3
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
    hybrid.weights_bytes = 20 * 1024**3
    info = ModelLayersInfo.from_metadata(hybrid)
    mlx = estimate_llm_gpu_memory_details(info, 35, "4bit", 8192, "mlx", engine="MLX")
    transformers = estimate_llm_gpu_memory_details(
        info, 35, "4bit", 8192, "mlx", engine="Transformers"
    )
    assert transformers.activation_mem > mlx.activation_mem
    assert transformers.model_mem == mlx.model_mem


def test_vision_and_prefix_cache_are_separate(hybrid):
    hybrid.weights_bytes = 20 * 1024**3
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


@pytest.mark.parametrize(
    "fmt,quant", [("gptq", "Int4"), ("awq", "Int4"), ("fp8", "FP8"), ("mlx", "4bit")]
)
def test_quantized_hybrid_without_checkpoint_size_is_unverified(hybrid, fmt, quant):
    with pytest.raises(ValueError, match="require checkpoint size"):
        estimate_llm_gpu_memory_details(
            ModelLayersInfo.from_metadata(hybrid), 35, quant, 2048, fmt
        )


@pytest.mark.parametrize("engine", ["vLLM", "SGLang"])
def test_serving_workspace_is_bounded_and_shared(hybrid, engine):
    hybrid.weights_bytes = 20 * 1024**3
    info = ModelLayersInfo.from_metadata(hybrid)
    first = estimate_llm_gpu_memory_details(
        info, 35, "FP8", 32768, "fp8", engine=engine
    )
    long = estimate_llm_gpu_memory_details(
        info, 35, "FP8", 65536, "fp8", engine=engine, concurrency=4
    )
    assert first.activation_mem < 2 * 1024
    assert long.activation_mem < first.activation_mem + 10
    assert long.kv_cache_mem == 8 * first.kv_cache_mem
    assert long.recurrent_state_mem >= 4 * first.recurrent_state_mem - 3


def test_image_attention_does_not_cross_image_boundaries(hybrid):
    hybrid.weights_bytes = 20 * 1024**3
    info = ModelLayersInfo.from_metadata(hybrid)
    image = estimate_llm_gpu_memory_details(
        info, 35, "4bit", 2048, "mlx", vision_tokens=256, vision_tokens_per_image=256
    )
    four = estimate_llm_gpu_memory_details(
        info, 35, "4bit", 2048, "mlx", vision_tokens=1024, vision_tokens_per_image=256
    )
    assert four.vision_mem <= 4 * image.vision_mem


def test_sliding_attention_uses_full_cache_upper_bound(hybrid):
    hybrid.layer_types = ["sliding_attention"] * hybrid.num_hidden_layers
    info = ModelLayersInfo.from_metadata(hybrid)
    from ..memory import estimate_kv_cache_memory

    assert estimate_kv_cache_memory(info, 32768) == 2560


@pytest.mark.parametrize(
    "headers,status,size",
    [
        ({"Content-Range": "bytes 0-0/123456", "Content-Length": "1"}, 206, 123456),
        ({"Content-Length": "123456"}, 200, 123456),
        ({"Content-Range": "bytes 0-0/*", "Content-Length": "1"}, 206, None),
    ],
)
@pytest.mark.parametrize("head_failure", [None, "transport", 403, 405])
def test_gguf_range_size_fallback(
    tmp_path, monkeypatch, headers, status, size, head_failure
):
    import requests

    head = MagicMock()
    head.__enter__.return_value = head
    head.headers = {}
    get = MagicMock()
    get.__enter__.return_value = get
    get.headers = headers
    get.status_code = status
    if head_failure == "transport":
        head_request = MagicMock(side_effect=requests.ConnectionError("HEAD failed"))
    else:
        head_request = MagicMock(return_value=head)
        if head_failure:
            head.raise_for_status.side_effect = requests.HTTPError(str(head_failure))
    monkeypatch.setattr("requests.head", head_request)
    request_get = MagicMock(return_value=get)
    monkeypatch.setattr("requests.get", request_get)
    result = collect_weights_size(
        ("modelscope", "test/model", "master", ("model.gguf",), ()), tmp_path, 1
    )
    if size is None:
        assert "error" in result
    else:
        assert result["weights_bytes"] == size
    assert request_get.call_args.kwargs["headers"]["Range"] == "bytes=0-0"
    get.raw.read.assert_not_called()
    get.iter_content.assert_not_called()


def test_single_safetensors_fallback_is_header_only(tmp_path, monkeypatch):
    index = MagicMock()
    index.__enter__.return_value = index
    index.status_code = 404
    header = json.dumps({"w": {"data_offsets": [0, 20]}}).encode()
    stream = io.BytesIO(struct.pack("<Q", len(header)) + header + b"WEIGHT PAYLOAD")
    weights = MagicMock()
    weights.__enter__.return_value = weights
    weights.raw = stream
    get = MagicMock(side_effect=[index, weights])
    monkeypatch.setattr("requests.get", get)
    result = collect_weights_size(("huggingface", "test/model", "pinned"), tmp_path, 1)
    assert result["weights_bytes"] == 20
    assert stream.tell() == 8 + len(header)


@pytest.mark.parametrize(
    "header",
    [
        b"[]",
        b"{}",
        b'{"w":{"data_offsets":[1,0]}}',
        b'{"w":{"data_offsets":[false,20]}}',
    ],
)
def test_invalid_safetensors_header_is_rejected(header):
    with pytest.raises(ValueError):
        read_safetensors_size(io.BytesIO(struct.pack("<Q", len(header)) + header))


def test_moe_prefers_expert_ffn_dimensions(hybrid):
    config = hybrid.dict()
    config.update(intermediate_size=6144, moe_intermediate_size=768)
    assert ModelMetadata.from_config(config).intermediate_size == 768


def test_cli_prints_new_components_and_reports_unsupported(monkeypatch):
    from click.testing import CliRunner

    from xinference.deploy.cmdline import cal_model_mem

    from ..memory import ModelMemInfo

    monkeypatch.setattr(
        "xinference.model.llm.memory.estimate_llm_gpu_memory",
        lambda **kwargs: ModelMemInfo(
            10, 2, 4, 1, 25, recurrent_state_mem=3, prefix_cache_mem=5
        ),
    )
    args = ["-s", "35", "-f", "mlx", "-q", "4bit", "-c", "2048"]
    result = CliRunner().invoke(cal_model_mem, args)
    assert result.exit_code == 0
    assert (
        "recurrent state: 3 MB" in result.output
        and "retained prefix cache: 5 MB" in result.output
    )

    def unsupported(**kwargs):
        raise ValueError("unknown_attention_layout")

    monkeypatch.setattr(
        "xinference.model.llm.memory.estimate_llm_gpu_memory", unsupported
    )
    result = CliRunner().invoke(cal_model_mem, args)
    assert result.exit_code == 1
    assert "Memory estimate unavailable: unknown_attention_layout" in result.output


def test_refreshed_qwen_gguf_catalog_has_pinned_sizes_and_vision():
    from pathlib import Path

    from ..model_metadata import ModelMetadata

    catalog = Path(__file__).parents[1] / "models"
    for name in ["qwen3.5", "qwen3.6"]:
        for family in json.loads((catalog / f"{name}.json").read_text()):
            for spec in family["model_specs"]:
                if spec["model_format"] != "ggufv2":
                    continue
                for hub, source in spec["model_src"].items():
                    entries = source.get(
                        "model_metadata_by_quantization"
                    ) or dict.fromkeys(
                        source["quantizations"], source["model_metadata"]
                    )
                    for quant, raw in entries.items():
                        metadata = ModelMetadata.parse_obj(raw)
                        if hub == "huggingface":
                            assert "/resolve/main/" not in metadata.config_source
                            if metadata.weights_source:
                                assert "/resolve/main/" not in metadata.weights_source
                        if name == "qwen3.6":
                            assert metadata.shared_expert_intermediate_size == 512
                            assert metadata.vision.spatial_merge_size == 2
                            assert metadata.vision.config_source.endswith(
                                "mmproj-BF16.gguf"
                            )
                            assert metadata.vision.config_sha256
                            if metadata.weights_bytes:
                                memory = estimate_llm_gpu_memory_details(
                                    ModelLayersInfo.from_metadata(metadata),
                                    35,
                                    quant,
                                    2048,
                                    "ggufv2",
                                    engine="llama.cpp",
                                    vision_tokens=128,
                                )
                                assert memory.vision_mem > 0
                        if (
                            name == "qwen3.5"
                            and hub == "huggingface"
                            and spec["model_size_in_billions"] == 9
                        ):
                            assert metadata.weights_bytes
