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
from unittest.mock import MagicMock

import pytest

from .. import collect_memory_catalog as collector
from ..gguf_model_metadata import read_gguf_config


def gguf_header(fields=None):
    def string(value):
        value = value.encode()
        return struct.pack("<Q", len(value)) + value

    fields = fields or {
        "general.architecture": "qwen3",
        "qwen3.embedding_length": 4096,
        "qwen3.block_count": 36,
        "qwen3.attention.head_count": 32,
        "qwen3.attention.head_count_kv": 8,
        "qwen3.attention.key_length": 128,
        "qwen3.feed_forward_length": 12288,
        "qwen3.vocab_size": 151936,
    }
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(fields))
    for key, value in fields.items():
        data += string(key)
        data += struct.pack("<I", 8 if isinstance(value, str) else 4)
        data += string(value) if isinstance(value, str) else struct.pack("<I", value)
    return data


def test_gguf_reader_stops_before_weights():
    header = gguf_header()
    stream = io.BytesIO(header + b"TENSOR CONTENT MUST NOT BE READ")
    config, digest, consumed = read_gguf_config(stream)
    assert stream.tell() == consumed == len(header)
    assert len(digest) == 64
    assert config["num_key_value_heads"] == 8
    assert config["head_dim"] == 128
    assert config["vocab_size"] == 151936
    with pytest.raises(ValueError, match="limit"):
        read_gguf_config(io.BytesIO(header), limit=20)
    with pytest.raises(ValueError, match="Truncated"):
        read_gguf_config(io.BytesIO(header[:-10]))


def test_catalog_collects_each_quantization_and_reports_missing(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "test.json").write_text(
        json.dumps(
            [
                dict(
                    model_name="test",
                    model_ability=["chat"],
                    model_specs=[
                        dict(
                            model_format="mlx",
                            model_size_in_billions=8,
                            model_src={
                                "huggingface": dict(
                                    model_id="org/model-{quantization}",
                                    quantizations=["4bit", "8bit"],
                                )
                            },
                        )
                    ],
                )
            ]
        )
    )
    calls = []

    def collect(job, cache, timeout):
        calls.append(job)
        if job[1].endswith("8bit"):
            return {"error": "repository unavailable"}
        return {
            "config": dict(
                vocab_size=32000,
                num_attention_heads=32,
                num_key_value_heads=8,
                hidden_size=4096,
                intermediate_size=14336,
                num_hidden_layers=32,
            ),
            "config_source": "https://huggingface.co/org/model-4bit/resolve/main/config.json",
            "config_sha256": "a" * 64,
        }

    monkeypatch.setattr(collector, "collect_config", collect)
    output = tmp_path / "output"
    collector.collect_catalog(catalog, output, tmp_path / "cache", 2, 1)
    source = json.loads((output / "models/test.json").read_text())[0]["model_specs"][0][
        "model_src"
    ]["huggingface"]
    assert set(source["model_metadata_by_quantization"]) == {"4bit"}
    assert len(calls) == 2
    report = json.loads((output / "coverage.json").read_text())
    assert [r["status"] for r in report] == ["collected", "unavailable"]
    with pytest.raises(FileExistsError):
        collector.collect_catalog(catalog, output, tmp_path / "cache", 2, 1)


@pytest.mark.parametrize("hub", ["huggingface", "modelscope", "openmind_hub", "csghub"])
def test_config_urls_encode_revision(hub):
    url = collector.config_url(hub, "org/model", "refs/pr/1")
    assert "org/model" in url
    assert "refs%2Fpr%2F1" in url


@pytest.mark.parametrize("hub", ["huggingface", "modelscope", "openmind_hub", "csghub"])
def test_weights_pin_older_config_snapshots(hub):
    result = {"config_source": collector.config_url(hub, "org/model", "b" * 40)}
    job = collector.checkpoint_weight_job(
        {"model_format": "mlx"}, {}, "4bit", (hub, "org/model", "main"), result
    )
    assert job == (hub, "org/model", "b" * 40)


def test_config_download_is_bounded_and_cached(tmp_path, monkeypatch):
    response = MagicMock()
    response.__enter__.return_value = response
    response.headers = {"x-repo-commit": "a" * 40}
    response.iter_content.return_value = [b'{"hidden_size": 4096}']
    get = MagicMock(return_value=response)
    monkeypatch.setattr("requests.get", get)
    job = ("huggingface", "org/model", "main")
    first = collector.collect_config(job, tmp_path, 1)
    assert first["config"]["hidden_size"] == 4096
    assert "a" * 40 in first["config_source"]
    assert collector.collect_config(job, tmp_path, 1) == first
    assert get.call_count == 1
    assert get.call_args.args[0].endswith("/config.json")
    response.iter_content.return_value = [b"x" * (2 * 1024**2 + 1)]
    result = collector.collect_config(
        ("huggingface", "org/oversize", "main"), tmp_path, 1
    )
    assert "exceeds 2 MiB" in result["error"]


@pytest.mark.parametrize("value_length", [256, 128])
def test_gguf_hybrid_dimensions(value_length):
    from ..memory import ModelLayersInfo, estimate_kv_cache_memory
    from ..model_metadata import ModelMetadata

    config, _, _ = read_gguf_config(
        io.BytesIO(
            gguf_header(
                {
                    "general.architecture": "qwen35moe",
                    "qwen35moe.embedding_length": 2048,
                    "qwen35moe.block_count": 40,
                    "qwen35moe.attention.head_count": 16,
                    "qwen35moe.attention.head_count_kv": 2,
                    "qwen35moe.attention.key_length": 256,
                    "qwen35moe.attention.value_length": value_length,
                    "qwen35moe.expert_feed_forward_length": 512,
                    "qwen35moe.feed_forward_length": 6144,
                    "qwen35moe.expert_count": 256,
                    "qwen35moe.expert_used_count": 8,
                    "qwen35moe.vocab_size": 248320,
                    "qwen35moe.full_attention_interval": 4,
                    "qwen35moe.ssm.conv_kernel": 4,
                    "qwen35moe.ssm.state_size": 128,
                    "qwen35moe.ssm.group_count": 16,
                    "qwen35moe.ssm.time_step_rank": 32,
                    "qwen35moe.ssm.inner_size": 4096,
                }
            )
        )
    )
    metadata = ModelMetadata.from_config(config)
    if value_length != 256:
        with pytest.raises(ValueError, match="unknown_attention_layout"):
            ModelLayersInfo.from_metadata(metadata)
        return
    info = ModelLayersInfo.from_metadata(metadata)
    assert metadata.full_attention_interval == 4
    assert metadata.layer_types is None
    assert info.active_experts == 8
    assert info.inter_dim == 512
    assert info.linear_value_head_dim == 128
    assert estimate_kv_cache_memory(info, 32768) == 640


def test_shared_config_keeps_quantization_weight_sizes_separate(tmp_path, monkeypatch):
    from .. import weights_metadata

    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "test.json").write_text(
        json.dumps(
            [
                dict(
                    model_name="test",
                    model_ability=["chat"],
                    model_specs=[
                        dict(
                            model_format="ggufv2",
                            model_size_in_billions=8,
                            model_file_name_template="model-{quantization}.gguf",
                            model_src={
                                "huggingface": dict(
                                    model_id="org/model",
                                    quantizations=["Q4_K_M", "Q8_0"],
                                )
                            },
                        )
                    ],
                )
            ]
        )
    )
    config = dict(
        vocab_size=32000,
        num_attention_heads=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
    )
    monkeypatch.setattr(
        collector,
        "collect_config",
        lambda *args: {
            "config": config,
            "config_source": "pinned",
            "config_sha256": "a" * 64,
            "resolved_revision": "b" * 40,
        },
    )
    calls = []

    def weights(job, *args):
        calls.append(job)
        return dict(
            weights_bytes=4000 if "Q4_K_M" in job[3][0] else 8000,
            weights_source="pinned",
            weights_sha256="c" * 64,
        )

    monkeypatch.setattr(weights_metadata, "collect_weights_size", weights)
    output = tmp_path / "output"
    collector.collect_catalog(catalog, output, tmp_path / "cache", 2, 1, weights=True)
    source = json.loads((output / "models/test.json").read_text())[0]["model_specs"][0][
        "model_src"
    ]["huggingface"]
    assert "model_metadata" not in source
    entries = source["model_metadata_by_quantization"]
    assert entries["Q4_K_M"]["weights_bytes"] == 4000
    assert entries["Q8_0"]["weights_bytes"] == 8000
    assert all(job[2] == "b" * 40 for job in calls)


@pytest.mark.parametrize("weights", [False, True])
@pytest.mark.parametrize("same_snapshot", [False, True])
@pytest.mark.parametrize("common", [False, True])
def test_collector_preserves_sizes_only_for_same_snapshot(
    tmp_path, monkeypatch, weights, same_snapshot, common
):
    from .. import weights_metadata

    config = dict(
        vocab_size=32000,
        num_attention_heads=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
    )
    old = dict(
        config,
        config_source="pinned",
        config_sha256="a" * 64,
        weights_bytes=1000,
        weights_source="weights",
        weights_sha256="c" * 64,
    )
    source = dict(model_id="org/model-{quantization}", quantizations=["4bit"])
    if common:
        source["model_metadata"] = old
    else:
        source["model_metadata_by_quantization"] = {"4bit": old, "8bit": old}
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "test.json").write_text(
        json.dumps(
            [
                dict(
                    model_name="test",
                    model_ability=["chat"],
                    model_specs=[
                        dict(
                            model_format="mlx",
                            model_size_in_billions=8,
                            model_src={"huggingface": source},
                        )
                    ],
                )
            ]
        )
    )
    monkeypatch.setattr(
        collector,
        "collect_config",
        lambda *args: dict(
            config=config,
            config_source="pinned",
            config_sha256=("a" if same_snapshot else "b") * 64,
        ),
    )
    monkeypatch.setattr(
        weights_metadata,
        "collect_weights_size",
        lambda *args: {"error": "temporary failure"},
    )
    output = tmp_path / "output"
    collector.collect_catalog(
        catalog, output, tmp_path / "cache", 2, 1, weights=weights
    )
    result = json.loads((output / "models/test.json").read_text())[0]["model_specs"][0][
        "model_src"
    ]["huggingface"]
    assert "model_metadata_by_quantization" not in result
    if same_snapshot:
        assert result["model_metadata"]["weights_bytes"] == 1000
    else:
        assert "weights_bytes" not in result["model_metadata"]


def test_checkpoint_job_splits_and_projectors_and_bnb():
    spec = dict(
        model_format="ggufv2",
        model_file_name_split_template="model-{quantization}-{part}.gguf",
        quantization_parts={"Q4_K_M": ["00001", "00002"]},
        multimodal_projectors=["vision-{quantization}.gguf"],
    )
    result = collector.checkpoint_weight_job(
        spec,
        {},
        "Q4_K_M",
        ("huggingface", "org/model", "main"),
        {"resolved_revision": "pinned"},
    )
    assert result == (
        "huggingface",
        "org/model",
        "pinned",
        ("model-Q4_K_M-00001.gguf", "model-Q4_K_M-00002.gguf"),
        ("vision-Q4_K_M.gguf",),
    )
    assert (
        collector.checkpoint_weight_job(
            {"model_format": "bnb"},
            {},
            "4bit",
            ("huggingface", "org/model", "main"),
            {},
        )
        is None
    )
