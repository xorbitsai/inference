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
import re
from types import SimpleNamespace

import pytest

from ...._compat import ValidationError
from ...._model_catalog import load_model_catalog
from ...utils import flatten_quantizations
from .. import llm_family
from ..collect_memory_metadata import main
from ..llm_family import LlamaCppLLMSpecV2, MLXLLMSpecV2, PytorchLLMSpecV2
from ..memory import (
    estimate_llm_gpu_memory,
    get_model_layers_info,
    load_model_config_json,
)
from ..memory_metadata import ModelMemoryMetadata


@pytest.fixture
def config():
    return dict(
        vocab_size=32000,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
    )


def no_download(*args, **kwargs):
    pytest.fail("Catalog-only estimation must not download config.json")


@pytest.mark.parametrize(
    "spec_type,model_format",
    [
        (PytorchLLMSpecV2, "pytorch"),
        (LlamaCppLLMSpecV2, "ggufv2"),
        (MLXLLMSpecV2, "mlx"),
    ],
)
def test_metadata_survives_flattening_and_roundtrip(config, spec_type, model_format):
    raw = dict(
        model_format=model_format,
        model_size_in_billions=7,
        model_file_name_template="model-{quantization}.gguf",
        memory_estimation=config,
        model_src={
            "huggingface": {
                "model_id": "test/model",
                "quantizations": ["none", "4-bit"],
            },
            "modelscope": {"model_id": "test/model", "quantizations": ["none"]},
        },
    )
    records = flatten_quantizations(raw)
    assert len(records) == 3
    for record in records:
        spec = spec_type.parse_obj(record)
        assert spec.memory_estimation.dict(exclude_none=True) == config
        assert (
            spec_type.parse_raw(spec.json()).memory_estimation == spec.memory_estimation
        )
        del record["memory_estimation"]
        assert spec_type.parse_obj(record).memory_estimation is None


@pytest.mark.parametrize("allow_download", [True, False])
def test_catalog_metadata_precedes_download(
    config, monkeypatch, tmp_path, allow_download
):
    spec = PytorchLLMSpecV2(
        model_format="pytorch",
        model_size_in_billions=7,
        quantization="none",
        memory_estimation=config,
    )
    monkeypatch.setattr(
        "xinference.model.llm.match_llm",
        lambda **kw: SimpleNamespace(model_specs=[spec]),
    )
    monkeypatch.setattr(llm_family, "cache_model_config", no_download)
    info = get_model_layers_info(
        7, "test", "pytorch", "none", allow_download=allow_download
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert info == load_model_config_json(str(path))
    result = estimate_llm_gpu_memory(
        7, "none", 2048, "pytorch", "test", allow_download=allow_download
    )
    assert result == estimate_llm_gpu_memory(
        7, None, 2048, "pytorch", "test", allow_download=allow_download
    )
    assert result.total > 0


@pytest.mark.parametrize("model_name", [None, "missing", "legacy"])
def test_offline_missing_metadata_is_unknown(monkeypatch, model_name):
    spec = PytorchLLMSpecV2(
        model_format="pytorch", model_size_in_billions=7, quantization="none"
    )
    monkeypatch.setattr(
        "xinference.model.llm.match_llm",
        lambda **kw: (
            None
            if kw["model_name"] == "missing"
            else SimpleNamespace(model_specs=[spec])
        ),
    )
    monkeypatch.setattr(llm_family, "cache_model_config", no_download)
    assert (
        estimate_llm_gpu_memory(
            7, None, 2048, "pytorch", model_name, allow_download=False
        )
        is None
    )


def test_legacy_download_fallback(config, monkeypatch, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    calls = []

    def cached_config(family):
        calls.append(family)
        return str(path)

    spec = PytorchLLMSpecV2(
        model_format="pytorch", model_size_in_billions=7, quantization="none"
    )
    monkeypatch.setattr(
        "xinference.model.llm.match_llm",
        lambda **kw: SimpleNamespace(model_specs=[spec]),
    )
    monkeypatch.setattr(llm_family, "cache_model_config", cached_config)
    assert get_model_layers_info(
        7, "legacy", "pytorch", None
    ) == load_model_config_json(str(path))
    assert len(calls) == 1
    assert (
        estimate_llm_gpu_memory("1_8", None, 2048, "pytorch", kv_cache_dtype=32).total
        == 5162
    )


@pytest.mark.parametrize(
    "field",
    [
        "vocab_size",
        "num_attention_heads",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
    ],
)
def test_missing_or_invalid_dimensions_rejected(config, field):
    config[field] = 0
    with pytest.raises(ValidationError):
        ModelMemoryMetadata.from_config(config)
    del config[field]
    with pytest.raises(ValidationError):
        ModelMemoryMetadata.from_config(config)


def test_aliases_and_no_kv_heads():
    metadata = ModelMemoryMetadata.from_config(
        dict(vocab_size=32000, n_head=32, n_embd=4096, n_inner=14336, n_layer=32)
    )
    assert metadata.hidden_size == 4096
    assert metadata.num_attention_heads == 32
    assert metadata.num_key_value_heads is None


def test_explicit_head_dim_is_preserved(config):
    config["head_dim"] = 128
    config["hidden_size"] = 2560
    metadata = ModelMemoryMetadata.from_config(config)
    assert metadata.head_dim == 128
    assert metadata.head_dim != metadata.hidden_size // metadata.num_attention_heads


def test_bundled_memory_metadata(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(llm_family, "cache_model_config", no_download)
    for helper in (
        "download_from_modelscope",
        "download_from_openmind_hub",
        "download_from_csghub",
    ):
        monkeypatch.setattr(llm_family, helper, lambda: False)
    catalogs = load_model_catalog(str(Path(llm_family.__file__).parent / "models"))
    count = 0
    for family in catalogs:
        for spec in family["model_specs"]:
            for hub, source in spec["model_src"].items():
                for quantization in source["quantizations"]:
                    raw = source.get("memory_estimation_by_quantization", {}).get(
                        quantization, source.get("memory_estimation")
                    )
                    if raw is None:
                        continue
                    count += 1
                    metadata = ModelMemoryMetadata.parse_obj(raw)
                    from urllib.parse import unquote

                    assert source["model_id"].replace(
                        "{quantization}", quantization
                    ) in unquote(metadata.config_source)
                    if metadata.config_sha256:
                        assert re.fullmatch(r"[0-9a-f]{64}", metadata.config_sha256)
                    if metadata.unsupported_reason:
                        continue
                    from ..memory import (
                        ModelLayersInfo,
                        estimate_llm_gpu_memory_details,
                    )

                    try:
                        estimate = estimate_llm_gpu_memory_details(
                            ModelLayersInfo.from_metadata(metadata),
                            llm_family.convert_model_size_to_float(
                                spec["model_size_in_billions"]
                            ),
                            quantization,
                            2048,
                            spec["model_format"],
                        )
                    except (KeyError, ValueError):
                        # Mixed/exotic quantization layouts need separate formulas.
                        continue
                    assert estimate.total > estimate.model_mem > 0
    assert count >= 40


def test_per_quantization_metadata_never_leaks(config):
    spec = dict(
        model_format="mlx",
        model_size_in_billions=8,
        model_src={
            "huggingface": dict(
                model_id="test/model-{quantization}",
                quantizations=["4bit", "8bit"],
                memory_estimation=config,
                memory_estimation_by_quantization={
                    "4bit": dict(config, hidden_size=1024)
                },
            )
        },
    )
    records = flatten_quantizations(spec)
    assert records[0]["memory_estimation"]["hidden_size"] == 1024
    assert records[1]["memory_estimation"] is None
    assert "memory_estimation_by_quantization" not in records[0]


@pytest.mark.parametrize(
    "extra,reason",
    [
        ({"num_experts": 8}, "moe"),
        ({"kv_lora_rank": 512}, "mla"),
        ({"layer_types": ["linear_attention", "full_attention"]}, "hybrid_attention"),
    ],
)
def test_unsupported_architecture_is_not_a_dense_estimate(config, extra, reason):
    from ..memory import ModelLayersInfo

    metadata = ModelMemoryMetadata.from_config(dict(config, **extra))
    assert metadata.unsupported_reason == reason
    with pytest.raises(ValueError, match="Unsupported memory architecture"):
        ModelLayersInfo.from_metadata(metadata)


def test_multimodal_metadata_is_collected_but_not_estimated(config):
    metadata = ModelMemoryMetadata.from_config({"text_config": config})
    assert metadata.hidden_size == config["hidden_size"]
    assert metadata.unsupported_reason == "multimodal"


def test_real_matching_keeps_metadata_per_size(config, monkeypatch):
    family = llm_family.LLMFamilyV2(
        version=2,
        model_name="offline-test-model",
        model_lang=["en"],
        model_ability=["generate"],
        model_specs=[
            dict(
                model_format="pytorch",
                model_size_in_billions=size,
                quantization="none",
                memory_estimation=dict(config, num_hidden_layers=layers),
            )
            for size, layers in [(7, 32), (14, 48)]
        ],
    )
    monkeypatch.setattr(llm_family, "BUILTIN_LLM_FAMILIES", [family])
    monkeypatch.setattr(llm_family, "cache_model_config", no_download)
    for size, layers in [(7, 32), (14, 48)]:
        info = get_model_layers_info(
            size, family.model_name, "pytorch", "none", allow_download=False
        )
        assert info.num_layers == layers
    assert (
        get_model_layers_info(
            72, family.model_name, "pytorch", "none", allow_download=False
        )
        is None
    )


def test_local_extraction_command(config, tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr("sys.argv", ["memory_metadata", str(path)])
    monkeypatch.setattr(llm_family, "cache_model_config", no_download)
    main()
    assert json.loads(capsys.readouterr().out) == config
