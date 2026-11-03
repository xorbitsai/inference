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

# CI: linux

import sys
from types import SimpleNamespace

import pytest

from ..llm_family import match_llm
from ..vllm import core as vllm_core
from ..weight_cache import WeightCachedModel


@pytest.fixture(
    params=[vllm_core.VLLMModel, vllm_core.VLLMChatModel, vllm_core.VLLMMultiModel]
)
def engine_cls(request, monkeypatch):
    monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", False)
    monkeypatch.setattr(vllm_core, "_virtual_env_allows_missing_vllm", lambda: True)
    monkeypatch.setattr(vllm_core.VLLMModel, "_has_cuda_device", lambda: True)
    monkeypatch.setattr(vllm_core.VLLMModel, "_is_linux", lambda: True)
    return request.param


@pytest.fixture
def offline_family(engine_cls):
    family = match_llm("qwen3.8", "pytorch", 27, "Int4", "huggingface").copy(deep=True)
    family.model_name = family.model_family = "unlisted-offline-checkpoint"
    spec = family.model_specs[0]
    spec.model_id = "example/arbitrary-checkpoint"
    spec.model_size_in_billions = 13
    if engine_cls is not vllm_core.VLLMMultiModel:
        family.architectures = ["LlamaForCausalLM"]
        family.model_ability = ["generate", "chat"]
    return family


@pytest.mark.parametrize("precision", ["Int4", "Int8"])
def test_declared_quantizer_is_independent_of_model_identity(
    engine_cls, offline_family, precision
):
    spec = offline_family.model_specs[0]
    spec.quantization = precision
    assert engine_cls.match_json(offline_family, spec, precision) is True


@pytest.mark.parametrize("method", [None, "unknown", "awq"])
def test_runtime_or_unsupported_quantization_remains_rejected(
    engine_cls, offline_family, method
):
    spec = offline_family.model_specs[0]
    spec.quantization_method = method
    result = engine_cls.match_json(offline_family, spec, "Int4")
    assert result[0] is False
    assert "pytorch format with quantization" in result[1]


def test_requested_precision_must_match_checkpoint(engine_cls, offline_family):
    result = engine_cls.match_json(
        offline_family, offline_family.model_specs[0], "8-bit"
    )
    assert result[0] is False
    assert "pytorch format with quantization" in result[1]


def test_unquantized_pytorch_remains_supported(engine_cls, offline_family):
    spec = offline_family.model_specs[0]
    spec.quantization = "none"
    spec.quantization_method = None
    assert engine_cls.match_json(offline_family, spec, "none") is True


@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
@pytest.mark.parametrize(
    "name,size",
    [
        ("qwen3.8", 27),
        ("MiniCPM-V-4.7", 35),
        ("Ling-3.0-flash", 124),
        ("Ling-3.0-tiny", "7_9"),
    ],
)
def test_existing_offline_checkpoints_declare_quantizer(hub, name, size):
    family = match_llm(name, "pytorch", size, "Int4", hub)
    assert family.model_specs[0].quantization_method == "compressed-tensors"
    plain = match_llm(name, "pytorch", size, "none", hub)
    assert plain.model_specs[0].quantization_method is None


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("option", [None, True, False])
def test_weight_cache_uses_loader_capability(
    engine_cls, offline_family, monkeypatch, supported, option
):
    monkeypatch.setattr(
        WeightCachedModel, "_weight_cache_unavailable_reason", lambda self: None
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        SimpleNamespace(
            CompressedTensorsLinearMethod=SimpleNamespace(
                supports_pre_processed_weights=supported
            )
        ),
    )
    model = object.__new__(engine_cls)
    model.model_spec = offline_family.model_specs[0]
    model._weight_cache_option = option
    if option is True and not supported:
        with pytest.raises(ValueError, match="preprocessed IPC weights"):
            model._resolve_weight_cache()
    else:
        model._resolve_weight_cache()
        assert model._enable_weight_cache is (supported and option is not False)


def test_weight_cache_keeps_other_unavailability_reasons(
    engine_cls, offline_family, monkeypatch
):
    monkeypatch.setattr(
        WeightCachedModel,
        "_weight_cache_unavailable_reason",
        lambda self: "unsupported engine version",
    )
    model = object.__new__(engine_cls)
    model.model_spec = offline_family.model_specs[0]
    assert model._weight_cache_unavailable_reason() == "unsupported engine version"


def test_weight_cache_does_not_import_compressed_tensors_for_other_checkpoints(
    engine_cls, offline_family, monkeypatch
):
    monkeypatch.setattr(
        WeightCachedModel, "_weight_cache_unavailable_reason", lambda self: None
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        None,
    )
    model = object.__new__(engine_cls)
    model.model_spec = offline_family.model_specs[0]
    model.model_spec.quantization_method = None
    assert model._weight_cache_unavailable_reason() is None
