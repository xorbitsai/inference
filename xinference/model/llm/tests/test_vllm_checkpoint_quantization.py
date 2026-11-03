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
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from types import SimpleNamespace

import pytest
from packaging.version import Version

from ...utils import flatten_model_src, flatten_quantizations
from ..llm_family import PytorchLLMSpecV2, match_llm
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


@pytest.mark.parametrize("precision", ["Int4", "4-bit", "8-bit"])
def test_runtime_quantization_requires_checkpoint_metadata(
    engine_cls, offline_family, precision
):
    spec = offline_family.model_specs[0]
    spec.quantization = precision
    spec.quantization_method = None
    result = engine_cls.match_json(offline_family, spec, spec.quantization)
    assert result[0] is False
    assert "quantization_method" in result[1]


@pytest.mark.parametrize("method", ["", "unknown", "awq", "compressed_tensors"])
def test_quantization_method_validation(method):
    with pytest.raises(ValueError, match="quantization_method"):
        PytorchLLMSpecV2(
            model_format="pytorch",
            model_size_in_billions=13,
            quantization="Int4",
            quantization_method=method,
        )


def test_source_quantization_method_survives_flattening():
    spec = {
        "model_format": "pytorch",
        "model_size_in_billions": 13,
        "model_src": {
            "huggingface": {
                "model_id": "example/int4",
                "quantizations": ["Int4"],
                "quantization_method": "compressed-tensors",
            },
            "modelscope": {"model_id": "example/plain", "quantizations": ["none"]},
        },
    }
    sources = {s["model_hub"]: s for s in flatten_model_src({"model_specs": [spec]})}
    assert sources["huggingface"]["quantization_method"] == "compressed-tensors"
    assert "quantization_method" not in sources["modelscope"]
    runtime_specs = {
        s["model_hub"]: PytorchLLMSpecV2.parse_obj(s)
        for s in flatten_quantizations(spec)
    }
    assert runtime_specs["huggingface"].quantization_method == "compressed-tensors"
    assert runtime_specs["modelscope"].quantization_method is None


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


@pytest.fixture
def ipc_methods(monkeypatch):
    linear = SimpleNamespace(supports_pre_processed_weights=True)
    moe = SimpleNamespace(supports_pre_processed_weights=True)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        SimpleNamespace(CompressedTensorsLinearMethod=linear),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe",
        SimpleNamespace(CompressedTensorsMoEMethod=moe),
    )
    return linear, moe


@pytest.fixture
def cached_model(engine_cls, offline_family, monkeypatch):
    monkeypatch.setattr(
        WeightCachedModel, "_weight_cache_unavailable_reason", lambda self: None
    )
    model = object.__new__(engine_cls)
    model.model_spec = offline_family.model_specs[0]
    model._weight_cache_option = None
    return model


@pytest.mark.parametrize("linear_supported", [False, True])
@pytest.mark.parametrize("moe_supported", [False, True])
@pytest.mark.parametrize("option", [None, True, False])
def test_weight_cache_uses_loader_capability(
    cached_model, ipc_methods, linear_supported, moe_supported, option
):
    linear, moe = ipc_methods
    linear.supports_pre_processed_weights = linear_supported
    moe.supports_pre_processed_weights = moe_supported
    model = cached_model
    model._weight_cache_option = option
    supported = linear_supported and moe_supported
    if option is True and not supported:
        with pytest.raises(ValueError, match="preprocessed IPC weights"):
            model._resolve_weight_cache()
    else:
        model._resolve_weight_cache()
        assert model._enable_weight_cache is (supported and option is not False)


@pytest.mark.parametrize("method_index", [0, 1])
@pytest.mark.parametrize("option", [None, True])
def test_weight_cache_missing_capability_falls_back(
    cached_model, ipc_methods, method_index, option
):
    delattr(ipc_methods[method_index], "supports_pre_processed_weights")
    cached_model._weight_cache_option = option
    if option is True:
        with pytest.raises(ValueError, match="preprocessed IPC weights"):
            cached_model._resolve_weight_cache()
    else:
        cached_model._resolve_weight_cache()
        assert cached_model._enable_weight_cache is False


@pytest.mark.parametrize(
    "module_name",
    [
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe",
    ],
)
@pytest.mark.parametrize("option", [None, True])
def test_weight_cache_import_failure_falls_back(
    cached_model, ipc_methods, monkeypatch, module_name, option
):
    monkeypatch.setitem(sys.modules, module_name, None)
    cached_model._weight_cache_option = option
    if option is True:
        with pytest.raises(ValueError, match="Cannot verify.*IPC"):
            cached_model._resolve_weight_cache()
    else:
        cached_model._resolve_weight_cache()
        assert cached_model._enable_weight_cache is False


def test_installed_vllm_compressed_tensors_ipc_api():
    pytest.importorskip("vllm")
    try:
        installed = Version(package_version("vllm"))
    except PackageNotFoundError:
        pytest.skip("A real vLLM distribution is required for this API check")
    if installed < Version("0.31.0"):
        pytest.skip("vLLM IPC weight caching requires vLLM >= 0.31.0")
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsLinearMethod,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsMoEMethod,
    )

    for method in (CompressedTensorsLinearMethod, CompressedTensorsMoEMethod):
        assert isinstance(method.supports_pre_processed_weights, bool)


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
