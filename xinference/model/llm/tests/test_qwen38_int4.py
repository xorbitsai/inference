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

from unittest.mock import Mock

import pytest
from packaging.version import Version

from ....core.utils import filter_virtualenv_packages_by_markers
from ....core.virtual_env_manager import expand_engine_dependency_placeholders
from ..llm_family import check_engine_by_spec_parameters_with_virtual_env, match_llm
from ..vllm import core as vllm_core


@pytest.fixture(params=["huggingface", "modelscope"])
def int4_family(request):
    return match_llm("qwen3.8", "pytorch", 27, "Int4", request.param).copy(deep=True)


@pytest.fixture
def vllm_worker(monkeypatch):
    monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", False)
    monkeypatch.setattr(vllm_core, "_virtual_env_allows_missing_vllm", lambda: True)
    monkeypatch.setattr(vllm_core.VLLMMultiModel, "_has_cuda_device", lambda: True)
    monkeypatch.setattr(vllm_core.VLLMMultiModel, "_is_linux", lambda: True)


def test_qwen38_int4_catalog_and_engine_selection(int4_family, vllm_worker):
    spec = int4_family.model_specs[0]
    assert spec.model_id == "RedHatAI/Qwen3.8-27B-INT4"
    assert spec.quantization_method == "compressed-tensors"
    assert (
        spec.model_revision
        == {
            "huggingface": "91bd022d5b49442a868bc35008f6c21e1860edfa",
            "modelscope": "20f1f091755dab5ff431fc545bc4245caf31f722",
        }[spec.model_hub]
    )
    assert int4_family.context_length == 262144
    assert {"vision", "tools", "reasoning"} <= set(int4_family.model_ability)
    assert spec.model_metadata.architecture_type == "hybrid_attention"
    assert spec.model_metadata.attention_layers().count("full_attention") == 16
    assert (
        check_engine_by_spec_parameters_with_virtual_env(
            "vLLM", "qwen3.8", "pytorch", 27, "Int4", llm_family=int4_family
        )
        is vllm_core.VLLMMultiModel
    )


def test_qwen38_int4_requires_quantization_metadata(int4_family, vllm_worker):
    int4_family.model_specs[0].quantization_method = None
    result = vllm_core.VLLMMultiModel.match_json(
        int4_family, int4_family.model_specs[0], "Int4"
    )
    assert result[0] is False
    assert "pytorch format with quantization" in result[1]


@pytest.mark.parametrize(
    "engine", ["Transformers", "vllm", "sglang", "MLX", "llama.cpp"]
)
def test_qwen38_checkpoint_quantizer_dependency(int4_family, engine):
    packages = expand_engine_dependency_placeholders(
        int4_family.virtualenv.packages, engine
    )
    packages = filter_virtualenv_packages_by_markers(packages, engine, None)
    expected = ["compressed-tensors>=0.15.0"] if engine == "Transformers" else []
    assert [p for p in packages if p.startswith("compressed-tensors")] == expected


@pytest.mark.parametrize("engine_version", ["0.21.0", "0.32.0"])
def test_qwen38_int4_preserves_checkpoint_quantizer(
    int4_family, monkeypatch, tmp_path, engine_version
):
    monkeypatch.setattr(vllm_core, "VLLM_VERSION", Version(engine_version))
    model = object.__new__(vllm_core.VLLMMultiModel)
    model.model_family = int4_family
    model.model_spec = int4_family.model_specs[0]
    model.model_path = str(tmp_path)
    model._device_count = model._n_worker = 1
    model._apply_draft_model = Mock()
    config = model._sanitize_model_config({"max_model_len": 131072})
    assert config["quantization"] is None
    assert config["max_model_len"] == 131072


@pytest.mark.parametrize("has_gpu,is_linux", [(False, True), (True, False)])
def test_qwen38_int4_preserves_platform_guards(
    int4_family, vllm_worker, monkeypatch, has_gpu, is_linux
):
    monkeypatch.setattr(vllm_core.VLLMMultiModel, "_has_cuda_device", lambda: has_gpu)
    monkeypatch.setattr(vllm_core.VLLMMultiModel, "_is_linux", lambda: is_linux)
    assert (
        vllm_core.VLLMMultiModel.match_json(
            int4_family, int4_family.model_specs[0], "Int4"
        )[0]
        is False
    )
