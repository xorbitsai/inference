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

import pytest

from ....core.utils import filter_virtualenv_packages_by_markers
from ....core.virtual_env_manager import expand_engine_dependency_placeholders
from ...utils import get_engine_params_by_name_with_virtual_env
from ..llm_family import check_engine_by_spec_parameters_with_virtual_env, match_llm
from ..sglang import core as sglang_core
from ..transformers.ling3 import Ling3PytorchChatModel
from ..vllm import core as vllm_core

VARIANTS = [
    ("pytorch", "none", ""),
    ("fp8", "FP8", "-fp8"),
    ("pytorch", "Int4", "-int4"),
]


@pytest.fixture
def virtualenv_worker(monkeypatch):
    for core, model_cls, installed_flag, virtualenv_check in (
        (
            vllm_core,
            vllm_core.VLLMModel,
            "VLLM_INSTALLED",
            "_virtual_env_allows_missing_vllm",
        ),
        (
            sglang_core,
            sglang_core.SGLANGModel,
            "SGLANG_INSTALLED",
            "_virtual_env_allows_missing_sglang",
        ),
    ):
        monkeypatch.setattr(core, installed_flag, False)
        monkeypatch.setattr(core, virtualenv_check, lambda: True)
        monkeypatch.setattr(model_cls, "_is_linux", classmethod(lambda cls: True))
        monkeypatch.setattr(
            model_cls, "_has_cuda_device", classmethod(lambda cls: True)
        )


@pytest.mark.parametrize(
    "hub,revision", [("huggingface", "main"), ("modelscope", "master")]
)
@pytest.mark.parametrize("model_format,quantization,suffix", VARIANTS)
def test_tiny_official_checkpoints_and_virtualenv_launch(
    virtualenv_worker, hub, revision, model_format, quantization, suffix
):
    family = match_llm("Ling-3.0-tiny", model_format, "7_9", quantization, hub)
    spec = family.model_specs[0]
    assert spec.model_id == f"inclusionAI/Ling-3.0-tiny{suffix}"
    assert spec.model_revision == revision
    assert spec.model_hub == hub
    assert family.context_length == 131072
    assert Ling3PytorchChatModel.match_json(family, spec, quantization) is True

    for engine, engine_cls in (
        ("vLLM", vllm_core.VLLMChatModel),
        ("SGLang", sglang_core.SGLANGChatModel),
    ):
        assert engine_cls.match_json(family, spec, quantization) is True
        assert (
            check_engine_by_spec_parameters_with_virtual_env(
                engine,
                family.model_name,
                model_format,
                "7_9",
                quantization,
                llm_family=family,
            )
            is engine_cls
        )


def test_tiny_virtualenv_engine_discovery_without_installed_libraries(
    virtualenv_worker,
):
    params = get_engine_params_by_name_with_virtual_env(
        "LLM", "Ling-3.0-tiny", enable_virtual_env=True
    )
    assert params is not None
    for engine in ("vLLM", "SGLang"):
        engine_params = params[engine]
        assert isinstance(engine_params, list)
        assert {
            (entry["model_format"], quantization)
            for entry in engine_params
            for quantization in entry["quantizations"]
        } == {
            (model_format, quantization) for model_format, quantization, _ in VARIANTS
        }


@pytest.mark.parametrize(
    "engine,requirement",
    [
        ("Transformers", "transformers>=4.57.1,<5.0.0"),
        ("vllm", "vllm==0.29.0"),
        ("sglang", "sglang==0.5.19"),
    ],
)
def test_tiny_virtualenv_install_requirements(engine, requirement):
    family = match_llm("Ling-3.0-tiny", "pytorch", "7_9", "none", "huggingface")
    expanded = expand_engine_dependency_placeholders(family.virtualenv.packages, engine)
    prepared = filter_virtualenv_packages_by_markers(expanded, engine, None)
    assert requirement in prepared
    if engine == "sglang":
        assert {"numpy<2.3", "pandas<3"} <= set(prepared)
    elif engine == "vllm":
        assert "#system_numpy#" in prepared
