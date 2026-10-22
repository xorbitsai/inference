# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import json
import sys

import pytest

from .....core.utils import filter_virtualenv_packages_by_markers
from .....core.virtual_env_manager import get_engine_virtualenv_packages
from ... import generate_engine_config_by_model_family
from ...llm_family import check_engine_by_spec_parameters_with_virtual_env, match_llm
from ..core import TRTLLMModel


def test_qwen3_virtualenv_selects_trt_without_installed_engine(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    family = match_llm("qwen3", "pytorch", "0_6", "none")
    assert (
        check_engine_by_spec_parameters_with_virtual_env(
            "TRT-LLM", "qwen3", "pytorch", "0_6", "none", family
        )
        is TRTLLMModel
    )
    packages = get_engine_virtualenv_packages("TRT-LLM")
    packages = filter_virtualenv_packages_by_markers(packages, "TRT-LLM", None)
    requirements = {package.split(";", 1)[0].strip() for package in packages}
    assert {
        "tensorrt_llm==1.2.1",
        "pyarrow==20.0.0",
        "xoscar[nixl]>=0.11.1",
    } <= requirements
    assert not any("sglang" in package or "vllm" in package for package in packages)


@pytest.mark.parametrize(
    "name,model_format,size,quantization",
    [
        ("qwen3", "pytorch", 30, "none"),
        ("qwen3", "fp8", "0_6", "fp8"),
        ("Qwen3-Instruct", "pytorch", 4, "none"),
        ("Qwen3-Thinking", "fp8", 4, "none"),
        ("qwen2.5-instruct", "pytorch", "0_5", "none"),
        ("qwen2.5-coder-instruct", "pytorch", "1_5", "none"),
        ("llama-3.1-instruct", "pytorch", 8, "none"),
        ("phi-3-mini-4k-instruct", "pytorch", 4, "none"),
        ("gpt-oss", "pytorch", 20, "none"),
        ("DeepSeek-V3.2", "pytorch", 671, "none"),
        ("Qwen3-Next-Instruct", "pytorch", 80, "none"),
    ],
)
def test_architecture_support_is_available_in_virtualenv_without_engine(
    monkeypatch, name, model_format, size, quantization
):
    monkeypatch.setattr(sys, "platform", "linux")
    family = match_llm(name, model_format, size, quantization).copy(deep=True)
    generate_engine_config_by_model_family(family, {})
    assert (
        check_engine_by_spec_parameters_with_virtual_env(
            "TRT-LLM", name, model_format, size, quantization, family
        )
        is TRTLLMModel
    )
    packages = get_engine_virtualenv_packages("TRT-LLM")
    packages = filter_virtualenv_packages_by_markers(packages, "TRT-LLM", None)
    assert any(package.startswith("tensorrt_llm==1.2.1") for package in packages)


@pytest.mark.parametrize(
    "model_format,size,quantization",
    [("awq", 4, "Int4"), ("gptq", "0_6", "Int8")],
)
def test_virtualenv_marker_does_not_bypass_trt_model_guards(
    monkeypatch, model_format, size, quantization
):
    monkeypatch.setattr(sys, "platform", "linux")
    family = match_llm("qwen3", model_format, size, quantization)
    assert family is not None
    with pytest.raises(ValueError, match="cannot be run"):
        check_engine_by_spec_parameters_with_virtual_env(
            "TRT-LLM", "qwen3", model_format, size, quantization, family
        )


def test_custom_derivatives_use_architecture_and_preserve_shared_settings(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    base = match_llm("qwen2.5-instruct", "pytorch", "0_5", "none")
    family = base.copy(update={"model_name": "custom-derivative"})
    before = list(base.virtualenv.packages)
    generate_engine_config_by_model_family(family, {})
    generate_engine_config_by_model_family(family, {})
    assert base.virtualenv.packages == before
    assert family.virtualenv.packages == before
    assert TRTLLMModel.match_json(family, family.model_specs[0], "none") is True


@pytest.mark.parametrize(
    "packages,transformers_allowed",
    [
        (None, True),
        ([], True),
        (["sentencepiece"], True),
        (['#transformers_dependencies# ; #engine# == "Transformers"'], True),
        (['#vllm_dependencies# ; #engine# == "vllm"'], False),
        (['#trt_llm_dependencies# ; #engine# == "TRT-LLM"'], False),
    ],
)
def test_custom_registration_preserves_engine_eligibility(
    monkeypatch, tmp_path, packages, transformers_allowed
):
    from ..... import constants
    from ....core import VirtualEnvSettings
    from ....custom import RegistryManager
    from ....utils import get_engine_params_by_name_with_virtual_env
    from ...custom import register_llm
    from ...llm_family import CustomLLMFamilyV2
    from ...transformers.core import PytorchChatModel

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(PytorchChatModel, "check_lib", classmethod(lambda cls: True))
    monkeypatch.setattr(TRTLLMModel, "check_lib", classmethod(lambda cls: False))
    base = match_llm("qwen2.5-instruct", "pytorch", "0_5", "none")
    data = base.dict()
    data.update(
        model_name="trt-review-unmarked-custom",
        model_family=base.model_name,
        is_builtin=False,
        architectures=None,
        model_specs=[base.model_specs[0].dict()],
        virtualenv=(
            VirtualEnvSettings(packages=packages).dict()
            if packages is not None
            else None
        ),
    )
    family = CustomLLMFamilyV2.parse_obj(data)
    spec = family.model_specs[0]
    original = json.loads(family.json())
    register_llm(family, persist=True)
    try:
        stored = tmp_path / "v2" / "llm" / f"{family.model_name}.json"
        assert json.loads(stored.read_text()) == original
        assert json.loads(family.json()) == original
        args = (
            family.model_name,
            spec.model_format,
            spec.model_size_in_billions,
            spec.quantization,
            family,
        )
        if transformers_allowed:
            assert (
                check_engine_by_spec_parameters_with_virtual_env("Transformers", *args)
                is PytorchChatModel
            )
        else:
            with pytest.raises(ValueError, match="not listed in virtualenv packages"):
                check_engine_by_spec_parameters_with_virtual_env("Transformers", *args)
        assert (
            check_engine_by_spec_parameters_with_virtual_env("TRT-LLM", *args)
            is TRTLLMModel
        )
        # Discover the genuinely registered custom family without substituting
        # a built-in family or modifying the builtin registry.
        params = get_engine_params_by_name_with_virtual_env(
            "LLM", family.model_name, enable_virtual_env=True
        )
        assert isinstance(params["Transformers"], list) == transformers_allowed
        assert isinstance(params["TRT-LLM"], list)
        assert all(item["virtualenv_required"] for item in params["TRT-LLM"])
    finally:
        RegistryManager.get_registry("llm").unregister(
            family.model_name, remove_file=False
        )


@pytest.mark.parametrize(
    "architecture,abilities,model_format,quantization,expected",
    [
        ("LlamaForCausalLM", ["chat"], "fp4", "FP4", True),
        ("Qwen3ForCausalLM", ["chat"], "fp8", "FP8", True),
        ("Qwen3_5ForConditionalGeneration", ["chat"], "pytorch", "none", False),
        ("Qwen3ForCausalLM", ["chat", "vision"], "pytorch", "none", False),
        ("Qwen2ForCausalLM", ["chat", "audio"], "pytorch", "none", False),
        ("LlamaForCausalLM", ["chat"], "bnb", "none", False),
        ("LlamaForCausalLM", ["chat"], "gptq", "Int4", False),
        ("LlamaForCausalLM", ["chat"], "awq", "Int4", False),
        ("LlamaForCausalLM", ["chat"], "mlx", "4bit", False),
        ("LlamaForCausalLM", ["chat"], "ggufv2", "Q4_K_M", False),
    ],
)
def test_architecture_and_quantization_matching(
    monkeypatch, architecture, abilities, model_format, quantization, expected
):
    from types import SimpleNamespace

    monkeypatch.setattr(sys, "platform", "linux")
    family = match_llm("qwen3", "pytorch", "0_6", "none").copy(
        update={"architectures": [architecture], "model_ability": abilities}
    )
    result = TRTLLMModel.match_json(
        family, SimpleNamespace(model_format=model_format), quantization
    )
    assert (result is True) == expected
    if model_format in {"bnb", "gptq", "awq", "mlx", "ggufv2"}:
        assert f"{model_format}/{quantization}" in result[1]
        assert ("use ModelOpt exports for AWQ" in result[1]) == (model_format == "awq")
