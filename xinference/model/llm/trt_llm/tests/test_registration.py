# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import sys

import pytest

from .....core.utils import filter_virtualenv_packages_by_markers
from .....core.virtual_env_manager import expand_engine_dependency_placeholders
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
    packages = expand_engine_dependency_placeholders(
        family.virtualenv.packages, "TRT-LLM"
    )
    packages = filter_virtualenv_packages_by_markers(packages, "TRT-LLM", None)
    requirements = {package.split(";", 1)[0].strip() for package in packages}
    assert {"tensorrt_llm==1.2.1", "pyarrow==20.0.0"} <= requirements
    assert not any("sglang" in package or "vllm" in package for package in packages)


@pytest.mark.parametrize(
    "model_format,size,quantization",
    [("fp8", "0_6", "fp8"), ("pytorch", 30, "none")],
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
