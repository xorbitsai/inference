# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import ctypes
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ..compat import initialize_worker, prepare_runtime


def test_nvml_query_passes_structure_by_pointer_and_preserves_cc_state(monkeypatch):
    class Settings(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint), ("cc_feature", ctypes.c_uint)]

    def query(pointer):
        assert pointer.contents.version == 1
        pointer.contents.cc_feature = 1
        return 0

    function = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(Settings))(query)
    function.argtypes = None
    nvml = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        _nvmlGetFunctionPointer=Mock(return_value=function),
        c_nvmlSystemConfComputeSettings_v1_t=Settings,
        NVMLError_FunctionNotFound=LookupError,
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    initialize_worker()
    settings = Settings(version=1)
    assert function(settings) == 0
    assert settings.cc_feature == 1
    nvml.nvmlInit.assert_called_once()
    nvml.nvmlShutdown.assert_called_once()


def test_old_driver_without_nvml_query_still_releases_nvml(monkeypatch):
    nvml = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        _nvmlGetFunctionPointer=Mock(side_effect=LookupError),
        NVMLError_FunctionNotFound=LookupError,
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    initialize_worker()
    nvml.nvmlShutdown.assert_called_once()


def test_runtime_uses_the_existing_model_process(monkeypatch):
    initialize = Mock()
    monkeypatch.setattr(
        "xinference.model.llm.trt_llm.compat.initialize_worker", initialize
    )
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)
    prepare_runtime({})
    import os

    assert os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] == "1"
    initialize.assert_called_once()


@pytest.mark.parametrize(
    "name", ["tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size"]
)
def test_parallel_configs_fail_before_runtime_initialization(monkeypatch, name):
    initialize = Mock()
    monkeypatch.setattr(
        "xinference.model.llm.trt_llm.compat.initialize_worker", initialize
    )
    with pytest.raises(ValueError, match="TP=PP=CP=1"):
        prepare_runtime({name: 2})
    initialize.assert_not_called()


def test_fp16_fails_before_runtime_initialization(monkeypatch):
    initialize = Mock()
    monkeypatch.setattr(
        "xinference.model.llm.trt_llm.compat.initialize_worker", initialize
    )
    with pytest.raises(ValueError, match="requires BF16"):
        prepare_runtime({"dtype": "float16"})
    initialize.assert_not_called()
