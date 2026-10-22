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


@pytest.mark.parametrize("dtype", ["auto", "float16", "bfloat16"])
def test_ordinary_runtime_leaves_dtype_validation_to_engine(monkeypatch, dtype):
    initialize = Mock()
    monkeypatch.setattr(
        "xinference.model.llm.trt_llm.compat.initialize_worker", initialize
    )
    prepare_runtime({"dtype": dtype})
    initialize.assert_called_once()


@pytest.mark.parametrize("own_pd", [False, True])
def test_pd_compat_preserves_other_hooks_and_retires_failed_loads(monkeypatch, own_pd):
    import torch

    from ..compat import initialize_pd_connector

    # Import the connector through the regular module name used by the runtime.
    class Worker:
        def __init__(self):
            self.client = SimpleNamespace(pd_gpu=own_pd)
            self.load_failed = None

    monkeypatch.setitem(
        sys.modules,
        "xinference.model.llm.trt_llm.connector",
        SimpleNamespace(XavierConnectorWorker=Worker),
    )
    layer = torch.nn.Linear(2, 2)
    preserved = layer.register_forward_hook(lambda *args: None)
    request = SimpleNamespace(request_id=7)

    def pending():
        return SimpleNamespace(loading={7: request}, saving={8: "source"})

    build = Mock(return_value="output")
    finished = Mock(return_value=True)
    manager = SimpleNamespace(
        worker=Worker(),
        scheduler_output_manager=SimpleNamespace(build_scheduler_output=build),
        request_finished=finished,
        new_async_requests=pending(),
        pending_async_requests=pending(),
        local_finished_async_requests=pending(),
        finished_async_loading_requests={7: request},
    )

    # Bound methods are what Torch stores for the real manager's layer hooks.
    class Hooks:
        def layer_pre_hook(self, *args):
            pass

        def layer_post_hook(self, *args):
            pass

    manager_class = type("Manager", (Hooks,), {})
    bound = manager_class()
    bound.__dict__.update(manager.__dict__)
    manager = bound
    layer.register_forward_pre_hook(manager.layer_pre_hook)
    layer.register_forward_hook(manager.layer_post_hook)

    class Executor:
        def _maybe_init_kv_connector_manager(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.pyexecutor.py_executor",
        SimpleNamespace(PyExecutor=Executor),
    )
    initialize_pd_connector()
    patched = Executor._maybe_init_kv_connector_manager
    initialize_pd_connector()
    assert Executor._maybe_init_kv_connector_manager is patched
    executor = Executor()
    executor.kv_connector_manager = manager
    executor.model_engine = SimpleNamespace(model=layer)
    executor._handle_errors = Mock()
    executor._maybe_init_kv_connector_manager()
    assert preserved.id in layer._forward_hooks
    assert len(layer._forward_hooks) == (1 if own_pd else 2)
    assert len(layer._forward_pre_hooks) == (0 if own_pd else 1)
    batch = SimpleNamespace(context_requests=[request], generation_requests=["decode"])
    manager.scheduler_output_manager.build_scheduler_output(batch, None, None)
    assert build.call_args.args[0].generation_requests == ([] if own_pd else ["decode"])
    if own_pd:
        manager.worker.load_failed(7, RuntimeError("source lost"))
        assert executor._handle_errors.call_args.kwargs == {"requests": [request]}
        assert manager.request_finished(request, []) is True
        finished.assert_called_once_with(request, [])
        for state in (
            manager.new_async_requests,
            manager.pending_async_requests,
            manager.local_finished_async_requests,
        ):
            assert not state.loading and state.saving == {8: "source"}
        assert not manager.finished_async_loading_requests
