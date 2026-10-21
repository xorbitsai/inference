# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Initialize TRT-LLM 1.2.1 workers before model construction."""

import ctypes
import os
from functools import wraps
from types import SimpleNamespace


def initialize_pd_connector() -> None:
    """Adapt private executor plumbing only for our pinned TP1 P/D connector."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    from .connector import XavierConnectorWorker

    original = PyExecutor._maybe_init_kv_connector_manager
    if getattr(original, "_xinference_pd", False):
        return

    def owns(manager):
        return (
            manager is not None
            and isinstance(manager.worker, XavierConnectorWorker)
            and manager.worker.client.pd_gpu
        )

    @wraps(original)
    def initialize(executor):
        original(executor)
        manager = executor.kv_connector_manager
        if not owns(manager):
            return
        # Both layer callbacks are no-ops: all KV writes finish before resume,
        # and publication fences the full forward pass in wait_for_save.
        for module in executor.model_engine.model.modules():
            for hooks in (module._forward_pre_hooks, module._forward_hooks):
                for key, callback in list(hooks.items()):
                    if getattr(callback, "__self__", None) is manager and getattr(
                        callback, "__name__", ""
                    ) in ("layer_pre_hook", "layer_post_hook"):
                        del hooks[key]

        output = manager.scheduler_output_manager
        build = output.build_scheduler_output

        def context_output(batch, async_requests, cache):
            # Our connector consumes only new context requests. Avoid copying
            # full token histories and block lists on every decode step.
            return build(
                SimpleNamespace(
                    context_requests=batch.context_requests, generation_requests=[]
                ),
                async_requests,
                cache,
            )

        output.build_scheduler_output = context_output
        finished = manager.request_finished

        def request_finished(request, blocks):
            try:
                return finished(request, blocks)
            finally:
                # An aborted load must not be resumed by a late completion.
                # The scheduler callback above drains its writes before return.
                rid = request.request_id
                for pending in (
                    manager.new_async_requests,
                    manager.pending_async_requests,
                    manager.local_finished_async_requests,
                ):
                    pending.loading.pop(rid, None)
                manager.finished_async_loading_requests.pop(rid, None)

        manager.request_finished = request_finished

        def load_failed(rid, error):
            request = manager.pending_async_requests.loading[rid]
            executor._handle_errors(
                f"TRT Xavier GPU KV load failed: {error}", requests=[request]
            )

        manager.worker.load_failed = load_failed

    setattr(initialize, "_xinference_pd", True)
    PyExecutor._maybe_init_kv_connector_manager = initialize


def initialize_worker() -> None:
    """Declare the NVML pointer argument used by TRT's CC-state query.

    TRT-LLM 1.2.1 passes a ctypes structure to this Python binding, whose
    underlying C function has no argtypes. Without the pointer declaration,
    ctypes passes the structure by value and the driver can segfault.
    This runs in the Xinference model process, before the TRT autotuner starts.
    """
    import pynvml

    pynvml.nvmlInit()
    try:
        try:
            function = pynvml._nvmlGetFunctionPointer(
                "nvmlSystemGetConfComputeSettings"
            )
        except pynvml.NVMLError_FunctionNotFound:
            return  # TRT already handles drivers that lack the query.
        function.argtypes = [
            ctypes.POINTER(pynvml.c_nvmlSystemConfComputeSettings_v1_t)
        ]
        function.restype = ctypes.c_int
    finally:
        pynvml.nvmlShutdown()


def prepare_runtime(config: dict) -> None:
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
    ):
        if type(config.get(name, 1)) is not int or config.get(name, 1) != 1:
            raise ValueError("The TRT-LLM adapter currently requires TP=PP=CP=1")
    if config.get("dtype", "auto") not in ("auto", "bfloat16"):
        raise ValueError("TRT-LLM 1.2.1 Qwen3 currently requires BF16 weights and KV")
    # Each Xinference replica already owns a model process. TRT's TP1 worker
    # runs there directly, including the NVML fix, without another MPI pool.
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    initialize_worker()
    connector = config.get("kv_connector_config") or {}
    if connector.get("connector_module") == "xinference.model.llm.trt_llm.connector":
        initialize_pd_connector()
