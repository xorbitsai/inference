# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Initialize TRT-LLM 1.2.1 workers before model construction."""

import ctypes
import os


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
