# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Opt-in IPC integration test with a local, small unquantized model.

XINFERENCE_TEST_WEIGHT_CACHE_MODEL_PATH=/models/Qwen2.5-0.5B-Instruct \
    pytest -v xinference/model/llm/tests/test_weight_cache_gpu.py

Requires a Linux CUDA/ROCm GPU and the supported upstream engine versions.
The two engine tests run sequentially, releasing all GPU resources in finally.
"""

import asyncio
import os

import pytest

from ..weight_cache import WeightCachedModel

pytestmark = pytest.mark.skipif(
    not os.getenv("XINFERENCE_TEST_WEIGHT_CACHE_MODEL_PATH"),
    reason="Set XINFERENCE_TEST_WEIGHT_CACHE_MODEL_PATH to run GPU IPC tests",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
async def test_gpu_weights_survive_engine_reload(engine):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU required")
    pytest.importorskip(engine)

    class EngineModel(WeightCachedModel):
        _weight_cache_engine = engine
        _n_worker = 1
        model_path = os.environ["XINFERENCE_TEST_WEIGHT_CACHE_MODEL_PATH"]

        def __init__(self):
            config = (
                {"max_model_len": 512, "max_num_seqs": 4, "enforce_eager": True}
                if engine == "vllm"
                else {
                    "context_length": 512,
                    "max_running_requests": 4,
                    "disable_cuda_graph": True,
                }
            )
            self._init_weight_cache({"enable_weight_cache": True, **config})
            self._model_config = config
            self._engine = None

        def _sanitize_model_config(self, config):
            return config

        def load(self):
            self._prepare_weight_cache()
            if engine == "vllm":
                from vllm.engine.arg_utils import AsyncEngineArgs
                from vllm.engine.async_llm_engine import AsyncLLMEngine

                self._engine = AsyncLLMEngine.from_engine_args(
                    AsyncEngineArgs(model=self.model_path, **self._model_config)
                )
            else:
                import sglang

                self._engine = sglang.Engine(
                    model_path=self.model_path, **self._model_config
                )

        def wait_for_load(self):
            pass

        def _stop_engine(self):
            if self._engine is not None:
                self._engine.shutdown()
                self._engine = None

        async def generate(self):
            if engine == "vllm":
                from vllm import SamplingParams

                result = None
                async for output in self._engine.generate(
                    "The capital of France is",
                    SamplingParams(temperature=0, max_tokens=8),
                    "test",
                ):
                    result = output
                return result.outputs[0].text
            result = await asyncio.to_thread(
                self._engine.generate,
                "The capital of France is",
                {"temperature": 0, "max_new_tokens": 8},
            )
            return result["text"]

    model = EngineModel()
    try:
        await asyncio.to_thread(model.load)
        daemon = model._weight_cache
        assert daemon is not None
        pid = daemon.process.pid
        sockets = dict(daemon._socket_paths)
        before = await model.generate()
        patch = {"max_num_seqs" if engine == "vllm" else "max_running_requests": 8}
        await asyncio.to_thread(model.reload, patch, lambda stage: None)
        assert model._weight_cache is daemon
        assert daemon.process.pid == pid and daemon.process.poll() is None
        assert sockets == daemon._socket_paths
        daemon.client_config()  # Verify every original socket/rank is still alive.
        assert before and await model.generate() == before
    finally:
        await asyncio.to_thread(model._stop_engine)
        if model._weight_cache is not None:
            await asyncio.to_thread(model._weight_cache.stop)
