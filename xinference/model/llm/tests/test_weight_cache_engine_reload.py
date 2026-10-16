# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Real engine classes, with only upstream engine construction mocked.

These tests live outside the GPU engine directories to run in default CI.
"""

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ..llm_family import match_llm
from ..sglang import core as sglang_core
from ..vllm import core as vllm_core


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
async def test_real_model_class_reloads_on_same_instance(
    monkeypatch, tmp_path, backend
):
    family = match_llm("qwen2.5-instruct", "pytorch", "0_5", "none", "huggingface")
    engines = []
    constructors = []

    def construct(**config):
        constructors.append(config)
        engine = SimpleNamespace(
            pid=1,
            shutdown=MagicMock(),
            check_health=AsyncMock(),
            model_config=SimpleNamespace(max_model_len=2048),
        )
        engines.append(engine)
        return engine

    class EngineArgs:
        def __init__(self, **config):
            self.config = config

        def create_engine_config(self):
            return self

    if backend == "vllm":
        monkeypatch.setattr(vllm_core, "VLLM_VERSION", vllm_core.VLLM_VERSION)
        monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", vllm_core.VLLM_INSTALLED)
        monkeypatch.setitem(
            sys.modules,
            "vllm",
            SimpleNamespace(
                __version__="0.31.0", envs=SimpleNamespace(VLLM_USE_V1=True)
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "vllm.engine.arg_utils",
            SimpleNamespace(AsyncEngineArgs=EngineArgs),
        )
        monkeypatch.setitem(
            sys.modules,
            "vllm.engine.async_llm_engine",
            SimpleNamespace(
                AsyncLLMEngine=SimpleNamespace(
                    from_engine_args=lambda args: construct(**args.config)
                )
            ),
        )
        monkeypatch.setitem(
            sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=object)
        )
        monkeypatch.setitem(
            sys.modules, "vllm.v1.executor", SimpleNamespace(Executor=object)
        )
        monkeypatch.setattr(vllm_core, "_init_guided_decoding_classes", lambda: None)
        monkeypatch.setattr(vllm_core, "_update_vllm_supported_lists", lambda: None)
        model = vllm_core.VLLMChatModel(
            "test",
            family,
            "/models/test",
            {
                "enable_weight_cache": True,
                "max_num_seqs": 16,
                "tensor_parallel_size": 1,
            },
        )
        model._loop = asyncio.get_running_loop()
        field = "max_num_seqs"
    else:
        monkeypatch.setitem(
            sys.modules,
            "sglang",
            SimpleNamespace(__version__="0.5.21", Runtime=construct),
        )
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.server_args",
            SimpleNamespace(ServerArgs=lambda **config: config),
        )
        monkeypatch.setattr(sglang_core, "get_next_port", lambda: 30001)
        start_method = MagicMock()
        monkeypatch.setattr(
            sglang_core.multiprocessing, "set_start_method", start_method
        )
        model = sglang_core.SGLANGChatModel(
            "test",
            family,
            "/models/test",
            {"enable_weight_cache": True, "max_running_requests": 16},
        )
        gc_close = MagicMock()
        monkeypatch.setattr(model._gc_freeze, "close", gc_close)
        field = "max_running_requests"
    monkeypatch.setattr(model, "_get_cuda_count", lambda: 1)
    monkeypatch.setattr(
        model, "prepare_parse_reasoning_content", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(model, "prepare_parse_tool_calls", lambda: None)
    cache = SimpleNamespace(
        client_config=lambda: {}, engine_config=lambda config: config, stop=MagicMock()
    )
    model._weight_cache = cache
    if backend == "sglang":
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SGLANG_JIT_CACHE_DIR", "jit-cache")
        prepare_weight_cache = model._prepare_weight_cache

        def prepare():
            assert os.environ["SGLANG_JIT_CACHE_DIR"] == str(tmp_path / "jit-cache")
            prepare_weight_cache()

        monkeypatch.setattr(model, "_prepare_weight_cache", prepare)
    try:
        await asyncio.to_thread(model.load)
        await asyncio.to_thread(model.wait_for_load)
        await asyncio.sleep(0)
        old_thread = model._loading_thread
        health = getattr(model, "_check_health_task", None)
        model._loading_error = (RuntimeError, RuntimeError("stale error"), None)
        await asyncio.to_thread(model.reload, {field: 32}, lambda stage: None)
        await asyncio.sleep(0)
        assert model._weight_cache is cache
        assert model._engine is engines[1]
        engines[0].shutdown.assert_called_once()
        cache.stop.assert_not_called()
        assert model._loading_error is None
        assert constructors[1][field] == 32
        assert model.get_reload_config()["model_config"][field] == 32
        if backend == "vllm":
            assert model._loading_thread is not old_thread
            assert health.cancelled()
            assert model._check_health_task is not health
        else:
            assert start_method.call_args_list == [(("spawn",), {"force": True})] * 2
            gc_close.assert_called_once()
    finally:
        await asyncio.to_thread(model.stop)
        await asyncio.sleep(0)
    cache.stop.assert_called_once()
    if backend == "sglang":
        assert gc_close.call_count == 2


@pytest.mark.parametrize("unsupported", ["ggufv2", "xoscar"])
def test_vllm_unsupported_cache_mode_fails_before_preload(monkeypatch, unsupported):
    model = object.__new__(vllm_core.VLLMModel)
    model._enable_weight_cache = True
    model._model_config = {}
    model._weight_cache = None
    model.model_spec = SimpleNamespace(model_format=unsupported)
    model._xinference_vllm_executor_backend = (
        "xoscar" if unsupported == "xoscar" else "auto"
    )
    preload = MagicMock()
    monkeypatch.setattr("xinference.model.llm.weight_cache.WeightCacheDaemon", preload)
    with pytest.raises(ValueError, match="GGUF|native vLLM"):
        model._prepare_weight_cache()
    preload.assert_not_called()
