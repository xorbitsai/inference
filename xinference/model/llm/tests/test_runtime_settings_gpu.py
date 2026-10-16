# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Opt-in vLLM/SGLang tests with local ModelScope checkpoints.

XINFERENCE_TEST_VISION_WEIGHT_CACHE_MODEL_PATH=/models/Qwen2.5-VL-3B-Instruct \
    pytest -s xinference/model/llm/tests/test_runtime_settings_gpu.py
"""

import asyncio
import base64
import io
import os

import pytest

vision_checkpoint = pytest.mark.skipif(
    not os.getenv("XINFERENCE_TEST_VISION_WEIGHT_CACHE_MODEL_PATH"),
    reason="Set XINFERENCE_TEST_VISION_WEIGHT_CACHE_MODEL_PATH for real GPU tests",
)


@pytest.mark.asyncio
@vision_checkpoint
@pytest.mark.parametrize(
    "backend,initial_limit",
    [("vllm", 1), ("sglang", 1), ("sglang", 0), ("sglang", None)],
)
async def test_image_count_reload(backend, initial_limit):
    import torch
    from PIL import Image

    from ....core.model import ModelActor
    from ..llm_family import match_llm

    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU required")
    pytest.importorskip(backend)

    family = match_llm("qwen2.5-vl-instruct", "pytorch", "3", "none", "modelscope")
    image = io.BytesIO()
    Image.new("RGB", (56, 56), "red").save(image, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
    config = {
        "max_model_len": 2048,
        "max_num_seqs": 2,
        "max_num_batched_tokens": 2048,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "kv_cache_memory_bytes": 256 * 1024**2,
        "enforce_eager": True,
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        "mm_processor_cache_gb": 0,
    }
    if backend == "vllm":
        from vllm.exceptions import VLLMValidationError

        from ..vllm.core import VLLMMultiModel

        model_class = VLLMMultiModel
        field = "limit_mm_per_prompt"
    else:
        from ..sglang.core import SGLANGVisionModel

        model_class = SGLANGVisionModel
        field = "limit_mm_data_per_request"
        config = {
            "context_length": 2048,
            "max_running_requests": 2,
            "max_total_tokens": 2048,
            "chunked_prefill_size": 512,
            "tp_size": 1,
            "mem_fraction_static": 0.65,
            "disable_cuda_graph": True,
            "disable_radix_cache": True,
            field: {
                "video": 0,
                **({"image": initial_limit} if initial_limit is not None else {}),
            },
        }
    model = model_class(
        "runtime-settings-test",
        family,
        os.environ["XINFERENCE_TEST_VISION_WEIGHT_CACHE_MODEL_PATH"],
        {**config, "enable_weight_cache": True},
    )
    if backend == "vllm":
        model.set_loop(asyncio.get_running_loop())
    actor = ModelActor.__new__(ModelActor)
    actor._model, actor._model_state, actor._serve_count = model, "ready", 0

    def messages(count):
        content = [
            {"type": "image_url", "image_url": {"url": image_url}} for _ in range(count)
        ]
        content.append({"type": "text", "text": "Name the main color."})
        return [{"role": "user", "content": content}]

    async def generate(count):
        output = await model.async_chat(
            messages(count),
            {"temperature": 0, "top_k": 1, "max_tokens": 8},
        )
        text = output["choices"][0]["message"]["content"]
        assert text
        return text

    async def assert_rejected(count, limit):
        if backend == "vllm":
            with pytest.raises(
                VLLMValidationError,
                match=r"(?i)(at most|maximum|limit).*(image)|image.*(at most|maximum|limit)",
            ):
                await generate(count)
        else:
            import aiohttp

            # Inspect the native response before raise_for_status discards its
            # body: unrelated HTTP 400 failures must not satisfy this check.
            prompt, images, *_ = await model._prepare_chat_inputs(messages(count), {})
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    model._engine.generate_url,
                    json={
                        "text": prompt,
                        "image_data": images,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    },
                    headers=model._engine_headers(),
                ) as response:
                    assert response.status == 400
                    assert (
                        f"Image count {count} exceeds limit {limit} per request."
                        in await response.text()
                    )

    try:
        await asyncio.to_thread(model.load)
        await asyncio.to_thread(model.wait_for_load)
        before = await generate(1) if initial_limit != 0 else None
        engine = model._engine
        assert "reasoning_content" not in model.get_reload_config()["parameters"]
        with pytest.raises(ValueError, match="does not support reasoning-content"):
            await actor.reload({"reasoning_content": True}, drain_timeout=30)
        assert model._engine is engine
        daemon = model._weight_cache
        pid = daemon.process.pid
        sockets = {p: p.stat().st_ino for p in daemon._socket_paths}
        if initial_limit is None:
            assert await generate(3)
        else:
            await assert_rejected(initial_limit + 1, initial_limit)
        with pytest.raises(ValueError, match="image_limits"):
            await actor.reload({field: {"image": 0}}, drain_timeout=30)
        assert model._engine is engine
        assert await actor.reload({field: {"image": 2}}, drain_timeout=30)
        assert model._engine is not engine
        assert model._weight_cache is daemon
        assert daemon.process.pid == pid and daemon.process.poll() is None
        assert {p: p.stat().st_ino for p in sockets} == sockets
        ipc = daemon.client_config()
        if backend == "vllm":
            assert ipc["load_format"] == "ipc_cache"
            assert ipc["model_loader_extra_config"]["mode"] == "zero_copy"
            assert ipc["model_loader_extra_config"]["fallback"] is False
        else:
            assert ipc == {"weight_cache_mode": "client"}
        assert model.get_reload_config()["model_config"][field] == {
            "image": 2,
            "video": 0,
        }
        assert await generate(2)
        after = await generate(1)
        if before is not None:
            assert after == before
        if backend == "sglang":
            await assert_rejected(3, 2)
    finally:
        await asyncio.to_thread(model.stop)
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("XINFERENCE_TEST_REASONING_MODEL_PATH"),
    reason="Set a local Qwen3-0.6B path",
)
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
async def test_reasoning_hot_update_with_and_without_weight_retention(backend):
    import torch

    from ....core.model import ModelActor
    from ..llm_family import match_llm

    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU required")
    pytest.importorskip(backend)
    if backend == "vllm":
        from ..vllm.core import VLLMChatModel

        model_class = VLLMChatModel
        config = {
            "max_model_len": 512,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 512,
            "gpu_memory_utilization": 0.8,
            "kv_cache_memory_bytes": 128 * 1024**2,
            "enforce_eager": True,
        }
    else:
        from ..sglang.core import SGLANGChatModel

        model_class = SGLANGChatModel
        config = {
            "context_length": 512,
            "max_running_requests": 2,
            "max_total_tokens": 1024,
            "chunked_prefill_size": 512,
            "mem_fraction_static": 0.65,
            "disable_cuda_graph": True,
            "disable_radix_cache": True,
            "tp_size": 1,
        }
    family = match_llm("qwen3", "pytorch", "0_6", "none", "modelscope")
    for cached in (False, True):
        model = model_class(
            "reasoning-settings-test",
            family,
            os.environ["XINFERENCE_TEST_REASONING_MODEL_PATH"],
            {
                "enable_weight_cache": cached,
                "enable_thinking": True,
                **config,
            },
        )
        if backend == "vllm":
            model.set_loop(asyncio.get_running_loop())
        actor = ModelActor.__new__(ModelActor)
        actor._model, actor._model_state, actor._serve_count = model, "ready", 0

        async def generate():
            result = await model.async_chat(
                [{"role": "user", "content": "What is 2 plus 2?"}],
                {"temperature": 0, "top_k": 1, "max_tokens": 16},
            )
            message = result["choices"][0]["message"]
            combined = (message.get("reasoning_content") or "") + (
                message.get("content") or ""
            )
            return (
                combined.replace("<think>", "").replace("</think>", "").strip(),
                message,
            )

        try:
            await asyncio.to_thread(model.load)
            await asyncio.to_thread(model.wait_for_load)
            before, message = await generate()
            assert before and not message.get("reasoning_content")
            parser, engine = model.reasoning_parser, model._engine
            assert not await actor.reload({"reasoning_content": True}, drain_timeout=30)
            assert model._engine is engine and not parser.reasoning_content
            after, message = await generate()
            assert message.get("reasoning_content") and after == before
            await actor.reload({"reasoning_content": False}, drain_timeout=30)
            assert model._engine is engine
            after, message = await generate()
            assert not message.get("reasoning_content") and after == before
        finally:
            await asyncio.to_thread(model.stop)
            await asyncio.sleep(0)
