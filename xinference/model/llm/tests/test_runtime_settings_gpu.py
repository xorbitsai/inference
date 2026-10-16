# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Opt-in vLLM test with a local Qwen2.5-VL-3B-Instruct checkpoint.

XINFERENCE_TEST_VISION_WEIGHT_CACHE_MODEL_PATH=/models/Qwen2.5-VL-3B-Instruct \
    pytest -s xinference/model/llm/tests/test_runtime_settings_gpu.py
"""

import asyncio
import base64
import io
import json
import os
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("XINFERENCE_TEST_VISION_WEIGHT_CACHE_MODEL_PATH"),
    reason="Set XINFERENCE_TEST_VISION_WEIGHT_CACHE_MODEL_PATH for real GPU tests",
)


@pytest.mark.asyncio
async def test_vllm_image_count_and_reasoning_updates():
    import torch
    from PIL import Image

    from ....core.model import ModelActor
    from ..llm_family import match_llm
    from ..vllm.core import VLLMMultiModel

    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU required")
    pytest.importorskip("vllm")
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
    records = []
    for cached in (True, False):
        model = VLLMMultiModel(
            "runtime-settings-test",
            family,
            os.environ["XINFERENCE_TEST_VISION_WEIGHT_CACHE_MODEL_PATH"],
            {**config, "enable_weight_cache": cached},
        )
        model.set_loop(asyncio.get_running_loop())
        actor = ModelActor.__new__(ModelActor)
        actor._model, actor._model_state, actor._serve_count = model, "ready", 0

        async def generate(count):
            content = [
                {"type": "image_url", "image_url": {"url": image_url}}
                for _ in range(count)
            ]
            content.append({"type": "text", "text": "Name the main color."})
            output = await model.async_chat(
                [{"role": "user", "content": content}],
                {"temperature": 0, "max_tokens": 8},
            )
            text = output["choices"][0]["message"]["content"]
            assert text
            return text

        try:
            await asyncio.to_thread(model.load)
            await asyncio.to_thread(model.wait_for_load)
            before = await generate(1)
            engine = model._engine
            started = time.perf_counter()
            await actor.reload({"reasoning_content": True}, drain_timeout=30)
            hot_seconds = time.perf_counter() - started
            assert model._engine is engine
            assert model.reasoning_parser.reasoning_content
            assert await generate(1) == before
            record = {"cached": cached, "hot_seconds": hot_seconds, "output": before}
            records.append(record)
            if not cached:
                assert model._weight_cache is None
                continue
            daemon = model._weight_cache
            pid = daemon.process.pid
            sockets = {p: p.stat().st_ino for p in daemon._socket_paths}
            with pytest.raises(
                ValueError,
                match=r"(?i)(at most|maximum|limit).*(image)|image.*(at most|maximum|limit)",
            ):
                await generate(2)
            with pytest.raises(ValueError, match="image_limits"):
                await actor.reload(
                    {"limit_mm_per_prompt": {"image": 0}}, drain_timeout=30
                )
            assert model._engine is engine
            started = time.perf_counter()
            await actor.reload({"limit_mm_per_prompt": {"image": 2}}, drain_timeout=30)
            record["reload_seconds"] = time.perf_counter() - started
            assert model._engine is not engine
            assert model._weight_cache is daemon
            assert daemon.process.pid == pid and daemon.process.poll() is None
            assert {p: p.stat().st_ino for p in sockets} == sockets
            assert model.get_reload_config()["model_config"]["limit_mm_per_prompt"] == {
                "image": 2,
                "video": 0,
            }
            record["two_image_output"] = await generate(2)
            assert await generate(1) == before
            record["daemon_pid"] = pid
            record["socket_inodes"] = {str(p): inode for p, inode in sockets.items()}
            record["strict_ipc_config"] = daemon.client_config()
        finally:
            await asyncio.to_thread(model.stop)
            await asyncio.sleep(0)
    if path := os.getenv("XINFERENCE_TEST_RUNTIME_SETTINGS_RESULT"):
        Path(path).write_text(json.dumps(records, indent=2))
