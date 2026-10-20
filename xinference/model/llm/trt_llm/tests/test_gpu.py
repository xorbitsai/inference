# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Opt-in live Qwen3-0.6B validation in two fresh engine processes."""

import asyncio
import json
import os
import subprocess
import sys

import pytest
import torch
import xoscar as xo

from ...xavier.backends.torch.storage import XavierCacheActor

pytestmark = pytest.mark.skipif(
    sys.platform != "linux"
    or not torch.cuda.is_available()
    or not os.environ.get("XINFERENCE_TRTLLM_TEST_MODEL"),
    reason="Requires Linux/CUDA, TRT-LLM 1.2.1 and XINFERENCE_TRTLLM_TEST_MODEL",
)

SCRIPT = r"""
import asyncio
import json
import sys
import xoscar as xo
from xinference.model.llm.llm_family import match_llm
from xinference.model.llm.trt_llm.core import TRTLLMModel

async def main():
    family = match_llm("qwen3", "pytorch", "0_6", "none")
    model = TRTLLMModel("live", family, sys.argv[1], {
        "max_seq_len": 1024, "dtype": "bfloat16", "enable_thinking": False,
        "max_batch_size": 4, "max_num_tokens": 1024,
        "cuda_graph_config": None,
        "kv_cache_config": {"max_tokens": 2048},
        "_xavier_cache_config": {"address": sys.argv[2], "uid": "trt-live"},
    })
    try:
        model.load()
        cache = await xo.actor_ref(address=sys.argv[2], uid="trt-live")
        before = await cache.get_stats()
        result = await model.async_generate(
            "The capital of France is Paris. " * 32,
            {"max_tokens": 16, "temperature": 0, "ignore_eos": True,
             "stop": [], "stop_token_ids": []},
        )
        after = await cache.get_stats()
        print("TRT_XAVIER_RESULT=" + json.dumps({
            "completion": result, "before": before, "after": after,
        }))
    finally:
        model.stop()

if __name__ == "__main__":
    asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_fresh_gpu_instances_share_prefix_pages():
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierCacheActor,
            512 * 1024**2,
            address=pool.external_address,
            uid="trt-live",
        )

        async def run():
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    "-c",
                    SCRIPT,
                    os.environ["XINFERENCE_TRTLLM_TEST_MODEL"],
                    actor.address,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            line = next(
                line
                for line in result.stdout.splitlines()
                if line.startswith("TRT_XAVIER_RESULT=")
            )
            return json.loads(line.split("=", 1)[1])

        cold = await run()
        assert (await actor.get_stats())["stored_pages"] > 0
        warm = await run()
        stats = await actor.get_stats()
        print("TRT_XAVIER_STATS=" + json.dumps(stats, sort_keys=True))
        assert stats["imported_pages"] > 0
        # Check the actual request, excluding any model warmup during load().
        assert warm["after"]["imported_pages"] > warm["before"]["imported_pages"]
        assert (
            cold["completion"]["choices"][0]["text"]
            == warm["completion"]["choices"][0]["text"]
        )
        assert warm["completion"]["usage"]["completion_tokens"] == 16
