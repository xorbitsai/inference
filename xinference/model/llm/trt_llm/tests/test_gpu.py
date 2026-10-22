# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Opt-in live Qwen3-0.6B validation in two fresh engine processes.

XINFERENCE_TRTLLM_TEST_PYTHON can select a separate engine interpreter when
the test host runs LMCache in its own environment. The optional LMCache case
is skipped when that package is not installed on the test host.
"""

import asyncio
import json
import os
import subprocess
import sys

import pytest
import torch
import xoscar as xo

from ...xavier.backends.torch.storage import XavierCacheActor
from ...xavier.tests.utils import DiskCacheActor

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
@pytest.mark.parametrize("backend", ["native", "lmcache"])
async def test_fresh_gpu_instances_share_prefix_pages(tmp_path, backend):
    if backend == "lmcache":
        pytest.importorskip("lmcache")
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor_class = XavierCacheActor
        options = {}
        if backend == "lmcache":
            actor_class = DiskCacheActor
            options["cache_backend"] = dict(
                name="lmcache",
                config=dict(
                    cpu_bytes=512 << 20,
                    disk_bytes=1024 << 20,
                    disk_path=str(tmp_path / "kv"),
                ),
                namespace="trt-disk",
            )
        actor = await xo.create_actor(
            actor_class,
            512 * 1024**2,
            **options,
            address=pool.external_address,
            uid="trt-live",
        )

        async def run():
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    os.environ.get("XINFERENCE_TRTLLM_TEST_PYTHON", sys.executable),
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
        if backend == "lmcache":
            await actor.force_disk_reads()
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
        if backend == "lmcache":
            assert stats["backend_stats"]["l2_hits"] > 0
        (tmp_path / "result.json").write_text(
            json.dumps(
                dict(backend=backend, cold=cold, warm=warm, stats=stats), indent=2
            )
        )
        await xo.destroy_actor(actor)
        if backend == "lmcache":
            assert not list((tmp_path / "kv").iterdir())


SPECULATIVE_SCRIPT = r"""
import asyncio
import json
import sys
from xinference.model.llm.llm_family import match_llm
from xinference.model.llm.trt_llm.core import TRTLLMModel


async def main():
    mode = sys.argv[2]
    path = sys.argv[1]
    config = dict(
        max_seq_len=1024,
        max_batch_size=4,
        max_num_tokens=1024,
        dtype="bfloat16",
        enable_thinking=False,
        cuda_graph_config=None,
        kv_cache_config={"max_tokens": 2048},
        iter_stats_max_iterations=1024,
        disable_overlap_scheduler=True,
        enable_iter_perf_stats=True,
    )
    if mode == "ngram":
        config["speculative_config"] = dict(
            decoding_type="NGram",
            max_draft_len=3,
            max_matching_ngram_size=4,
            is_public_pool=False,
        )
    elif mode == "draft":
        config.update(draft_model_path=path, num_speculative_tokens=3)
    # Record real runtime acceptance without relying on polling/RPC stats.
    observed = []
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    original = PyExecutor._update_iter_stats

    def collect(executor, *args, **kwargs):
        stats = original(executor, *args, **kwargs)
        if stats.specdec_stats is not None:
            spec = stats.specdec_stats
            observed.append(
                dict(draft=spec.num_draft_tokens, accepted=spec.num_accepted_tokens)
            )
        return stats

    PyExecutor._update_iter_stats = collect
    model = TRTLLMModel(
        "spec-live", match_llm("qwen3", "pytorch", "0_6", "none"), path, config
    )
    try:
        model.load()
        observed.clear()
        prompt = "red green blue yellow " * 12
        sampling = dict(
            max_tokens=32, temperature=0, ignore_eos=True, stop=[], stop_token_ids=[]
        )
        completion = await model.async_generate(prompt, sampling)
        chunks = [
            c
            async for c in await model.async_generate(
                prompt,
                dict(sampling, stream=True, stream_options={"include_usage": True}),
            )
        ]
        assert (
            "".join(c["choices"][0]["text"] for c in chunks if c["choices"])
            == completion["choices"][0]["text"]
        )
        assert chunks[-1]["usage"] == completion["usage"]
        messages = [
            {"role": "user", "content": "Repeat red green blue yellow twelve times."}
        ]
        chat = await model.async_chat(messages, sampling)
        stream = await model.async_chat(
            messages,
            dict(sampling, stream=True, stream_options={"include_usage": True}),
        )
        chat_chunks = [c async for c in stream]
        assert (
            "".join(
                c["choices"][0]["delta"].get("content", "")
                for c in chat_chunks
                if c["choices"]
            )
            == chat["choices"][0]["message"]["content"]
        )
        assert chat_chunks[-1]["usage"] == chat["usage"]
        cancel = await model.async_generate(
            prompt, dict(sampling, max_tokens=128, stream=True), "cancel"
        )
        await anext(cancel)
        assert await model.abort_request("cancel") == "DONE"
        await cancel.aclose()
        assert not model._requests
        recovery = await model.async_generate(prompt, sampling)
        assert recovery["choices"][0]["text"] == completion["choices"][0]["text"]
        stats = observed
        if mode != "baseline":
            assert sum(s["draft"] for s in stats) > 0, stats
            assert sum(s["accepted"] for s in stats) > 0, stats
        report = dict(mode=mode, completion=completion, chat=chat, stats=stats)
        print("SPEC_RESULT=" + json.dumps(report))
    finally:
        model.stop()


asyncio.run(main())

"""


def test_ordinary_speculation_preserves_outputs(tmp_path):
    reports = {}
    for mode in ("baseline", "ngram", "draft"):
        result = subprocess.run(
            [
                os.environ.get("XINFERENCE_TRTLLM_TEST_PYTHON", sys.executable),
                "-c",
                SPECULATIVE_SCRIPT,
                os.environ["XINFERENCE_TRTLLM_TEST_MODEL"],
                mode,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        line = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("SPEC_RESULT=")
        )
        reports[mode] = json.loads(line.split("=", 1)[1])
    baseline = reports["baseline"]
    for mode in ("ngram", "draft"):
        assert (
            reports[mode]["completion"]["choices"] == baseline["completion"]["choices"]
        )
        assert reports[mode]["completion"]["usage"] == baseline["completion"]["usage"]
        assert reports[mode]["chat"]["choices"] == baseline["chat"]["choices"]
        assert reports[mode]["chat"]["usage"] == baseline["chat"]["usage"]
    (tmp_path / "speculation.json").write_text(json.dumps(reports, indent=2))
