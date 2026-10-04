# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Opt-in GPU transfer, native SGLang P/D and slot-lifetime regressions."""

import asyncio
import json
import multiprocessing
import os
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("XINFERENCE_TEST_SGLANG_XAVIER") != "1",
    reason="requires two CUDA GPUs, xoscar[nixl]>=0.11.1 and SGLang >=0.5.21",
)


@pytest.mark.asyncio
async def test_gpu_bytes_chunks_and_cancel():
    import torch
    import xoscar as xo

    from ...xavier.contract import KVCacheContract
    from ...xavier.transport import gpu_pool_options
    from ..xavier.directory import XavierPDDirectory
    from ..xavier.gpu import XavierGPUActor

    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 2, 2, 4, 4, "float16"
    )
    item = contract.layer_nbytes // 2
    allocations = []
    async with AsyncExitStack() as stack:
        directory_pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
        await stack.enter_async_context(directory_pool)
        directory = await xo.create_actor(
            XavierPDDirectory, address=directory_pool.external_address, uid="directory"
        )
        await directory.configure("namespace")
        actors = []
        for rank in range(2):
            torch.cuda.set_device(rank)
            caches = [
                torch.randint(0, 256, (8, item), dtype=torch.uint8, device=rank)
                for _ in range(4)
            ]
            aux = [torch.arange(48, dtype=torch.uint8).reshape(8, 6)]
            if rank:
                for cache in caches + aux:
                    cache.zero_()
            allocations.append((caches, aux))
            args = SimpleNamespace(
                gpu_id=rank,
                kv_data_ptrs=[buf.data_ptr() for buf in caches],
                kv_data_lens=[buf.numel() for buf in caches],
                kv_item_lens=[item] * 4,
                page_size=4,
                kv_cache_dtype_str="auto",
                state_types=[],
                num_draft_entries=0,
                aux_data_ptrs=[buf.data_ptr() for buf in aux],
                aux_data_lens=[buf.numel() for buf in aux],
                aux_item_lens=[6],
            )
            options = gpu_pool_options("127.0.0.1", os.environ)
            pool = await xo.create_actor_pool(options["external_address"], n_process=0)
            await stack.enter_async_context(pool)
            actors.append(
                await xo.create_actor(
                    XavierGPUActor,
                    args,
                    contract,
                    directory,
                    "namespace",
                    rank,
                    address=pool.external_address,
                    uid=f"{XavierGPUActor.default_uid()}-{rank}",
                )
            )
        p, d = actors
        for role in ("prefill", "decode"):
            await directory.prepare(1, "namespace", "prompt", role)
        await p.open(1)
        await p.init(1, 2, 3)
        incoming = asyncio.create_task(d.receive(1, [2, 5], 4))
        await p.add_chunk(1, [1])
        assert not await p.done(1)
        await p.add_chunk(1, [3])
        await asyncio.wait_for(incoming, 30)
        assert await p.done(1)
        for source, target in zip(allocations[0][0], allocations[1][0]):
            assert torch.equal(source[[1, 3]].cpu(), target[[2, 5]].cpu())
        assert torch.equal(allocations[0][1][0][3], allocations[1][1][0][4])
        stats = await d.get_stats()
        assert stats["gpu_batches"] > 0 and stats["cpu_batches"] == 0
        assert (await p.get_stats())["useful_bytes"] == 2 * 4 * item
        assert (await directory.get_stats())["gpu_bytes"] == 2 * 4 * item
        assert (await p.get_stats())["active_transfers"] == 0
        await p.clear(1)
        await directory.release(1)

        # Aborting while waiting for a source must drain the receiver before
        # native SGLang can return destination slots to its allocator.
        for role in ("prefill", "decode"):
            await directory.prepare(2, "namespace", "cancel", role)
        incoming = asyncio.create_task(d.receive(2, [1], 0))
        await asyncio.sleep(0.02)
        await d.abort(2)
        with pytest.raises(asyncio.CancelledError):
            await incoming
        await directory.release(2)
        assert (await directory.get_stats())["active_handoffs"] == 0


@pytest.mark.asyncio
async def test_native_sglang_gpu_pd(tmp_path, monkeypatch):
    import sglang as sgl
    import xoscar as xo
    from sglang.srt.plugins.hook_registry import HookRegistry
    from transformers import AutoTokenizer

    from ..core import SGLANGModel
    from ..xavier.config import configure_xavier
    from ..xavier.directory import XavierPDDirectory
    from ..xavier.gpu import GPU_CONFIG_ENV, XavierGPUActor
    from ..xavier.pd import SGLangXavierHandoff
    from ..xavier.plugin import register

    multiprocessing.set_start_method("spawn", force=True)
    register()
    HookRegistry.apply_hooks()
    monkeypatch.setenv("SGLANG_PLUGINS", "xinference_xavier")
    model_path = os.environ["XINFERENCE_TEST_PD_MODEL_PATH"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = "The Xavier prefill replica computes this prompt before decode. " * 70
    sampling = {"temperature": 0, "max_tokens": 16, "ignore_eos": True}
    common = dict(
        tp_size=1,
        dtype="float16",
        context_length=4096,
        max_total_tokens=4096,
        mem_fraction_static=0.3,
        disable_cuda_graph=True,
        page_size=64,
        stream_interval=1,
        log_level="warning",
    )
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    results, engines = {}, []
    async with pool:
        directory = await xo.create_actor(
            XavierPDDirectory, address=pool.external_address, uid="directory"
        )
        models = []
        try:
            for rank, role in enumerate(("prefill", "decode")):
                cache = dict(
                    address=directory.address,
                    uid="directory",
                    role=role,
                    rank=rank,
                    host="127.0.0.1",
                )
                config = dict(common)
                await asyncio.to_thread(configure_xavier, model_path, config, cache)
                monkeypatch.setenv(GPU_CONFIG_ENV, json.dumps(cache))
                engine = await asyncio.to_thread(
                    sgl.Runtime, model_path=model_path, base_gpu_id=rank, **config
                )
                engines.append(engine)
                model = object.__new__(SGLANGModel)
                model.model_uid = role
                model._active_request_ids = set()
                model._engine = engine
                model._xavier_handoff = SGLangXavierHandoff(cache, 64, tokenizer)
                models.append(model)

            async def request(room, text, stream=False):
                handoff = dict(mode="gpu", room=room)
                prefill = asyncio.create_task(
                    models[0].async_generate(
                        text,
                        generate_config=dict(
                            sampling,
                            max_tokens=1,
                            _pd_kv_transfer_params=dict(
                                do_remote_decode=True, sglang_xavier=handoff
                            ),
                        ),
                    )
                )
                d = await models[1].async_generate(
                    text,
                    generate_config=dict(
                        sampling,
                        stream=stream,
                        _pd_kv_transfer_params=dict(
                            do_remote_prefill=True, sglang_xavier=handoff
                        ),
                    ),
                )

                async def completed():
                    p = await prefill
                    assert p["_pd_kv_transfer_params"]["sglang_xavier"] == handoff

                if stream:

                    async def chunks():
                        try:
                            async for chunk in d:
                                await completed()
                                yield chunk
                        finally:
                            await d.aclose()
                            prefill.cancel()
                            await asyncio.gather(prefill, return_exceptions=True)

                    return chunks()
                await completed()
                return d

            results["pd"] = await request(1, prompt)
            results["concurrent"] = await asyncio.gather(
                *(request(i + 2, f"Request {i}: " + prompt) for i in range(4))
            )
            stream = await request(6, prompt, stream=True)
            results["stream"] = [chunk async for chunk in stream]
            assert results["stream"][-1]["choices"][0]["finish_reason"]
            cancelled = await request(7, prompt, stream=True)
            await anext(cancelled)
            await cancelled.aclose()
            results["after_cancel"] = await request(8, prompt)
            results["directory"] = await directory.get_stats()
            assert results["directory"]["active_handoffs"] == 0
            assert results["directory"]["completed_requests"] == 8
            assert results["directory"]["gpu_bytes"] > 0
            for rank, address in results["directory"]["peers"].items():
                actor = await xo.actor_ref(
                    address=address, uid=f"{XavierGPUActor.default_uid()}-{rank}"
                )
                stats = await actor.get_stats()
                results[f"gpu{rank}"] = stats
                assert stats["active_rooms"] == stats["active_transfers"] == 0
                assert stats["cpu_batches"] == 0
            assert results["gpu1"]["gpu_batches"] > 0
        finally:
            for engine in engines:
                await asyncio.to_thread(engine.shutdown)
        # A fresh ordinary engine establishes greedy output equivalence.
        baseline = await asyncio.to_thread(
            sgl.Runtime, model_path=model_path, base_gpu_id=0, **common
        )
        try:
            models[0]._engine = baseline
            models[0]._xavier_handoff = None
            results["baseline"] = await models[0].async_generate(
                prompt, generate_config=sampling
            )
            assert (
                results["pd"]["choices"][0]["text"]
                == results["baseline"]["choices"][0]["text"]
            )
        finally:
            await asyncio.to_thread(baseline.shutdown)
            Path(
                os.environ.get(
                    "XINFERENCE_SGLANG_XAVIER_RESULT", str(tmp_path / "gpu-pd.json")
                )
            ).write_text(json.dumps(results, indent=2))
