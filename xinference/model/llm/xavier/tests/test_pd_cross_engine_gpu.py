# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Opt-in bitwise tests: XINFERENCE_TEST_CROSS_ENGINE_GPU=1 pytest this file."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import xoscar as xo

from ...sglang.xavier.directory import XavierPDDirectory
from ..backends.torch.pd import CrossEngineGPUActor, canonical_vllm_views
from ..pd_contract import prompt_digest
from ..transport import gpu_pool_options

pytestmark = pytest.mark.skipif(
    os.environ.get("XINFERENCE_TEST_CROSS_ENGINE_GPU") != "1",
    reason="Requires an explicitly reserved pair of CUDA GPUs with NIXL",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["vllm", "sglang"])
async def test_gpu_pages_survive_cross_engine_layouts_bitwise(kv_contract, producer):
    from torch.multiprocessing.reductions import reduce_tensor
    from xoscar.backends.allocate_strategy import ProcessIndex

    assert torch.cuda.device_count() >= 2
    contract = replace(kv_contract, num_layers=2, block_size=64, head_dim=8)
    caches, views = [], []
    for gpu in (0, 1):
        engine = producer if gpu == 0 else ("sglang" if producer == "vllm" else "vllm")
        if engine == "vllm":
            # vLLM 0.28 logical [B,H,T,2D], stored physically in NHD order.
            native = {
                f"model.layers.{layer}.self_attn.attn": torch.zeros(
                    7, 64, 2, 16, device=f"cuda:{gpu}", dtype=torch.float16
                ).transpose(1, 2)
                for layer in range(2)
            }
            canonical = canonical_vllm_views(native, 7, contract)
        else:
            native = canonical = {
                str(layer): torch.zeros(
                    7, 64, 2, 8, device=f"cuda:{gpu}", dtype=torch.float16
                )
                for layer in range(4)
            }
        caches.append(native)
        views.append(canonical)
    for layer, cache in enumerate(views[0].values()):
        cache.copy_(
            (torch.arange(cache.numel(), device="cuda:0") % 257).reshape(cache.shape)
            + layer * 300
        )
    for gpu in (0, 1):
        torch.cuda.synchronize(gpu)
    options = gpu_pool_options("127.0.0.1", os.environ)
    pool = await xo.create_actor_pool(
        options["external_address"], n_process=2, subprocess_start_method="spawn"
    )
    actors = []
    try:
        await pool.start()
        directory = await xo.create_actor(
            XavierPDDirectory, address=pool.external_address
        )
        await directory.configure(contract.fingerprint)
        for gpu in (0, 1):
            actor = await xo.create_actor(
                CrossEngineGPUActor,
                SimpleNamespace(gpu_id=gpu, aux_item_lens=[]),
                contract,
                directory,
                contract.fingerprint,
                gpu + 1,
                ipc_descriptors={
                    key: reduce_tensor(value)[1] for key, value in views[gpu].items()
                },
                address=pool.external_address,
                allocate_strategy=ProcessIndex(gpu + 1),
                uid=f"{CrossEngineGPUActor.default_uid()}-{gpu + 1}",
            )
            actors.append(actor)
        room, source, destination = 123, [5, 1, 3], [2, 6, 4]
        for role in ("prefill", "decode"):
            await directory.prepare(
                room, contract.fingerprint, prompt_digest(list(range(129))), role, 129
            )
        if producer == "vllm":
            await actors[0].register_prefill(room, "request", source, 42, 129)
        else:
            await actors[0].open(room)
            await actors[0].init(room, len(source), 0)
            await actors[0].add_chunk(room, source[:1])
            await actors[0].add_chunk(room, source[1:], [b"first-token"])
        assert await actors[1].receive(room, destination, 0) == set()
        assert await actors[0].poll_direct_gpu_v1() == {
            "request" if producer == "vllm" else f"{room}:0"
        }
        for key, original in views[0].items():
            assert torch.equal(original[source].cpu(), views[1][key][destination].cpu())
        stats = await directory.get_stats()
        assert stats["imported_tokens"] == 128 and stats["active_handoffs"] == 0
        for actor in actors:
            stats = await actor.get_stats()
            assert stats["cpu_batches"] == stats["active_transfers"] == 0
    finally:
        for actor in actors:
            await xo.destroy_actor(actor)
        await pool.stop()
