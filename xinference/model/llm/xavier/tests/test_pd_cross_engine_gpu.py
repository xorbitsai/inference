# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Opt-in bitwise tests: XINFERENCE_TEST_CROSS_ENGINE_GPU=1 pytest this file."""

import os
import struct
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
@pytest.mark.parametrize(
    "producer,full_allocation", [("vllm", False), ("sglang", False), ("sglang", True)]
)
@pytest.mark.parametrize("prompt_tokens", [65, 129, 193])
async def test_gpu_pages_survive_cross_engine_layouts_bitwise(
    kv_contract, producer, full_allocation, prompt_tokens
):
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
        metadata_sizes = [64, 64, 64, 64, 512, 512, 64, 128, 1024, 64]
        for gpu in (0, 1):
            actor = await xo.create_actor(
                CrossEngineGPUActor,
                SimpleNamespace(
                    gpu_id=gpu,
                    aux_item_lens=(
                        metadata_sizes if gpu == 1 and producer == "vllm" else []
                    ),
                ),
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
        source_count = (prompt_tokens + 63) // 64
        target_count = (
            source_count if producer == "vllm" else (prompt_tokens - 1 + 63) // 64
        )
        allocated_count = source_count if full_allocation else target_count
        room, source, destination = (
            123,
            [5, 1, 3, 0][:source_count],
            [2, 6, 4, 0][:allocated_count],
        )
        for rank, role in enumerate(("prefill", "decode"), 1):
            await directory.prepare(
                room,
                contract.fingerprint,
                prompt_digest(list(range(prompt_tokens))),
                role,
                rank=rank,
                prompt_tokens=prompt_tokens,
            )
        if producer == "vllm":
            await actors[0].register_prefill(room, "request", source, 42, prompt_tokens)
        else:
            await actors[0].open(room)
            await actors[0].init(room, len(source), 0)
            await actors[0].add_chunk(room, source[:1])
            await actors[0].add_chunk(room, source[1:], [b"first-token"])
        received = await actors[1].receive(room, destination, 0)
        if producer == "vllm":
            nbytes, payload = received
            assert struct.unpack_from("<i", payload[0])[0] == 42
            assert struct.unpack_from("<i", payload[1])[0] == prompt_tokens
            assert struct.unpack_from("<Q", payload[-1])[0] == room
            await directory.complete(room, nbytes)
            await directory.release(room, "decode")
        else:
            assert received == set()
        assert await actors[0].poll_direct_gpu_v1() == {
            "request" if producer == "vllm" else f"{room}:0"
        }
        for key, original in views[0].items():
            assert torch.equal(
                original[source[:target_count]].cpu(),
                views[1][key][destination[:target_count]].cpu(),
            )
            assert views[1][key][destination[target_count:]].count_nonzero().item() == 0
        stats = await directory.get_stats()
        assert (
            stats["imported_tokens"]
            == (prompt_tokens if producer == "vllm" else prompt_tokens - 1)
            and stats["active_handoffs"] == 0
        )
        for actor in actors:
            stats = await actor.get_stats()
            assert stats["cpu_batches"] == stats["active_transfers"] == 0
    finally:
        for actor in actors:
            await xo.destroy_actor(actor)
        await pool.stop()
