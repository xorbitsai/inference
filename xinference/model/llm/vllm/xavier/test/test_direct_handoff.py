# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
import xoscar as xo

from ..direct_handoff import DirectGPUTransfer
from ..request_transfer import LayerRead
from .test_gpu_transfer import runtime


def direct_runtime(monkeypatch):
    r = runtime(monkeypatch, gpu_slots=0)
    r.__class__ = DirectGPUTransfer
    r.direct_requests, r.finished_sending = {}, set()
    r.metrics.update(direct_registered=0, direct_finished=0, direct_expired=0)
    return r


def peer_for(source):
    async def send(*args):
        return await source.run(source.send_direct, *args)

    return SimpleNamespace(
        send_direct_gpu_v1=AsyncMock(side_effect=send),
        release_direct_gpu_v1=AsyncMock(side_effect=source.release_direct),
    )


@pytest.mark.asyncio
async def test_direct_handoff_preserves_bits_and_uses_no_snapshots(monkeypatch):
    source, dest = direct_runtime(monkeypatch), direct_runtime(monkeypatch)
    source.caches["K"].view(torch.int16).copy_(
        torch.tensor(
            [
                [0, -32768],
                [32767, -1],
                [32640, -128],
                [1, -32767],
                [8, 9],
                [10, 11],
                [12, 13],
                [14, 15],
            ],
            dtype=torch.int16,
        )
    )
    source.register_direct("ticket", "producer", [1, 2])
    peer = peer_for(source)
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=peer))

    async def copy(buffers, refs):
        refs[0].copy_(buffers[0])

    monkeypatch.setattr(xo, "copy_to", copy)
    await dest.run(dest.load_direct, [{0: {"K": {1: 5, 2: 3}}}], ["ticket"])
    assert torch.equal(
        dest.caches["K"].view(torch.int16)[[5, 3]],
        source.caches["K"].view(torch.int16)[[1, 2]],
    )
    assert not source.store.blocks and not dest.store.blocks
    assert source.poll_direct() == {"producer"}
    assert not source.poll_direct() and not source.direct_requests


@pytest.mark.asyncio
async def test_release_during_read_waits_for_copy_and_fence(monkeypatch):
    source = direct_runtime(monkeypatch)
    source.register_direct("ticket", "producer", [1])
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def copy(buffers, refs):
        entered.set()
        await release.wait()
        refs[0].copy_(buffers[0])
        calls.append("copy")

    monkeypatch.setattr(xo, "copy_to", copy)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("fence"))
    read = LayerRead("K", [1], [0], (2,), torch.bfloat16)
    task = asyncio.create_task(
        source.run(
            source.send_direct, "ticket", [read], torch.zeros(16, dtype=torch.uint8), 16
        )
    )
    await asyncio.wait_for(entered.wait(), 2)
    source.release_direct("ticket")
    assert not source.poll_direct()
    release.set()
    await task
    assert calls == ["fence", "copy", "fence"]
    assert source.poll_direct() == {"producer"}


@pytest.mark.asyncio
async def test_expired_request_cannot_read_reused_engine_blocks(monkeypatch):
    source = direct_runtime(monkeypatch)
    source.register_direct("ticket", "producer", [1])
    source.direct_requests["ticket"].deadline = 0
    assert source.poll_direct() == {"producer"}
    with pytest.raises(RuntimeError, match="expired"):
        await source.send_direct(
            "ticket", [LayerRead("K", [1], [0], (2,), torch.bfloat16)], None, 16
        )
    assert not source.metrics["wire_bytes"]


@pytest.mark.asyncio
async def test_cancelled_decoder_drains_before_releasing_source(monkeypatch):
    source, dest = direct_runtime(monkeypatch), direct_runtime(monkeypatch)
    source.caches["K"].fill_(7)
    source.register_direct("ticket", "producer", [1])
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=peer_for(source)))
    entered, release = asyncio.Event(), asyncio.Event()

    async def copy(buffers, refs):
        entered.set()
        await release.wait()
        refs[0].copy_(buffers[0])

    monkeypatch.setattr(xo, "copy_to", copy)
    task = asyncio.create_task(
        dest.run(dest.load_direct, [{0: {"K": {1: 5}}}], ["ticket"])
    )
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not source.poll_direct()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert dest.caches["K"][5].tolist() == [7, 7]
    assert source.poll_direct() == {"producer"}


def test_direct_metadata_preserves_local_prefix_offset(connector):
    connector._direct_test = True
    request = SimpleNamespace(
        request_id="decoder",
        kv_transfer_params={
            "xavier_direct": {
                "rank": 0,
                "ticket": "ticket",
                "tokens": 32,
                "blocks": [2, 3],
            }
        },
    )
    assert connector.get_num_new_matched_tokens(request, 16) == (16, True)
    blocks = SimpleNamespace(get_block_ids=lambda: [[4, 5]])
    connector.update_state_after_alloc(request, blocks, 16)
    load = connector._requests_need_load["decoder"]
    assert load.local_transfers_by_group == {0: {0: {3: 5}}}


@pytest.mark.parametrize("prompt_tokens", [33, 30])
def test_direct_producer_retains_original_engine_blocks(
    connector, monkeypatch, prompt_tokens
):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.request",
        SimpleNamespace(RequestStatus=SimpleNamespace(FINISHED_ABORTED="aborted")),
    )
    connector._direct_test = True
    actor = SimpleNamespace(register_direct_gpu_v1=AsyncMock())
    connector._get_transfer_ref = AsyncMock(return_value=actor)
    request = SimpleNamespace(
        request_id="producer",
        prompt_token_ids=list(range(prompt_tokens)),
        status="finished",
        kv_transfer_params={"do_remote_decode": True},
    )
    retained, metadata = connector.request_finished(request, [4, 5, 6])
    assert retained
    handoff = metadata["xavier_direct"]
    assert handoff["blocks"] == [4, 5] and handoff["tokens"] == prompt_tokens - 1
    actor.register_direct_gpu_v1.assert_awaited_once_with(
        handoff["ticket"], "producer", [4, 5]
    )
