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
    r.metrics.update(
        direct_registered=0, direct_finished=0, direct_expired=0, index_uploads=0
    )
    r._init_history(0)
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
    state = source.direct_requests["ticket"]
    source.release_direct("ticket")
    assert state.indices
    assert not source.poll_direct()
    release.set()
    await task
    assert calls == ["fence", "copy", "fence"]
    assert not state.indices
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
    connector._direct_handoff = True
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
    connector._direct_handoff = True
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


@pytest.mark.asyncio
async def test_indices_reused_across_layers_and_slabs_but_not_requests(monkeypatch):
    source, dest = direct_runtime(monkeypatch), direct_runtime(monkeypatch)
    for layer in ("K", "V", "Q"):
        source.caches[layer] = torch.arange(16, dtype=torch.bfloat16).reshape(8, 2)
        dest.caches[layer] = torch.zeros_like(source.caches[layer])
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=peer_for(source)))

    async def copy(buffers, refs):
        refs[0].copy_(buffers[0])

    monkeypatch.setattr(xo, "copy_to", copy)
    for ticket, destinations in (("first", [5, 3]), ("second", [3, 5])):
        source.register_direct(ticket, ticket, [1, 2])
        state = source.direct_requests[ticket]
        mapping = {name: dict(zip([1, 2], destinations)) for name in source.caches}
        await dest.run(dest.load_direct, [{0: mapping}], [ticket])
        for layer in source.caches:
            assert torch.equal(
                dest.caches[layer][destinations], source.caches[layer][[1, 2]]
            )
        assert not state.indices
    assert source.metrics["index_uploads"] == dest.metrics["index_uploads"] == 2
    assert dest.metrics["gpu_batches"] == 4
    assert source.poll_direct() == {"first", "second"}


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["release", "expire", "close"])
async def test_request_indices_reclaimed_on_lifecycle_end(monkeypatch, finish):
    source = direct_runtime(monkeypatch)
    source.register_direct("ticket", "request", [1, 2])
    state = source.direct_requests["ticket"]
    index = source._index_tensor(state.indices, [1, 2])
    assert source._index_tensor(state.indices, [1, 2]) is index
    assert source._index_tensor(state.indices, [2, 1]).tolist() == [2, 1]
    assert index.dtype == torch.long
    if finish == "release":
        source.release_direct("ticket")
    elif finish == "expire":
        state.deadline = 0
        source.poll_direct()
    else:
        await source.close()
    assert not source.direct_requests
    assert not state.indices


@pytest.mark.asyncio
async def test_failed_later_slab_releases_both_index_sets(monkeypatch):
    source, dest = direct_runtime(monkeypatch), direct_runtime(monkeypatch)
    for layer in ("V", "Q"):
        source.caches[layer] = source.caches["K"].clone()
        dest.caches[layer] = dest.caches["K"].clone()
    source.register_direct("ticket", "producer", [1, 2])
    state = source.direct_requests["ticket"]
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=peer_for(source)))
    copies = 0
    receiver_indices = []
    original_index = dest._index_tensor

    def index(indices, values):
        receiver_indices.append(indices)
        return original_index(indices, values)

    async def copy(buffers, refs):
        nonlocal copies
        copies += 1
        if copies == 2:
            raise RuntimeError("transfer failed")
        refs[0].copy_(buffers[0])

    monkeypatch.setattr(dest, "_index_tensor", index)
    monkeypatch.setattr(xo, "copy_to", copy)
    mapping = {name: {1: 5, 2: 3} for name in source.caches}
    with pytest.raises(RuntimeError, match="transfer failed"):
        await dest.run(dest.load_direct, [{0: mapping}], ["ticket"])
    assert receiver_indices and all(not indices for indices in receiver_indices)
    assert not state.indices
    assert source.poll_direct() == {"producer"}


@pytest.mark.parametrize("role", ["prefill", "decode", "hybrid"])
@pytest.mark.parametrize("budget", [None, 0, 268435456])
def test_direct_handoff_selection_is_per_model(role, budget, monkeypatch):
    from ..transport import uses_direct_handoff

    # No environment setting is consulted, including the old prototype flag.
    monkeypatch.setenv("XINFERENCE_XAVIER_DIRECT_TEST", "1")
    assert uses_direct_handoff({"role": role, "gpu_cache_bytes": budget}) is (
        role != "hybrid" and budget is not None
    )
    assert not uses_direct_handoff(None)
