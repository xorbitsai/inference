# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch

from ..backends.bytes.pd import XavierHostPDSource
from ..backends.torch.pd import CrossEngineGPUActor


def source(kv_contract):
    c = replace(kv_contract, num_layers=2)
    actor = CrossEngineGPUActor(None, c, AsyncMock(), c.fingerprint, 1)
    actor.caches = {
        str(i): torch.arange(
            10 * c.block_size * c.num_kv_heads * c.head_dim, dtype=torch.float16
        ).reshape(10, c.block_size, c.num_kv_heads, c.head_dim)
        + i
        for i in range(4)
    }
    return actor


def test_host_export_preserves_physical_order_and_clears_padding(kv_contract):
    actor = source(kv_contract)
    c = actor.contract
    original = {k: v.clone() for k, v in actor.caches.items()}
    pages = actor._export_host_pages([7, 2], 0, c.block_size + 1)
    data = np.stack(
        [
            np.frombuffer(p, dtype="<f2").reshape(
                c.num_layers, 2, c.block_size, c.num_kv_heads, c.head_dim
            )
            for p in pages
        ]
    )
    for layer in range(c.num_layers):
        for kind in (0, 1):
            expected = (
                actor.caches[str(layer + kind * c.num_layers)][[7, 2]].numpy().copy()
            )
            expected[1, 1:] = 0
            np.testing.assert_array_equal(data[:, layer, kind], expected)
    assert all(torch.equal(original[k], v) for k, v in actor.caches.items())
    extra = actor._export_host_pages([2], 2, c.block_size + 1)
    assert not any(extra[0])


@pytest.mark.asyncio
async def test_host_source_budget_expiry_and_direct_peer(kv_contract):
    c = kv_contract
    size = c.layer_nbytes * c.num_layers
    directory = AsyncMock()
    directory.request_info.return_value = dict(prompt_tokens=c.block_size + 1)
    actor = XavierHostPDSource(directory, 1, capacity_bytes=size * 2)
    actor.address, actor.uid = "mac:1234", "source"
    pages = [b"a" * size, b"b" * size]
    await actor.publish(1, c.to_dict(), pages, 7, c.block_size + 1)
    directory.publish_source.assert_awaited_once_with(
        1, dict(address="mac:1234", uid="source", rank=1, transport="host")
    )
    with pytest.raises(RuntimeError, match="budget"):
        await actor.publish(2, c.to_dict(), pages, 7, c.block_size + 1)
    reply = actor.read(1)
    actor.abort(1)
    assert reply["pages"] == tuple(pages)  # In-flight immutable RPC stays valid.
    assert actor.get_stats()["active_bytes"] == 0
    await actor.publish(2, c.to_dict(), pages, 7, c.block_size + 1)
    actor.rooms[2]["deadline"] = 0
    with pytest.raises(RuntimeError, match="expired"):
        actor.read(2)
    assert not actor.rooms and actor.active_bytes == 0


def test_host_import_preserves_destination_physical_order(kv_contract):
    actor = source(kv_contract)
    original = {k: v.clone() for k, v in actor.caches.items()}
    pages = actor._export_host_pages([7, 2], 0, actor.contract.block_size + 1)
    actor._import_host_pages(pages, [3, 8])
    for key, cache in actor.caches.items():
        assert torch.equal(cache[3], original[key][7])
        assert torch.equal(cache[8, :1], original[key][2, :1])
        assert not cache[8, 1:].count_nonzero()
        assert torch.equal(
            cache[[0, 1, 2, 4, 5, 6, 7, 9]], original[key][[0, 1, 2, 4, 5, 6, 7, 9]]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sglang,full_allocation", [(False, False), (False, True), (True, True)]
)
async def test_host_receive_full_prompt_or_prefix(
    kv_contract, monkeypatch, sglang, full_allocation
):
    actor = source(kv_contract)
    c = actor.contract
    actor.args = SimpleNamespace(aux_item_lens=[64] * 10 if sglang else [])
    actor.transfer = SimpleNamespace(metrics={})
    actor.directory.request_info.return_value = dict(prompt_tokens=c.block_size + 1)
    actor.directory.wait_source.return_value = dict(
        address="mac", uid="source", transport="host"
    )
    pages = actor._export_host_pages([7, 2], 0, c.block_size + 1)
    sender = AsyncMock()
    sender.read.return_value = dict(
        pages=pages, next=2, total=2, first_token=4, prompt_tokens=c.block_size + 1
    )
    monkeypatch.setattr("xoscar.actor_ref", AsyncMock(return_value=sender))
    spare = {k: v[8].clone() for k, v in actor.caches.items()}
    targets = [3, 8] if full_allocation else [3]
    result = await actor.receive(1, targets, 0)
    sender.release.assert_awaited_once_with(1)
    assert actor.transfer.metrics["host_imported_bytes"] == sum(map(len, pages))
    if sglang:
        nbytes, payload = result
        import struct

        assert struct.unpack_from("<i", payload[0])[0] == 4
        assert struct.unpack_from("<i", payload[1])[0] == c.block_size + 1
        actor.directory.complete.assert_not_awaited()
    else:
        assert result == set()
        assert all(torch.equal(v[8], spare[k]) for k, v in actor.caches.items())
        actor.directory.complete.assert_awaited_once_with(
            1, 0, c.block_size, sum(map(len, pages))
        )


@pytest.mark.asyncio
async def test_host_cancel_drains_destination_writes(kv_contract, monkeypatch):
    actor = source(kv_contract)
    actor.args = SimpleNamespace(aux_item_lens=[])
    actor.transfer = SimpleNamespace(metrics={}, send_lock=asyncio.Lock())
    actor.directory.wait_source.return_value = dict(
        address="mac", uid="source", transport="host"
    )
    actor.directory.request_info.return_value = dict(prompt_tokens=3)
    pages = actor._export_host_pages([7], 0, 3)
    sender = AsyncMock()
    sender.read.return_value = dict(
        pages=pages, next=1, total=1, first_token=4, prompt_tokens=3
    )
    monkeypatch.setattr("xoscar.actor_ref", AsyncMock(return_value=sender))
    started, finish = threading.Event(), threading.Event()

    def copy(*args):
        started.set()
        assert finish.wait(10)

    actor._import_host_pages = copy
    incoming = asyncio.create_task(actor.receive(1, [3], 0))
    assert await asyncio.to_thread(started.wait, 10)
    for _ in range(2):
        incoming.cancel()
        await asyncio.sleep(0)
    abort = asyncio.create_task(actor.abort(1))
    await asyncio.sleep(0.01)
    assert not incoming.done() and not abort.done()
    actor.directory.complete.assert_not_awaited()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await incoming
    await abort
    sender.abort.assert_awaited_once_with(1)
    assert not actor.tasks


@pytest.mark.asyncio
async def test_host_truncated_read_is_not_committed(kv_contract, monkeypatch):
    actor = source(kv_contract)
    actor.args = SimpleNamespace(aux_item_lens=[])
    actor.transfer = SimpleNamespace(metrics={})
    actor.directory.request_info.return_value = dict(prompt_tokens=3)
    sender = AsyncMock()
    sender.read.return_value = dict(
        pages=[b"truncated"], next=1, total=1, first_token=4, prompt_tokens=3
    )
    monkeypatch.setattr("xoscar.actor_ref", AsyncMock(return_value=sender))
    original = {k: v.clone() for k, v in actor.caches.items()}
    with pytest.raises(ValueError, match="Incomplete"):
        await actor._receive_host(1, dict(address="mac", uid="source"), [3])
    assert all(torch.equal(original[k], v) for k, v in actor.caches.items())
    actor.directory.complete.assert_not_awaited()
    sender.abort.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_host_cancel_keeps_source_pinned_until_cpu_copy_drains(kv_contract):
    actor = source(kv_contract)
    chunk = dict(ticket="1:0", pages=[1], final=True)
    actor.address = "gpu:1234"
    await actor.open(1)
    actor.rooms[1]["chunks"].append(chunk)
    state = SimpleNamespace(
        reading=False,
        claimed=False,
        released=False,
        layer_blocks={str(i): {1} for i in range(4)},
    )
    actor.transfer = SimpleNamespace(
        send_lock=asyncio.Lock(),
        _live_direct=Mock(return_value=state),
        release_direct=Mock(),
        metrics={},
    )
    actor.directory.request_info.return_value = dict(prompt_tokens=3)
    started, drained = threading.Event(), threading.Event()

    def copy(*args):
        started.set()
        assert drained.wait(10)
        return [b"owned"]

    actor._export_host_pages = copy
    task = asyncio.create_task(actor.export_host_pages(1, 0))
    assert await asyncio.to_thread(started.wait, 10)
    task.cancel()
    abort = asyncio.create_task(actor.abort(1))
    await asyncio.sleep(0.01)
    assert state.reading and not abort.done()
    actor.transfer.release_direct.assert_not_called()
    drained.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await abort
    assert not state.reading and not actor.rooms
    actor.transfer.release_direct.assert_called_once_with("1:0")


@pytest.mark.asyncio
async def test_host_export_rejects_expired_ticket_and_cursor(kv_contract):
    actor = source(kv_contract)
    actor.rooms[1] = dict(chunks=[dict(ticket="1:0", pages=[1], final=True)])
    actor.transfer = SimpleNamespace(
        send_lock=asyncio.Lock(), _live_direct=Mock(return_value=None)
    )
    with pytest.raises(ValueError, match="cursor"):
        await actor.export_host_pages(1, 0, -1)
    with pytest.raises(RuntimeError, match="expired"):
        await actor.export_host_pages(1, 0)
