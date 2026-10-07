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
    await actor.reserve(1, c.to_dict(), c.block_size + 1)
    await actor.publish(1, c.to_dict(), pages, 7, c.block_size + 1)
    directory.publish_source.assert_awaited_once_with(
        1, dict(address="mac:1234", uid="source", rank=1, transport="host")
    )
    waiting = asyncio.create_task(actor.reserve(2, c.to_dict(), c.block_size + 1))
    await asyncio.sleep(0)
    assert not waiting.done()
    reply = actor.read(1)
    actor.abort(1)
    assert reply["pages"] == tuple(pages)  # In-flight immutable RPC stays valid.
    assert actor.get_stats()["active_bytes"] == 0
    await waiting
    await actor.publish(2, c.to_dict(), pages, 7, c.block_size + 1)
    actor.rooms[2]["deadline"] = 0
    with pytest.raises(RuntimeError, match="expired"):
        actor.read(2)
    assert not actor.rooms and actor.active_bytes == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["token", "page", "count", "contract", "publish"])
async def test_host_source_publish_failure_rolls_back_reservation(kv_contract, failure):
    c = kv_contract
    size = c.layer_nbytes * c.num_layers
    directory = AsyncMock()
    directory.request_info.return_value = dict(prompt_tokens=1)
    actor = XavierHostPDSource(directory, 1, capacity_bytes=size)
    actor.address, actor.uid = "mac", "source"
    await actor.reserve(1, c.to_dict(), 1)
    metadata, pages, token = c.to_dict(), [b"a" * size], 7
    if failure == "token":
        token = -1
    elif failure == "page":
        pages = [b"short"]
    elif failure == "count":
        pages = []
    elif failure == "contract":
        metadata["weights_fingerprint"] = "1" * 64
    else:
        directory.publish_source.side_effect = RuntimeError("publish failed")
    with pytest.raises((ValueError, RuntimeError)):
        await actor.publish(1, metadata, pages, token, 1)
    assert not actor.rooms and actor.active_bytes == 0
    await actor.reserve(2, c.to_dict(), 1)
    assert actor.active_bytes == size
    actor.abort(2)


@pytest.mark.asyncio
async def test_host_source_rejects_oversize_before_admission_and_preserves_duplicates(
    kv_contract,
):
    c = kv_contract
    size = c.layer_nbytes * c.num_layers
    directory = AsyncMock()
    directory.request_info.return_value = dict(prompt_tokens=1)
    actor = XavierHostPDSource(directory, 1, capacity_bytes=size)
    actor.address, actor.uid = "mac", "source"
    with pytest.raises(ValueError, match="byte budget"):
        await actor.reserve(1, c.to_dict(), c.block_size + 1)
    directory.configure.assert_not_awaited()
    assert not actor.rooms and actor.active_bytes == 0
    await actor.reserve(1, c.to_dict(), 1)
    await actor.publish(1, c.to_dict(), [b"a" * size], 7, 1)
    with pytest.raises(ValueError, match="Duplicate"):
        await actor.publish(1, c.to_dict(), [b"a" * size], 7, 1)
    assert actor.read(1)["first_token"] == 7
    directory.configure.assert_awaited_once_with(c.fingerprint, c.to_dict())


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", ["task", "abort", "timeout", "expire"])
async def test_host_capacity_wait_is_bounded_and_cancellable(
    kv_contract, monkeypatch, cancel
):
    c = kv_contract
    size = c.layer_nbytes * c.num_layers
    directory = AsyncMock()
    directory.request_info.return_value = dict(prompt_tokens=1)
    actor = XavierHostPDSource(directory, 1, capacity_bytes=size)
    await actor.reserve(1, c.to_dict(), 1)
    if cancel in ("timeout", "expire"):
        monkeypatch.setenv("XINFERENCE_SGLANG_XAVIER_TRANSFER_TIMEOUT", "0.1")
    waiting = asyncio.create_task(actor.reserve(2, c.to_dict(), 1))
    while 2 not in actor._waiting:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    if cancel == "task":
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    elif cancel == "abort":
        actor.abort(2)
        with pytest.raises(RuntimeError, match="cancelled"):
            await waiting
    elif cancel == "timeout":
        with pytest.raises(TimeoutError, match="capacity wait"):
            await waiting
    else:
        actor.rooms[1]["deadline"] = 0
        actor._capacity_changed.set()
        await waiting
        assert list(actor.rooms) == [2]
        actor.abort(2)
    assert not actor._waiting
    actor.abort(1)
    assert actor.active_bytes == 0


@pytest.mark.asyncio
async def test_host_capacity_wait_allows_release_rpc(kv_contract):
    import xoscar as xo

    from ...sglang.xavier.directory import XavierPDDirectory

    c = kv_contract
    size = c.layer_nbytes * c.num_layers
    pool = await xo.create_actor_pool("127.0.0.1:0", n_process=0)
    async with pool:
        directory = await xo.create_actor(
            XavierPDDirectory, address=pool.external_address
        )
        await directory.configure(c.fingerprint, c.to_dict())
        for room in (1, 2):
            await directory.prepare(
                room, c.fingerprint, str(room), "prefill", prompt_tokens=1
            )
        actor = await xo.create_actor(
            XavierHostPDSource, directory, 1, size, address=pool.external_address
        )
        await actor.reserve(1, c.to_dict(), 1)
        await actor.publish(1, c.to_dict(), [b"a" * size], 7, 1)
        waiting = asyncio.create_task(actor.reserve(2, c.to_dict(), 1))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        await asyncio.wait_for(actor.release(1), 2)
        await asyncio.wait_for(waiting, 2)
        assert (await actor.get_stats())["active_bytes"] == size
        await actor.abort(2)
        assert (await actor.get_stats())["active_bytes"] == 0


@pytest.mark.asyncio
async def test_host_export_reads_prompt_once_outside_send_lock(kv_contract):
    actor = source(kv_contract)
    await actor.open(1)
    actor.rooms[1]["chunks"].append(dict(ticket="1:0", pages=[1, 2], final=True))
    state = SimpleNamespace(
        reading=False,
        claimed=False,
        released=False,
        layer_blocks={str(i): {1, 2} for i in range(4)},
    )
    actor.transfer = SimpleNamespace(
        send_lock=asyncio.Lock(), _live_direct=Mock(return_value=state), metrics={}
    )

    async def request_info(room):
        assert not actor.transfer.send_lock.locked()
        return dict(prompt_tokens=65)

    actor.directory.request_info.side_effect = request_info
    await actor.export_host_pages(1, 0, 0)
    await actor.export_host_pages(1, 0, 1)
    actor.directory.request_info.assert_awaited_once_with(1)


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
