# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
import ctypes
import json
import logging
import queue
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from ..request_transfer import LayerRead, batch_reads, pack_reads, unpack_reads
from ..snapshot import KVSnapshotStore
from ..transfer import TransferActor


def test_batch_limits_and_dtype_boundaries(monkeypatch):
    from .. import request_transfer as module

    monkeypatch.setattr(module, "MAX_REQUEST_BYTES", 16)
    reads = [
        LayerRead("K", list(range(10)), list(range(10)), (1,), torch.float32),
        LayerRead("V", [0], [0], (1,), torch.float16),
    ]
    batches = list(batch_reads(reads))
    assert [sum(r.nbytes for r in b) for b in batches] == [16, 16, 8, 2]
    assert [k for b in batches for r in b if r.layer == "K" for k in r.keys] == list(
        range(10)
    )
    big = LayerRead("large", [1, 2], [4, 5], (8,), torch.float32)
    assert len(list(batch_reads([big]))) == 2


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_cross_layer_round_trip(dtype):
    store = KVSnapshotStore(3)
    source = torch.arange(12).reshape(3, 4).to(dtype)
    for name in ["K", "V"]:
        store.stage(name, [1, 2, 3], source)
    store.publish([1, 2, 3], {"K", "V"})
    reads = [
        LayerRead("K", [3, 1], [0, 2], (4,), dtype),
        LayerRead("V", [2], [1], (4,), dtype),
    ]
    payload = pack_reads(store, reads)
    decoded = list(unpack_reads(payload, reads))
    assert torch.equal(decoded[0][1], source[[2, 0]])
    assert torch.equal(decoded[1][1], source[[1]])
    with pytest.raises(ValueError, match="payload"):
        list(unpack_reads(payload[:-1], reads))
    with pytest.raises(ValueError, match="layout"):
        pack_reads(store, [LayerRead("K", [1], [0], (3,), dtype)])
    with pytest.raises(KeyError):
        pack_reads(store, [LayerRead("K", [99], [0], (4,), dtype)])


@pytest.mark.asyncio
async def test_actual_actor_cross_layer_read(monkeypatch, caplog):
    from .. import profiling

    monkeypatch.setattr(profiling, "_ENABLED", True)
    caplog.set_level(logging.INFO, logger=profiling.__name__)
    import xoscar as xo
    from xoscar.collective import xoscar_pygloo as xp

    wire = queue.Queue()
    calls = []

    def send(ctx, ptr, count, dtype, rank):
        calls.append(count)
        wire.put(ctypes.string_at(ptr, count))

    def recv(ctx, ptr, count, dtype, rank):
        data = wire.get(timeout=5)
        assert len(data) == count
        ctypes.memmove(ptr, data, count)

    monkeypatch.setattr(xp, "send", send)
    monkeypatch.setattr(xp, "recv", recv)
    store = KVSnapshotStore(1)
    for layer in ["K", "V"]:
        store.stage(layer, [7], torch.tensor([[1.0, 2.0]]))
    store.publish([7], {"K", "V"})
    sender = SimpleNamespace(
        _snapshot_store=store,
        _context=None,
        _layer_send_tasks_v1=set(),
        get_gloo_dtype=lambda t: t,
    )
    ref = SimpleNamespace(
        start_send_request_blocks_v1=AsyncMock(
            side_effect=MethodType(TransferActor.start_send_request_blocks_v1, sender)
        )
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=ref))
    receiver = SimpleNamespace(
        _world_addresses=["sender"], _rank=1, _context=None, get_gloo_dtype=lambda t: t
    )
    reads = [LayerRead(name, [7], [0], (2,), torch.float32) for name in ["K", "V"]]
    payload = await TransferActor.read_request_blocks_v1(receiver, 0, reads)
    await asyncio.gather(*sender._layer_send_tasks_v1)
    assert calls == [16]
    events = [
        json.loads(r.message.split("Xavier profile: ")[1])
        for r in caplog.records
        if r.message.startswith("Xavier profile: ")
    ]
    assert {e["stage"] for e in events} == {
        "actor_control",
        "actor_receive",
        "gloo_receive",
    }
    assert all(
        e["nbytes"] == 16 and e["blocks"] == 2 and e["succeeded"] for e in events
    )
    assert all(
        torch.equal(t, torch.tensor([[1.0, 2.0]]))
        for _, t in unpack_reads(payload, reads)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_connector_writes_correct_destinations(
    connector, connector_module, dtype, monkeypatch, caplog
):
    from .. import profiling

    monkeypatch.setattr(profiling, "_ENABLED", True)
    caplog.set_level(logging.INFO, logger=profiling.__name__)
    caches = {name: torch.zeros(8, 4, dtype=dtype) for name in ["K", "V"]}
    connector._registered_kv_caches = caches
    source = KVSnapshotStore(1)
    producer = SimpleNamespace(_snapshot_store=source, _rank=2)
    values = torch.tensor([[1.0, -2.0, 1e10, 1e-10]], dtype=dtype)
    for name in caches:
        TransferActor.stage_layer_blocks_v1(producer, "r", name, [77], values)
        assert source.read(name, [77]).dtype == (
            torch.float16 if dtype == torch.bfloat16 else dtype
        )
    source.publish([77], set(caches))
    transfer = SimpleNamespace(
        read_request_blocks_v1=AsyncMock(
            side_effect=lambda rank, reads: pack_reads(source, reads)
        )
    )
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    request = connector_module.XavierLoadRequest(
        "r", {2: {77: 3}}, local_transfers_by_group={0: {2: {77: 3}}}
    )
    connector._load_request_blocks(request)
    assert transfer.read_request_blocks_v1.await_count == 1
    events = [
        json.loads(r.message.split("Xavier profile: ")[1])
        for r in caplog.records
        if r.message.startswith("Xavier profile: ")
    ]
    rpc = [e for e in events if e["stage"] == "load_rpc"]
    h2d = [e for e in events if e["stage"] == "load_h2d"]
    expected_bytes = values.numel() * values.element_size()
    assert (
        len(rpc) == 1
        and rpc[0]["nbytes"] == 2 * expected_bytes
        and rpc[0]["blocks"] == 2
    )
    assert len(h2d) == 2
    assert all(
        e["nbytes"] == expected_bytes and e["blocks"] == 1 and e["succeeded"]
        for e in h2d
    )
    for cache in caches.values():
        assert torch.equal(cache[3], values[0])
        assert cache[:3].count_nonzero() == 0


def test_payload_preserves_all_bf16_bit_patterns():
    # Raw packing preserves the bits independently of the staging carrier.
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(8, 8192)
    store = KVSnapshotStore(8)
    store.stage("K", list(range(8)), bits.view(torch.bfloat16))
    store.publish(list(range(8)), {"K"})
    reads = [LayerRead("K", list(range(8)), list(range(8)), (8192,), torch.bfloat16)]
    _, restored = next(unpack_reads(pack_reads(store, reads), reads))
    assert torch.equal(restored.view(torch.int16), bits)


@pytest.mark.asyncio
async def test_cancelled_receive_keeps_buffer_alive(monkeypatch):
    import gc
    import threading
    import weakref
    from contextlib import suppress

    import xoscar as xo
    from xoscar.collective import xoscar_pygloo as xp

    entered, release, finished = (threading.Event() for _ in range(3))
    buffers = []
    failures = []
    empty = torch.empty

    def allocate(*args, **kwargs):
        tensor = empty(*args, **kwargs)
        buffers.append(weakref.ref(tensor))
        return tensor

    def recv(*args):
        entered.set()
        try:
            if not release.wait(5):
                failures.append("receive was not released")
            if buffers[0]() is None:
                failures.append("buffer freed while native receive was active")
        finally:
            finished.set()

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(xp, "recv", recv)
    monkeypatch.setattr(
        xo,
        "actor_ref",
        AsyncMock(
            return_value=SimpleNamespace(start_send_request_blocks_v1=AsyncMock())
        ),
    )
    receiver = SimpleNamespace(
        _world_addresses=["sender"], _rank=1, _context=None, get_gloo_dtype=lambda t: t
    )
    reads = [LayerRead("K", [1], [0], (2,), torch.float32)]
    task = asyncio.create_task(TransferActor.read_request_blocks_v1(receiver, 0, reads))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        del task
        gc.collect()
        assert buffers[0]() is not None
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
    assert not failures


def test_block_limit_after_byte_limit_split(monkeypatch):
    from .. import request_transfer as module

    monkeypatch.setattr(module, "MAX_REQUEST_BYTES", 20)
    read = LayerRead("K", list(range(150)), list(range(150)), (1,), torch.float32)
    parts = [part for batch in batch_reads([read]) for part in batch]
    assert [key for part in parts for key in part.keys] == list(range(150))
    assert all(len(part.keys) <= module.MAX_REQUEST_BLOCKS for part in parts)


def test_block_limit_with_large_byte_budget(monkeypatch):
    from .. import request_transfer as module

    monkeypatch.setattr(module, "MAX_REQUEST_BYTES", 4096)
    read = LayerRead("K", list(range(150)), list(range(150)), (1,), torch.float32)
    batches = list(batch_reads([read]))
    assert [len(part.keys) for batch in batches for part in batch] == [64, 64, 22]
    assert all(sum(part.nbytes for part in batch) <= 4096 for batch in batches)


def test_connector_cross_layer_preserves_all_bf16_bits(connector, connector_module):
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(1, -1)
    values = bits.view(torch.bfloat16)
    cache = torch.zeros(8, 65536, dtype=torch.bfloat16)
    connector._registered_kv_caches = {"K": cache}
    source = KVSnapshotStore(1)
    producer = SimpleNamespace(_snapshot_store=source, _rank=2)
    TransferActor.stage_layer_blocks_v1(producer, "r", "K", [77], values)
    source.publish([77], {"K"})
    transfer = SimpleNamespace(
        read_request_blocks_v1=AsyncMock(
            side_effect=lambda rank, reads: pack_reads(source, reads)
        )
    )
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    request = connector_module.XavierLoadRequest(
        "r", {2: {77: 3}}, local_transfers_by_group={0: {2: {77: 3}}}
    )
    connector._load_request_blocks(request)
    assert torch.equal(cache[3].view(torch.int16), bits[0])


@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float16])
def test_cross_layer_rejects_logical_dtype_mismatch_after_split(
    source_dtype, monkeypatch
):
    from .. import request_transfer as module

    monkeypatch.setattr(module, "MAX_REQUEST_BYTES", 2)
    source = KVSnapshotStore(2)
    producer = SimpleNamespace(_snapshot_store=source, _rank=2)
    TransferActor.stage_layer_blocks_v1(
        producer, "r", "K", [1, 2], torch.ones(2, 1, dtype=source_dtype)
    )
    source.publish([1, 2], {"K"})
    other_dtype = torch.float16 if source_dtype == torch.bfloat16 else torch.bfloat16
    read = LayerRead("K", [1, 2], [3, 4], (1,), torch.float16, other_dtype)
    batches = list(batch_reads([read]))
    assert len(batches) == 2
    for batch in batches:
        assert batch[0].logical_dtype == other_dtype
        with pytest.raises(ValueError, match="logical KV dtype"):
            pack_reads(source, batch)
        batch[0].logical_dtype = source_dtype
        assert pack_reads(source, batch).numel() == 2
