# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
import xoscar as xo
from xoscar.backends.allocate_strategy import ProcessIndex

from ..block_tracker import VLLMBlockTracker
from ..snapshot import KVSnapshotStore
from ..transfer import TransferActor


@pytest.mark.parametrize("wait", [False, True])
def test_exportless_completion_poll_needs_no_cpu_export_state(connector_module, wait):
    connector = connector_module.XavierConnector.__new__(
        connector_module.XavierConnector
    )
    # Cross-engine completion hooks have no CPU snapshots to publish. An empty
    # poll must not access queue/poll fields or enter an actor event loop.
    connector._ipc_export_jobs = {}
    connector._poll_ipc_exports(wait=wait)


class SnapshotExportActor(xo.StatelessActor):
    enqueue_snapshot_export_v1 = TransferActor.enqueue_snapshot_export_v1
    poll_snapshot_exports_v1 = TransferActor.poll_snapshot_exports_v1
    publish_blocks_v1 = TransferActor.publish_blocks_v1
    register_snapshot_export_source_v1 = (
        TransferActor.register_snapshot_export_source_v1
    )
    close_snapshot_export_source_v1 = TransferActor.close_snapshot_export_source_v1

    def __init__(self):
        self._rank = 1
        self._snapshot_store = KVSnapshotStore(8)
        self._snapshot_export_jobs_v1 = {}
        self._snapshot_export_lock_v1 = asyncio.Lock()

    def snapshot_bits(self, layer, keys):
        store = self._snapshot_store
        return store.read(layer, keys).view(torch.uint8).reshape(-1).tolist()

    async def __pre_destroy__(self):
        await asyncio.gather(
            *self._snapshot_export_jobs_v1.values(), return_exceptions=True
        )
        getattr(self, "_snapshot_export_sources_v1", {}).clear()


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(),
    reason="Linux CUDA required",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.asyncio
async def test_gpu_arena_reuse_and_fresh_registration_after_actor_recovery(dtype):
    from ....xavier.backends.torch.gpu_export import GPUExportArena

    pool = await xo.create_actor_pool(
        "127.0.0.1", n_process=1, subprocess_start_method="spawn"
    )
    async with pool:
        tracker = await xo.create_actor(
            VLLMBlockTracker, address=pool.external_address, uid="arena-tracker"
        )
        actor = await xo.create_actor(
            SnapshotExportActor,
            address=pool.external_address,
            uid="arena-export",
            allocate_strategy=ProcessIndex(1),
        )
        arena = GPUExportArena(4096, 1, torch.device("cuda", 0))
        expected = torch.ones(2, 2, dtype=dtype).view(torch.uint8).reshape(-1).tolist()
        try:
            for ticket, keys, fill in (("a", [111, 222], 1), ("b", [333, 444], 9)):
                slot, value = arena.allocate((2, 4))
                value.view(dtype).fill_(float(fill))
                descriptor = arena.record(slot, tuple(value.shape))
                batch = [(keys, descriptor, [("K", (2,), dtype, 0, 4)])]
                # Missing mappings consume no descriptors and start no writes.
                if ticket == "a":
                    assert (
                        await actor.enqueue_snapshot_export_v1(
                            ticket, batch, tracker.address, tracker.uid
                        )
                        is False
                    )
                    await actor.register_snapshot_export_source_v1(
                        arena.source, arena.metadata()
                    )
                await actor.enqueue_snapshot_export_v1(
                    ticket, batch, tracker.address, tracker.uid
                )
                result = await actor.poll_snapshot_exports_v1([ticket], wait=True)
                assert set(result[ticket][0]) == set(keys)
                arena.release([slot])
            assert await actor.snapshot_bits("K", [111, 222]) == expected
            await xo.destroy_actor(actor)
            actor = await xo.create_actor(
                SnapshotExportActor,
                address=pool.external_address,
                uid="arena-export",
                allocate_strategy=ProcessIndex(1),
            )
            slot, value = arena.allocate((2, 4))
            value.view(dtype).fill_(1)
            descriptor = arena.record(slot, tuple(value.shape))
            batch = [([555, 666], descriptor, [("K", (2,), dtype, 0, 4)])]
            assert (
                await actor.enqueue_snapshot_export_v1(
                    "c", batch, tracker.address, tracker.uid
                )
                is False
            )
            await actor.register_snapshot_export_source_v1(
                arena.source, arena.metadata()
            )
            await actor.enqueue_snapshot_export_v1(
                "c", batch, tracker.address, tracker.uid
            )
            result = await actor.poll_snapshot_exports_v1(["c"], wait=True)
            assert set(result["c"][0]) == {555, 666}
            arena.release([slot])
            await actor.close_snapshot_export_source_v1(arena.source)
        finally:
            arena.close()
        assert await actor.snapshot_bits("K", [555, 666]) == expected


def test_packed_export_preserves_mixed_dtypes_and_block_ownership(
    connector, connector_module, monkeypatch
):
    from torch.multiprocessing import reductions

    monkeypatch.setattr(reductions, "reduce_tensor", lambda value: (None, (value,)))
    transfer = SimpleNamespace(
        enqueue_snapshot_export_v1=AsyncMock(), poll_snapshot_exports_v1=AsyncMock()
    )
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    connector._registered_kv_caches = {
        "K": torch.arange(8, dtype=torch.uint8).view(8, 1),
        "V": torch.tensor([1e30, -1e30] * 8).to(torch.bfloat16).view(8, 2),
    }
    connector._layer_group_ids = {"K": 0, "V": 0}
    request = connector_module.XavierStoreRequest("r", [2, 4], [111, 222], [[2, 4]])
    expected = connector._registered_kv_caches["V"][[2, 4]].view(torch.uint8).clone()
    connector._call(
        connector._queue_ipc_export(list(connector._cpu_export_batches([request])), 32)
    )
    ticket, batches, _, _ = transfer.enqueue_snapshot_export_v1.await_args.args
    keys, (packed,), layers = batches[0]
    assert packed.shape == (2, 16)  # dtype-aligned rows and layer offsets.
    store = KVSnapshotStore(2)
    store.stage_packed_blocks(keys, packed, layers)
    store.publish(keys, {"K", "V"})
    packed.zero_()
    assert store.read("K", keys).tolist() == [[2], [4]]
    assert torch.equal(store.read("V", keys).view(torch.uint8), expected)
    assert store.logical_dtypes[111]["V"] == torch.bfloat16
    assert store.blocks[111]["K"].untyped_storage().nbytes() == 16
    assert (
        store.blocks[111]["K"].untyped_storage().data_ptr()
        != store.blocks[222]["K"].untyped_storage().data_ptr()
    )
    transfer.poll_snapshot_exports_v1.return_value = {ticket: (keys, [])}
    connector._poll_ipc_exports(wait=True)
    assert not connector._ipc_export_jobs
    assert connector._exported_keys == set(keys)


def test_ipc_export_failure_releases_completed_owners_and_invalidates_mirror(connector):
    transfer = SimpleNamespace(
        poll_snapshot_exports_v1=AsyncMock(return_value={"bad": (None, "copy failed")})
    )
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    connector._ipc_export_jobs["bad"] = ({111}, 16, [torch.ones(16)])
    connector._exported_keys = {111, 222}
    with pytest.raises(RuntimeError, match="copy failed"):
        connector._poll_ipc_exports(wait=True)
    assert not connector._ipc_export_jobs and not connector._exported_keys


def test_new_exports_do_not_postpone_periodic_discovery_refresh(connector):
    transfer = SimpleNamespace(
        poll_snapshot_exports_v1=AsyncMock(return_value={"new": ([111], [])})
    )
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    connector._ipc_export_jobs["new"] = ({111}, 16, [torch.ones(16)])
    connector._export_refresh_at = (
        1.0  # already due, even under continuous cold traffic
    )
    connector._poll_ipc_exports(wait=True)
    assert connector._export_refresh_at == 1.0


def test_lost_enqueue_ack_keeps_ipc_producer_storage(
    connector, connector_module, monkeypatch
):
    from torch.multiprocessing import reductions

    monkeypatch.setattr(reductions, "reduce_tensor", lambda value: (None, (value,)))
    transfer = SimpleNamespace(
        enqueue_snapshot_export_v1=AsyncMock(
            side_effect=RuntimeError("connection lost")
        ),
        poll_snapshot_exports_v1=AsyncMock(),
    )
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    connector._registered_kv_caches = {"K": torch.ones(8, 2)}
    request = connector_module.XavierStoreRequest("r", [2], [111], [[2]])
    with pytest.raises(RuntimeError, match="connection lost"):
        connector._call(
            connector._queue_ipc_export(
                list(connector._cpu_export_batches([request])), 8
            )
        )
    ticket = transfer.enqueue_snapshot_export_v1.await_args.args[0]
    assert connector._ipc_export_jobs[ticket][2][0].numel() == 8
    transfer.poll_snapshot_exports_v1.return_value = {ticket: ([111], [])}
    connector._poll_ipc_exports(wait=True)
    assert not connector._ipc_export_jobs


@pytest.mark.parametrize("lost_ack", [False, True])
def test_async_arena_submission_excludes_unacknowledged_jobs_and_retains_owners(
    connector, connector_module, lost_ack
):
    gate = asyncio.Event()
    actor_jobs = {}

    async def enqueue(ticket, batches, *_):
        actor_jobs[ticket] = batches
        await gate.wait()
        if lost_ack:
            raise RuntimeError("lost acknowledgement")

    async def poll(tickets, *, wait):
        assert all(ticket in actor_jobs for ticket in tickets)
        return {ticket: ([111], []) for ticket in tickets}

    transfer = SimpleNamespace(
        enqueue_snapshot_export_v1=AsyncMock(side_effect=enqueue),
        poll_snapshot_exports_v1=AsyncMock(side_effect=poll),
        close_snapshot_export_source_v1=AsyncMock(),
    )
    arena = SimpleNamespace(source="source", release=Mock(), close=Mock())
    connector._gpu_export_arena = arena
    connector._gpu_export_registered = True
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    packed = torch.ones(1, 8, dtype=torch.uint8)
    indices = torch.tensor([2])
    entries = connector_module._PackedExportEntries(
        [111], packed, [("K", (8,), torch.uint8, 0, 8)], slot=0
    )
    connector._enqueue_ipc_export([(entries, [indices], [])], 8)
    ticket = next(iter(connector._ipc_export_enqueues))
    assert not connector._ipc_export_enqueues[ticket].done()
    assert connector._ipc_export_jobs[ticket][2] == [packed, indices]
    connector._poll_ipc_exports()
    transfer.poll_snapshot_exports_v1.assert_not_awaited()
    arena.release.assert_not_called()
    gate.set()
    if lost_ack:
        with pytest.raises(RuntimeError, match="lost acknowledgement"):
            connector._poll_ipc_exports(wait=True)
        assert connector._ipc_export_jobs[ticket][2] == [packed, indices]
        assert connector._gpu_export_tickets[ticket] == [0]
        arena.release.assert_not_called()
        # The source-close handshake must precede freeing its GPU owners.
        connector._transfer_ref = transfer
        transfer.close_snapshot_export_source_v1.side_effect = (
            lambda _: arena.close.assert_not_called()
        )
        connector.shutdown()
        transfer.close_snapshot_export_source_v1.assert_awaited_once_with("source")
        arena.close.assert_called_once()
    else:
        connector._poll_ipc_exports(wait=True)
        assert not connector._ipc_export_jobs and not connector._ipc_export_enqueues
        assert connector._exported_keys == {111}
        arena.release.assert_called_once_with([0])


def test_failed_submission_drain_waits_for_all_late_tasks(connector):
    completed = []

    async def submit(ticket):
        await asyncio.sleep(0)
        completed.append(ticket)
        if ticket == "bad":
            raise RuntimeError("lost acknowledgement")

    connector._call(asyncio.sleep(0))
    for ticket in ("bad", "late"):
        connector._ipc_export_enqueues[ticket] = connector._loop.create_task(
            submit(ticket)
        )
        connector._ipc_export_jobs[ticket] = ({111}, 8, [torch.ones(8)])
    with pytest.raises(RuntimeError, match="lost acknowledgement"):
        connector._poll_ipc_exports(wait=True)
    assert completed == ["bad", "late"] and not connector._ipc_export_enqueues
    assert set(connector._ipc_export_jobs) == {"bad", "late"}
    connector._ipc_export_jobs.clear()


def test_packed_snapshot_respects_leases_and_immutable_partial_content():
    store = KVSnapshotStore(1)
    layers = [("K", (1,), torch.float32, 0, 4), ("V", (1,), torch.float32, 4, 4)]
    payload = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).view(torch.uint8)
    store.stage("K", [111], torch.tensor([[99.0]]))
    store.stage_packed_blocks([111], payload[:1], layers)
    store.publish([111], {"K", "V"})
    assert store.read("K", [111]).item() == 99.0
    assert store.read("V", [111]).item() == 2.0
    assert store.reserve("reader", [111])
    store.stage_packed_blocks([222], payload[1:], layers)
    assert store.ready == {111}
    assert set(store.blocks) == {111}
    store.release("reader")
    store.stage_packed_blocks([222], payload[1:], layers)
    assert store.publish([222], {"K", "V"}) == [222]
    assert store.ready == {222} and store.evicted == {111}


def test_retained_slabs_charge_and_evict_whole_allocations_with_leases():
    store = KVSnapshotStore(4)
    layers = [("K", (2,), torch.float32, 0, 8)]
    payloads = [torch.full((2, 2), float(i)).view(torch.uint8) for i in range(3)]
    for keys, payload in zip(([1, 2], [3, 4]), payloads):
        store.stage_packed_blocks(keys, payload, layers, retain_storage=True)
        store.publish(keys, {"K"})
    assert store._used_slots() == 4
    assert store.reserve("reader", [1])
    store.stage_packed_blocks([5, 6], payloads[2], layers, retain_storage=True)
    store.publish([5, 6], {"K"})
    assert store.ready == {1, 2, 5, 6}
    assert store.evicted == {3, 4}
    assert store._used_slots() == 4
    assert torch.equal(store.read("K", [1, 2]), payloads[0].view(torch.float32))
    assert store.blocks[5]["K"].untyped_storage().data_ptr() == (
        payloads[2].untyped_storage().data_ptr()
    )
    store.release("reader")
    store.stage("K", [7], torch.ones(1, 2))
    assert set(store.blocks) == {5, 6, 7}
    assert store._used_slots() == 3


def test_retained_payload_charges_backing_storage_and_falls_back_when_oversized():
    store = KVSnapshotStore(4)
    layers = [("K", (2,), torch.float32, 0, 8)]
    backing = torch.ones(10, 2).view(torch.uint8)
    store.stage_packed_blocks([1, 2], backing[:2], layers, retain_storage=True)
    store.publish([1, 2], {"K"})
    backing.zero_()
    assert store.read("K", [1, 2]).tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert store._used_slots() == 2 and not store._packed_slabs
    assert store.blocks[1]["K"].untyped_storage().nbytes() == 8


def test_packed_reads_preserve_order_and_bits_without_materializing_layer_views():
    store = KVSnapshotStore(2)
    payload = torch.tensor([[1e30, -1e30], [3e30, -3e30]]).to(torch.bfloat16)
    packed = payload.view(torch.uint8)
    layers = [("K", (2,), torch.bfloat16, 0, 4)]
    store.stage_packed_blocks([1, 2], packed, layers, retain_storage=True)
    store.publish([1, 2], {"K"})
    result = store.read("K", [2, 1])
    assert torch.equal(result.view(torch.uint8), packed[[1, 0]])
    assert all(not block._layers for block in store.blocks.values())
    # MutableMapping updates must not keep using the original packed bytes.
    store.blocks[1]["K"] = torch.zeros(2, dtype=torch.float16)
    assert store.read("K", [1, 2])[0].count_nonzero() == 0


def test_compact_blocks_share_one_slab_tensor_and_preserve_mapping_mutations():
    store = KVSnapshotStore(2)
    packed = torch.arange(8).reshape(2, 4).float().view(torch.uint8)
    layers = [("K", (2,), torch.float32, 0, 8), ("V", (2,), torch.float32, 8, 8)]
    store.stage_packed_blocks([1, 2], packed, layers, retain_storage=True)
    first, second = store.blocks.values()
    assert first._packed is packed and second._packed is packed
    assert [first._row, second._row] == [0, 1]
    assert not first._layers and first._overrides is None and first._removed is None
    first["K"] = torch.tensor([99.0, 100.0])
    assert store.read("K", [1, 2]).tolist() == [[99.0, 100.0], [4.0, 5.0]]
    del first["V"]
    assert set(first) == {"K"} and len(first) == 1
    assert store.publish([1, 2], {"K", "V"}) == [2]
    first["V"] = torch.tensor([88.0, 89.0])
    assert set(first) == {"K", "V"} and len(first) == 2
    assert store.publish([1], {"K", "V"}) == [1]


def test_packed_arena_submission_does_not_create_unused_layer_tensor_views(
    connector, connector_module, monkeypatch
):
    transfer = SimpleNamespace(enqueue_snapshot_export_v1=AsyncMock())
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    connector._gpu_export_arena = SimpleNamespace(source="source")
    connector._gpu_export_registered = True
    packed = torch.ones(2, 16, dtype=torch.uint8)

    def unexpected_view(*_):
        pytest.fail("Packed IPC metadata must not materialize layer tensor views")

    monkeypatch.setattr(torch.Tensor, "__getitem__", unexpected_view)
    try:
        entries = connector_module._PackedExportEntries(
            [111, 222], packed, [("K", (4,), torch.float32, 0, 16)], slot=0
        )
        connector._call(connector._queue_ipc_export([(entries, [], [])], 32))
        transfer.enqueue_snapshot_export_v1.assert_awaited_once()
    finally:
        connector._gpu_export_arena = None
        connector._gpu_export_tickets.clear()
        connector._ipc_export_jobs.clear()


def test_ipc_backpressure_waits_before_gathering(
    connector, connector_module, monkeypatch
):
    from torch.multiprocessing import reductions

    transfer = SimpleNamespace(ready_blocks_for_export_v1=AsyncMock(return_value=[]))
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    connector._registered_kv_caches = {"K": torch.ones(8, 2), "V": torch.ones(8, 2)}
    connector._layer_group_ids = {"K": 0, "V": 1}
    transfer.enqueue_snapshot_export_v1 = AsyncMock()
    transfer.poll_snapshot_exports_v1 = AsyncMock(return_value={"old": ([111], [])})
    connector._ipc_export_jobs["old"] = ({111}, 16, [torch.ones(16)])
    connector._ipc_export_poll_at = float("inf")
    monkeypatch.setattr(connector_module, "_MAX_PENDING_EXPORT_BYTES", 16)
    connector._ipc_export_budget = 16
    monkeypatch.setattr(connector, "_can_queue_ipc_export", lambda _: True)
    monkeypatch.setattr(reductions, "reduce_tensor", lambda value: (None, (value,)))
    original_prepare = connector._prepare_cpu_export

    def prepare(*args, **kwargs):
        assert not connector._ipc_export_jobs
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(connector, "_prepare_cpu_export", prepare)
    request = connector_module.XavierStoreRequest("new", [4], [222], [[4], [5]])
    connector._get_connector_metadata = (
        lambda: connector_module.XavierConnectorMetadata(store_requests=[request])
    )
    connector.wait_for_save()
    transfer.poll_snapshot_exports_v1.assert_awaited_once_with(["old"], wait=True)
    ticket = transfer.enqueue_snapshot_export_v1.await_args.args[0]
    assert connector._ipc_export_jobs[ticket][0] == {222}
    transfer.poll_snapshot_exports_v1.return_value = {ticket: ([222], [])}
    connector._poll_ipc_exports(wait=True)


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(),
    reason="Linux CUDA IPC required",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.asyncio
async def test_actor_publishes_gpu_snapshot_without_producer_polling(dtype):
    from torch.multiprocessing.reductions import reduce_tensor

    pool = await xo.create_actor_pool(
        "127.0.0.1", n_process=1, subprocess_start_method="spawn"
    )
    async with pool:
        tracker = await xo.create_actor(
            VLLMBlockTracker, address=pool.external_address, uid="snapshot-tracker"
        )
        actor = await xo.create_actor(
            SnapshotExportActor,
            address=pool.external_address,
            uid="snapshot-export",
            allocate_strategy=ProcessIndex(1),
        )
        source = (
            torch.arange(16, dtype=torch.float32, device="cuda").to(dtype).view(8, 2)
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gathered = source.index_select(0, torch.tensor([2, 4], device="cuda"))
            packed = gathered.view(torch.uint8).reshape(2, -1)
            descriptor = reduce_tensor(packed)[1]
        expected = (
            torch.arange(16)
            .to(dtype)
            .view(8, 2)[[2, 4]]
            .view(torch.uint8)
            .reshape(-1)
            .tolist()
        )
        await actor.enqueue_snapshot_export_v1(
            "export",
            [([111, 222], descriptor, [("K", (2,), dtype, 0, 4)])],
            tracker.address,
            tracker.uid,
        )
        with torch.cuda.stream(stream):
            source.zero_()
        # Do not drive any producer connector loop. The actor must publish and
        # answer discovery while the producer is idle.
        for _ in range(100):
            remote = await tracker.query_blocks(0, [(111, 0), (222, 1)])
            if remote:
                break
            await asyncio.sleep(0.01)
        assert remote == {1: {(111, 111, 0), (222, 222, 1)}}
        assert await actor.snapshot_bits("K", [111, 222]) == expected
        result = await actor.poll_snapshot_exports_v1(["export"], wait=True)
        assert set(result["export"][0]) == {111, 222}
        assert result["export"][1] == []
