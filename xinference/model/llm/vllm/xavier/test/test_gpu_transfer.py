# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
import xoscar as xo

from ..gpu_transfer import GPUTransfer
from ..request_transfer import LayerRead, pack_reads
from ..tiered_snapshot import TieredKVSnapshotStore


def runtime(monkeypatch, gpu_slots=1):
    # Run transport orchestration on CPU; real CUDA IPC/NIXL is covered by the
    # two-GPU integration run. Only hardware operations are substituted here.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    r = GPUTransfer.__new__(GPUTransfer)
    r.device = torch.device("cpu")
    r.caches = {"K": torch.zeros(8, 2, dtype=torch.bfloat16)}
    r.store = TieredKVSnapshotStore(2, gpu_slots * 4, 4, r.device)
    r.slab_bytes = 16
    r.send_buffer = torch.zeros(16, dtype=torch.uint8)
    r.recv_buffer = torch.zeros_like(r.send_buffer)
    r.recv_ref = r.recv_buffer
    r.send_buffers = {r.slab_bytes: r.send_buffer}
    r.recv_buffers = {r.slab_bytes: r.recv_buffer}
    r.recv_refs = {r.slab_bytes: r.recv_ref}
    r.send_lock, r.recv_lock = asyncio.Lock(), asyncio.Lock()
    r.tasks, r.closing = set(), False
    r.metrics = dict(
        gpu_batches=0,
        cpu_batches=0,
        wire_bytes=0,
        useful_bytes=0,
        load_calls=0,
        load_requests=0,
    )
    r.actor = SimpleNamespace(_world_addresses=["source"])
    return r


def stage(r, key):
    r.store.stage("K", [key], torch.tensor([[key, -key]], dtype=torch.bfloat16))
    r.store.publish([key], {"K"})


@pytest.mark.asyncio
@pytest.mark.parametrize("small_slab", [False, True])
async def test_mixed_tier_load_preserves_bits_and_destinations(monkeypatch, small_slab):
    source, dest = runtime(monkeypatch), runtime(monkeypatch)
    if small_slab:
        source.send_buffers[4] = source.send_buffer[:4]
        dest.recv_buffers[4] = dest.recv_buffer[:4]
        dest.recv_refs[4] = dest.recv_buffers[4]
    stage(source, 1)
    stage(source, 2)
    assert source.store.tiers == {1: "cpu", 2: "gpu"}

    async def copy(buffers, refs):
        refs[0].copy_(buffers[0])

    monkeypatch.setattr(xo, "copy_to", copy)
    peer = SimpleNamespace(
        gpu_snapshot_locations_v1=AsyncMock(side_effect=source.locations),
        send_gpu_request_v1=AsyncMock(side_effect=source.send),
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=peer))
    dest.actor.read_request_blocks_v1 = AsyncMock(
        side_effect=lambda rank, reads: pack_reads(source.store, reads)
    )
    await dest.run(dest.load, {0: {"K": {1: 5, 2: 3}}})
    assert dest.caches["K"][[5, 3]].tolist() == [[1, -1], [2, -2]]
    assert dest.caches["K"][:3].count_nonzero() == 0
    assert dest.metrics["gpu_batches"] == dest.metrics["cpu_batches"] == 1
    assert source.metrics["useful_bytes"] == 4
    assert source.metrics["wire_bytes"] == (4 if small_slab else 16)


@pytest.mark.asyncio
async def test_load_selects_smallest_slab_and_reuses_buffer_objects(monkeypatch):
    source, dest = runtime(monkeypatch, gpu_slots=2), runtime(monkeypatch)
    stage(source, 1)
    stage(source, 2)
    source.send_buffers[4] = source.send_buffer[:4]
    dest.recv_buffers[4] = dest.recv_buffer[:4]
    dest.recv_refs[4] = dest.recv_buffers[4]
    buffers_seen = []

    async def copy(buffers, refs):
        buffers_seen.append(buffers[0])
        assert buffers[0].numel() == refs[0].numel()
        refs[0].copy_(buffers[0])

    monkeypatch.setattr(xo, "copy_to", copy)
    peer = SimpleNamespace(
        gpu_snapshot_locations_v1=AsyncMock(side_effect=source.locations),
        send_gpu_request_v1=AsyncMock(side_effect=source.send),
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=peer))
    await dest.run(dest.load, {0: {"K": {1: 0}}})
    await dest.run(dest.load, {0: {"K": {2: 1}}})
    await dest.run(dest.load, {0: {"K": {1: 2, 2: 3}}})
    assert [b.numel() for b in buffers_seen] == [4, 4, 16]
    assert buffers_seen[0] is buffers_seen[1] is source.send_buffers[4]
    assert buffers_seen[2] is source.send_buffer
    assert dest.caches["K"][:4].tolist() == [[1, -1], [2, -2], [1, -1], [2, -2]]
    assert source.metrics["wire_bytes"] == 24
    assert source.metrics["useful_bytes"] == 16


@pytest.mark.asyncio
async def test_send_rejects_payload_exceeding_selected_slab(monkeypatch):
    r = runtime(monkeypatch, gpu_slots=2)
    stage(r, 1)
    stage(r, 2)
    r.send_buffers[4] = r.send_buffer[:4]
    copy = AsyncMock()
    monkeypatch.setattr(xo, "copy_to", copy)
    reads = [LayerRead("K", [1, 2], [0, 1], (2,), torch.bfloat16)]
    with pytest.raises(ValueError, match="exceeds transfer slab"):
        await r.send(reads, r.recv_ref, 4)
    with pytest.raises(ValueError, match="slab sizes differ"):
        await r.send(reads, r.recv_ref, 8)
    copy.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_waits_for_transfer_before_buffer_reuse(monkeypatch):
    r = runtime(monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()

    async def operation():
        async with r.recv_lock:
            entered.set()
            await release.wait()

    task = asyncio.create_task(r.run(operation))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and r.recv_lock.locked()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and r.recv_lock.locked()
    if hasattr(task, "cancelling"):
        assert task.cancelling() == 2
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not r.recv_lock.locked()
    assert not r.tasks


@pytest.mark.asyncio
async def test_close_drains_operations_and_rejects_new_work(monkeypatch):
    r = runtime(monkeypatch)
    release = asyncio.Event()
    task = asyncio.create_task(r.run(release.wait))
    await asyncio.sleep(0)
    close = asyncio.create_task(r.close())
    await asyncio.sleep(0)
    assert not close.done()
    with pytest.raises(RuntimeError, match="shutting down"):
        await r.run(release.wait)
    release.set()
    await asyncio.gather(task, close)
    assert not r.caches and r.recv_ref is None
    assert not r.recv_refs and not r.recv_buffers and not r.send_buffers


@pytest.mark.asyncio
async def test_invalid_layout_fails_before_nixl_copy(monkeypatch):
    r = runtime(monkeypatch)
    stage(r, 1)
    copy = AsyncMock()
    monkeypatch.setattr(xo, "copy_to", copy)
    reads = [LayerRead("K", [1], [0], (2,), torch.float16)]
    with pytest.raises(ValueError, match="layouts"):
        await r.send(reads, r.recv_ref, r.slab_bytes)
    copy.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_failure_releases_lock_for_retry(monkeypatch):
    r = runtime(monkeypatch)
    stage(r, 1)
    monkeypatch.setattr(
        xo, "copy_to", AsyncMock(side_effect=RuntimeError("transfer failed"))
    )
    with pytest.raises(RuntimeError, match="transfer failed"):
        await r.run(
            r.send,
            [LayerRead("K", [1], [0], (2,), torch.bfloat16)],
            r.recv_ref,
            r.slab_bytes,
        )
    assert not r.send_lock.locked() and not r.tasks


@pytest.mark.parametrize("value", [-1, 1.5, True, "1024"])
def test_invalid_gpu_budget(value):
    from ..transport import validate_gpu_cache_budget

    with pytest.raises(ValueError, match="integer"):
        validate_gpu_cache_budget(value, True, 2)


def test_gpu_budget_is_opt_in_and_requires_xavier():
    from ..transport import validate_gpu_cache_budget

    assert validate_gpu_cache_budget(None, False, 1) is None
    assert validate_gpu_cache_budget(0, True, 2) == 0
    assert validate_gpu_cache_budget(1024, True, 2) == 1024
    with pytest.raises(ValueError, match="multiple replicas"):
        validate_gpu_cache_budget(1, False, 2)
    with pytest.raises(ValueError, match="multiple replicas"):
        validate_gpu_cache_budget(1, True, 1)


def test_nixl_pool_options_keep_explicit_ucx_transport(monkeypatch):
    import importlib.metadata
    import importlib.util

    from ..transport import gpu_pool_options

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.11.1")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    env = {"UCX_TLS": "rc,cuda_copy,cuda_ipc"}
    assert gpu_pool_options("10.0.0.1:1234", env) == {
        "external_address": "nixl://10.0.0.1:0"
    }
    assert env == {"UCX_TLS": "rc,cuda_copy,cuda_ipc", "UCX_MEMTYPE_CACHE": "n"}
    monkeypatch.setenv("UCX_TLS", "rc,cuda_copy")
    inherited = {}
    gpu_pool_options("10.0.0.1:1234", inherited)
    assert inherited["UCX_TLS"] == "rc,cuda_copy"


@pytest.mark.asyncio
async def test_simultaneous_hybrid_loads_do_not_deadlock(monkeypatch):
    first, second = runtime(monkeypatch), runtime(monkeypatch)
    stage(first, 1)
    stage(second, 2)
    first.actor._world_addresses = ["first", "second"]
    second.actor._world_addresses = ["first", "second"]

    async def copy(buffers, refs):
        await asyncio.sleep(0)
        refs[0].copy_(buffers[0])

    monkeypatch.setattr(xo, "copy_to", copy)
    peers = {}
    for address, source in [("first", first), ("second", second)]:
        peers[address] = SimpleNamespace(
            gpu_snapshot_locations_v1=AsyncMock(side_effect=source.locations),
            send_gpu_request_v1=AsyncMock(side_effect=source.send),
        )
    monkeypatch.setattr(
        xo, "actor_ref", AsyncMock(side_effect=lambda address, uid: peers[address])
    )
    await asyncio.wait_for(
        asyncio.gather(
            first.run(first.load, {1: {"K": {2: 0}}}),
            second.run(second.load, {0: {"K": {1: 0}}}),
        ),
        timeout=2,
    )
    assert first.caches["K"][0].tolist() == [2, -2]
    assert second.caches["K"][0].tolist() == [1, -1]


def test_shutdown_releases_loop_after_gpu_close_failure(
    connector, connector_module, monkeypatch, caplog
):
    connector._gpu_cache_mapped = True
    connector._transfer_ref = SimpleNamespace(
        close_gpu_caches_v1=AsyncMock(side_effect=RuntimeError("peer lost"))
    )

    async def start():
        return None

    connector._call(start())
    loop = connector._loop
    connector.shutdown()
    assert "Failed to close Xavier GPU caches during shutdown" in caplog.text
    assert connector._loop is None
    assert loop.is_closed()


@pytest.mark.asyncio
async def test_cancelled_close_finishes_ipc_cleanup(monkeypatch):
    r = runtime(monkeypatch)
    release = asyncio.Event()
    active = asyncio.create_task(r.run(release.wait))
    await asyncio.sleep(0)
    closing = asyncio.create_task(r.close())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await active
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert not r.caches and r.recv_ref is None
    assert not r.recv_refs and not r.recv_buffers and not r.send_buffers
    await r.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
async def test_gpu_packing_preserves_all_bf16_bits(monkeypatch, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required for GPU packing")
    r = runtime(monkeypatch)
    r.device = torch.device(device)
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(8, 8192)
    r.store = TieredKVSnapshotStore(1, 131072, 16384, r.device)
    r.store.stage("K", list(range(8)), bits.view(torch.bfloat16).to(device))
    r.store.publish(list(range(8)), {"K"})
    r.slab_bytes = 131072
    r.send_buffer = torch.empty(r.slab_bytes, dtype=torch.uint8, device=device)
    r.send_buffers = {r.slab_bytes: r.send_buffer}
    received = torch.empty_like(r.send_buffer)

    async def copy(buffers, refs):
        refs[0].copy_(buffers[0])

    monkeypatch.setattr(xo, "copy_to", copy)
    await r.send(
        [LayerRead("K", list(range(8)), list(range(8)), (8192,), torch.bfloat16)],
        received,
        r.slab_bytes,
    )
    assert torch.equal(received.view(torch.int16).reshape_as(bits).cpu(), bits)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [torch.cuda.OutOfMemoryError, RuntimeError])
async def test_staging_failure_drains_and_drops_only_unpublished(monkeypatch, failure):
    r = runtime(monkeypatch, gpu_slots=3)
    stage(r, 1)
    assert r.store.reserve("2:live", [1])
    original = r.store.stage
    calls = []

    def copy(layer, keys, tensors):
        original(layer, keys, tensors)
        raise failure("copy failed after partial staging")

    monkeypatch.setattr(r.store, "stage", copy)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    await r.run(r.stage, [{"K": ([1, 2], [0, 1])}])
    assert calls == ["sync"]
    assert list(r.store.blocks) == [1]
    assert r.store.publish([2], {"K"}) == []
    assert r.store.leases == {"2:live": {1}}
    assert r.store.read("K", [1]).tolist() == [[1, -1]]


@pytest.mark.asyncio
async def test_failed_gather_still_synchronizes(monkeypatch):
    r = runtime(monkeypatch)
    calls = []

    def fail(*args):
        raise torch.cuda.OutOfMemoryError("gather failed")

    monkeypatch.setattr(torch.Tensor, "index_select", fail)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    await r.run(r.stage, [{"K": ([1], [0])}])
    assert calls == ["sync"]
    assert not r.store.blocks


@pytest.mark.asyncio
async def test_sticky_cuda_error_cleans_state_and_still_propagates(monkeypatch):
    r = runtime(monkeypatch)

    def fail(device):
        raise RuntimeError("sticky CUDA error")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError, match="sticky"):
        await r.run(r.stage, [{"K": ([1], [0])}])
    assert not r.store.blocks
    for _ in range(2):
        with pytest.raises(RuntimeError, match="sticky"):
            await r.close()
        assert not r.caches and r.recv_ref is None


@pytest.mark.asyncio
async def test_staging_fence_yields_and_cancellation_retains_ownership(monkeypatch):
    import threading

    r = runtime(monkeypatch)
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def sync(device):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)

    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    task = asyncio.create_task(r.run(r.stage, [{"K": ([1], [0])}]))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        # The actor loop remains usable while the CUDA fence waits in a thread.
        assert not task.done()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and r.tasks
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert r.store.publish([1], {"K"}) == [1]


def test_constructor_and_mapping_use_registered_caches(monkeypatch):
    import torch.multiprocessing.reductions as reductions

    from .. import gpu_transfer
    from ..gpu_transfer import GPUTransferMixin
    from ..snapshot import KVSnapshotStore

    monkeypatch.setattr(gpu_transfer, "version", lambda name: "0.11.1")
    # Only substitute the CUDA boundary; use real tensor sizes and store policy.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(xo, "buffer_ref", lambda address, buffer: buffer)
    cache = torch.ones(8, 2)
    monkeypatch.setattr(reductions, "rebuild_cuda_tensor", lambda *desc: cache)
    actor = SimpleNamespace(
        address="nixl://127.0.0.1:1234", _snapshot_store=KVSnapshotStore(3)
    )
    old_store = actor._snapshot_store
    GPUTransferMixin.map_gpu_caches_v1(actor, {"K": ("descriptor",)}, 16)
    r = actor._gpu_transfer
    assert actor._snapshot_store is r.store and r.store is not old_store
    assert r.store.cpu_capacity == 3 and r.store.gpu_capacity == 2
    assert r.store.block_bytes == 8 and r.slab_bytes == 8 * 64
    assert r.caches["K"] is cache and r.recv_ref is r.recv_buffer
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    asyncio.run(GPUTransferMixin.stage_gpu_requests_v1(actor, [{"K": ([42], [3])}]))
    assert r.store.publish([42], {"K"}) == [42]
    assert r.store.read("K", [42]).tolist() == [[1, 1]]
    with pytest.raises(ValueError, match="one CUDA device"):
        GPUTransfer(
            actor,
            {
                "K": SimpleNamespace(is_cuda=True, device="cuda:0"),
                "V": SimpleNamespace(is_cuda=True, device="cuda:1"),
            },
            16,
        )
    with pytest.raises(RuntimeError, match="already registered"):
        GPUTransferMixin.map_gpu_caches_v1(actor, {"K": ()}, 16)
    monkeypatch.setattr(gpu_transfer, "version", lambda name: "0.11.0")
    with pytest.raises(RuntimeError, match="0.11.1"):
        GPUTransfer(actor, {"K": cache}, 16)
    monkeypatch.setattr(gpu_transfer, "version", lambda name: "0.11.1")
    actor.address = "127.0.0.1:1234"
    with pytest.raises(RuntimeError, match="NIXL actor pool"):
        GPUTransfer(actor, {"K": cache}, 16)
    actor.address = "nixl://127.0.0.1:1234"
    with pytest.raises(ValueError, match="registered CUDA"):
        GPUTransfer(actor, {}, 16)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: False))
    with pytest.raises(ValueError, match="registered CUDA"):
        GPUTransfer(actor, {"K": cache}, 16)


@pytest.mark.parametrize(
    "address, expected",
    [
        ("0.0.0.0:1234", "nixl://10.0.0.9:0"),
        ("[::]:1234", "nixl://10.0.0.9:0"),
        ("[2001:db8::1]:1234", "nixl://[2001:db8::1]:0"),
    ],
)
def test_xavier_advertises_reachable_nixl_address(monkeypatch, address, expected):
    import importlib.metadata
    import importlib.util
    from unittest.mock import MagicMock

    from ... import pd
    from ..transport import gpu_pool_options

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.11.1")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    sock = MagicMock()
    sock.__enter__.return_value = sock
    sock.getsockname.return_value = ("10.0.0.9", 0)
    monkeypatch.setattr(pd.socket, "socket", lambda *args: sock)
    assert gpu_pool_options(address, {}) == {"external_address": expected}
    sock.getsockname.return_value = ("127.0.0.1", 0)
    with pytest.raises(ValueError, match="remotely reachable"):
        gpu_pool_options("0.0.0.0:1234", {})


@pytest.mark.asyncio
async def test_staging_reused_layer_skips_gather_and_preserves_lru(monkeypatch):
    r = runtime(monkeypatch, gpu_slots=2)
    stage(r, 1)
    stage(r, 2)
    before = r.store.blocks[1]["K"]
    syncs = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: syncs.append(args))

    def unexpected(*args, **kwargs):
        raise AssertionError("immutable cached layer must not be gathered")

    monkeypatch.setattr(torch.Tensor, "index_select", unexpected)
    await r.stage([{"K": ([1], [7])}])
    assert list(r.store.blocks) == [2, 1]
    assert r.store.blocks[1]["K"] is before
    assert not syncs


@pytest.mark.asyncio
async def test_staging_partial_hit_still_copies_and_synchronizes(monkeypatch):
    r = runtime(monkeypatch, gpu_slots=2)
    stage(r, 1)
    r.caches["K"][3] = torch.tensor([3, -3])
    syncs = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: syncs.append(args))
    await r.stage([{"K": ([1, 3], [0, 3])}])
    assert r.store.read("K", [1, 3]).tolist() == [[1, -1], [3, -3]]
    assert len(syncs) == 1
    r.caches["K"].zero_()
    assert r.store.read("K", [3]).tolist() == [[3, -3]]


@pytest.mark.asyncio
async def test_staging_missing_layer_is_not_a_reuse_hit(monkeypatch):
    r = runtime(monkeypatch, gpu_slots=2)
    r.store.block_bytes = 8
    stage(r, 1)
    r.caches["V"] = torch.ones_like(r.caches["K"])
    await r.stage([{"K": ([1], [0]), "V": ([1], [0])}])
    assert r.store.read("K", [1]).tolist() == [[1, -1]]
    assert r.store.read("V", [1]).tolist() == [[1, 1]]


@pytest.mark.asyncio
async def test_unpublished_snapshot_still_synchronizes(monkeypatch):
    r = runtime(monkeypatch)
    r.store.stage("K", [1], torch.ones(1, 2, dtype=torch.bfloat16))
    syncs = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: syncs.append(args))
    await r.stage([{"K": ([1], [0])}])
    assert len(syncs) == 1


@pytest.mark.asyncio
async def test_batch_load_preserves_shared_keys_and_bounds_transfers(monkeypatch):
    source, dest = runtime(monkeypatch), runtime(monkeypatch)
    stage(source, 1)
    stage(source, 2)
    sent = []

    async def copy(buffers, refs):
        refs[0].copy_(buffers[0])

    async def send(reads, ref, size):
        assert sum(r.nbytes for r in reads) <= size
        sent.append(reads)
        await source.send(reads, ref, size)

    monkeypatch.setattr(xo, "copy_to", copy)
    peer = SimpleNamespace(
        gpu_snapshot_locations_v1=AsyncMock(side_effect=source.locations),
        send_gpu_request_v1=AsyncMock(side_effect=send),
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=peer))
    dest.actor.read_request_blocks_v1 = AsyncMock(
        side_effect=lambda rank, reads: pack_reads(source.store, reads)
    )
    requests = [{0: {"K": {2: i}}} for i in range(6)]
    requests += [{0: {"K": {1: 6}}}, {0: {"K": {1: 7, 2: 0}}}]
    await dest.run(dest.load_requests, requests)
    assert dest.caches["K"].tolist() == [[2, -2]] * 6 + [[1, -1]] * 2
    peer.gpu_snapshot_locations_v1.assert_awaited_once()
    assert len(sent) == 2
    assert [len(r.keys) for batch in sent for r in batch] == [4, 2]
    assert dest.metrics["gpu_batches"] == 2
    assert dest.metrics["cpu_batches"] == 1


@pytest.mark.asyncio
async def test_batch_conflicting_destinations_fail_before_transfer(monkeypatch):
    r = runtime(monkeypatch)
    lookup = AsyncMock()
    monkeypatch.setattr(xo, "actor_ref", lookup)
    with pytest.raises(ValueError, match="Conflicting"):
        await r.run(r.load_requests, [{0: {"K": {1: 0}}}, {1: {"K": {2: 0}}}])
    lookup.assert_not_awaited()
    assert r.caches["K"].count_nonzero() == 0


@pytest.mark.asyncio
async def test_connector_batch_builds_one_actor_call(
    connector, connector_module, monkeypatch
):
    requests = [
        connector_module.XavierLoadRequest(
            str(i),
            {2: {1: i}},
            lease=str(i),
            local_transfers_by_group={0: {2: {1: i}}},
        )
        for i in range(2)
    ]
    connector._registered_kv_caches = {"layer": torch.zeros(8, 2)}
    transfer = SimpleNamespace(load_gpu_requests_v1=AsyncMock())
    connector._ensure_gpu_cache_mapping = AsyncMock(return_value=transfer)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    await connector._submit_gpu_requests(requests)
    task = next(iter(connector._gpu_load_jobs))
    transfer.load_gpu_requests_v1.assert_awaited_once_with(
        [{2: {"layer": {1: 0}}}, {2: {"layer": {1: 1}}}],
        [(r.lease, r.transfers) for r in requests],
    )
    assert connector._gpu_load_jobs.pop(task) == ["0", "1"]
    await task


@pytest.mark.asyncio
async def test_async_load_progress_cleanup_and_close(monkeypatch):
    r = runtime(monkeypatch)
    started, finish, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def load(requests):
        started.set()
        await finish.wait()

    async def cleanup(*args):
        await release.wait()

    r.load_requests = AsyncMock(side_effect=load)
    r.actor.release_remote_blocks_v1 = AsyncMock(side_effect=cleanup)
    job = asyncio.create_task(r.run(r.load_requests_with_leases, [{}], [("lease", {})]))
    await started.wait()  # Progress without an EngineCore RPC loop or poll.
    assert not job.done()
    close = asyncio.create_task(r.close())
    await asyncio.sleep(0)
    assert not close.done()
    assert r.caches
    finish.set()
    await asyncio.sleep(0)
    assert not job.done()  # Lease cleanup is part of completion.
    release.set()
    await close
    r.actor.release_remote_blocks_v1.assert_awaited_once_with("lease", {})
    assert not r.caches
    await job


@pytest.mark.asyncio
async def test_async_load_failure_is_not_ready_and_releases_all_leases(monkeypatch):
    r = runtime(monkeypatch)
    r.load_requests = AsyncMock(side_effect=RuntimeError("write failed"))
    r.actor.release_remote_blocks_v1 = AsyncMock()
    with pytest.raises(RuntimeError, match="write failed"):
        await r.run(r.load_requests_with_leases, [{}], [("a", {}), ("b", {})])
    assert r.actor.release_remote_blocks_v1.await_count == 2
    await r.close()


def test_async_load_metadata_and_abort_preserve_ownership(connector, connector_module):
    connector._gpu_budget = 0
    load = connector_module.XavierLoadRequest("r", {2: {1: 0}}, lease="lease")
    connector._requests_need_load["r"] = load
    connector._leased_requests["r"] = load
    meta = connector_module.XavierConnectorMetadata()
    connector._build_load_meta(SimpleNamespace(), meta)
    assert not meta.load_requests  # Not allocated yet.
    req = SimpleNamespace(request_id="r")
    connector.update_state_after_alloc(
        req, SimpleNamespace(get_block_ids=lambda: ([3],)), 16
    )
    connector._release_load_request = AsyncMock()
    connector.request_finished(req, [3])
    connector._release_load_request.assert_not_awaited()
    connector._build_load_meta(SimpleNamespace(), meta)
    assert len(meta.load_requests) == 1
    assert meta.load_requests[0].local_transfers_by_group == {0: {2: {1: 3}}}
    assert not connector._leased_requests
    connector.request_finished(req, [3])
    connector._release_load_request.assert_not_awaited()
    connector._build_load_meta(
        SimpleNamespace(), connector_module.XavierConnectorMetadata()
    )


def test_async_completion_includes_aborted_requests(connector):
    ready = asyncio.Event()

    async def load():
        await ready.wait()

    async def submit():
        task = asyncio.create_task(load())
        connector._gpu_load_jobs[task] = ["live", "aborted"]

    connector._call(submit())
    for _ in range(3):
        assert connector.get_finished({"aborted"}) == (set(), set())
    ready.set()
    assert connector.get_finished(set()) == (set(), {"live", "aborted"})
    assert connector.get_finished(set()) == (set(), set())


@pytest.mark.asyncio
async def test_batch_small_requests_keep_small_slab_bound(monkeypatch):
    source, dest = runtime(monkeypatch, gpu_slots=2), runtime(monkeypatch)
    stage(source, 2)
    source.send_buffers[8] = source.send_buffer[:8]
    dest.recv_buffers[8] = dest.recv_buffer[:8]
    dest.recv_refs[8] = dest.recv_buffers[8]

    async def copy(buffers, refs):
        assert buffers[0].numel() == refs[0].numel() == 8
        refs[0].copy_(buffers[0])

    monkeypatch.setattr(xo, "copy_to", copy)
    peer = SimpleNamespace(
        gpu_snapshot_locations_v1=AsyncMock(side_effect=source.locations),
        send_gpu_request_v1=AsyncMock(side_effect=source.send),
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=peer))
    await dest.run(dest.load_requests, [{0: {"K": {2: i}}} for i in range(6)])
    assert dest.caches["K"][:6].tolist() == [[2, -2]] * 6
    assert peer.send_gpu_request_v1.await_count == 3
    assert source.metrics["wire_bytes"] == 24
    assert dest.metrics["load_calls"] == 1
    assert dest.metrics["load_requests"] == 6


@pytest.mark.asyncio
async def test_batch_groups_multiple_source_ranks_without_losing_destinations(
    monkeypatch,
):
    sources = [runtime(monkeypatch), runtime(monkeypatch)]
    dest = runtime(monkeypatch)
    dest.actor._world_addresses = ["first", "second"]
    for key, source in enumerate(sources, 1):
        stage(source, key)

    async def copy(buffers, refs):
        refs[0].copy_(buffers[0])

    monkeypatch.setattr(xo, "copy_to", copy)
    peers = {
        address: SimpleNamespace(
            gpu_snapshot_locations_v1=AsyncMock(side_effect=source.locations),
            send_gpu_request_v1=AsyncMock(side_effect=source.send),
        )
        for address, source in zip(dest.actor._world_addresses, sources)
    }
    monkeypatch.setattr(
        xo, "actor_ref", AsyncMock(side_effect=lambda address, uid: peers[address])
    )
    await dest.run(
        dest.load_requests,
        [
            {0: {"K": {1: 0}}},
            {1: {"K": {2: 1}}},
            {0: {"K": {1: 2}}},
            {1: {"K": {2: 3}}},
        ],
    )
    assert dest.caches["K"][:4].tolist() == [[1, -1], [2, -2], [1, -1], [2, -2]]
    for peer in peers.values():
        peer.gpu_snapshot_locations_v1.assert_awaited_once()
        peer.send_gpu_request_v1.assert_awaited_once()


@pytest.mark.parametrize("budget", [None, 0, 256])
def test_cache_hit_async_only_for_mapped_transfer_path(connector, budget):
    connector._gpu_budget = budget
    connector._query_remote_blocks = AsyncMock(return_value={2: {(1, 4, 0)}})
    connector._reserve_load_request = AsyncMock(return_value=True)
    connector._release_load_request = AsyncMock()
    request = SimpleNamespace(request_id="r", prompt_token_ids=list(range(17)))
    assert connector.get_num_new_matched_tokens(request, 0) == (16, budget is not None)
    # An unallocated hit is still owned by the scheduler and releases on abort.
    connector.request_finished(request, [])
    connector._release_load_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_load_attempts_every_release_after_cleanup_failure(monkeypatch):
    r = runtime(monkeypatch)
    r.load_requests = AsyncMock()
    r.actor.release_remote_blocks_v1 = AsyncMock(
        side_effect=[RuntimeError("release failed"), None]
    )
    with pytest.raises(RuntimeError, match="release failed"):
        await r.run(r.load_requests_with_leases, [{}], [("a", {}), ("b", {})])
    assert r.actor.release_remote_blocks_v1.await_count == 2
    await r.close()


def test_shutdown_drains_loads_before_closing_gpu_mapping(connector):
    async def load():
        await asyncio.sleep(0.01)

    async def submit():
        task = asyncio.create_task(load())
        connector._gpu_load_jobs[task] = ["r"]
        return task

    async def close():
        assert task.done()
        assert not connector._gpu_load_jobs

    task = connector._call(submit())
    connector._gpu_cache_mapped = True
    connector._transfer_ref = SimpleNamespace(
        close_gpu_caches_v1=AsyncMock(side_effect=close)
    )
    connector.shutdown()
    connector._transfer_ref.close_gpu_caches_v1.assert_awaited_once()
    assert connector._loop is None


@pytest.mark.asyncio
async def test_async_load_cancellation_holds_leases_until_writes_finish(monkeypatch):
    r = runtime(monkeypatch)
    started, finish = asyncio.Event(), asyncio.Event()

    async def load(requests):
        started.set()
        await finish.wait()

    r.load_requests = AsyncMock(side_effect=load)
    r.actor.release_remote_blocks_v1 = AsyncMock()
    job = asyncio.create_task(r.run(r.load_requests_with_leases, [{}], [("a", {})]))
    await started.wait()
    for _ in range(2):
        job.cancel()
        await asyncio.sleep(0)
    r.actor.release_remote_blocks_v1.assert_not_awaited()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await job
    r.actor.release_remote_blocks_v1.assert_awaited_once_with("a", {})
    await r.close()


@pytest.mark.asyncio
async def test_async_load_preserves_write_error_when_release_also_fails(
    monkeypatch, caplog
):
    r = runtime(monkeypatch)
    r.load_requests = AsyncMock(side_effect=RuntimeError("write failed"))
    r.actor.release_remote_blocks_v1 = AsyncMock(
        side_effect=[ValueError("release failed"), None]
    )
    with pytest.raises(RuntimeError, match="write failed"):
        await r.run(r.load_requests_with_leases, [{}], [("a", {}), ("b", {})])
    assert r.actor.release_remote_blocks_v1.await_count == 2
    assert "release failed" in caplog.text
    await r.close()
