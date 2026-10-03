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
    r._small_slab_streak = {}
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
    assert source.metrics["wire_bytes"] == 16


@pytest.mark.asyncio
async def test_load_switches_after_repeated_small_batches_and_reuses_views(monkeypatch):
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
    await dest.run(dest.load, {0: {"K": {2: 1}}})
    await dest.run(dest.load, {0: {"K": {1: 2, 2: 3}}})
    assert [b.numel() for b in buffers_seen] == [16, 4, 4, 16]
    assert buffers_seen[1] is buffers_seen[2] is source.send_buffers[4]
    assert buffers_seen[0] is buffers_seen[3] is source.send_buffer
    assert dest.caches["K"][:4].tolist() == [[1, -1], [2, -2], [1, -1], [2, -2]]
    assert source.metrics["wire_bytes"] == 40
    assert source.metrics["useful_bytes"] == 20


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


def test_gpu_connector_failure_releases_lease(connector, connector_module):
    request = connector_module.XavierLoadRequest("r", {2: {1: 0}}, lease="1:lease")
    connector._gpu_budget = 0
    connector._get_connector_metadata = (
        lambda: connector_module.XavierConnectorMetadata(load_requests=[request])
    )
    connector._load_gpu_requests = AsyncMock(side_effect=RuntimeError("peer failed"))
    connector._release_load_request = AsyncMock()
    with pytest.raises(RuntimeError, match="peer failed"):
        connector.start_load_kv(SimpleNamespace())
    connector._release_load_request.assert_awaited_once_with(request)


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
@pytest.mark.parametrize("gpu_slots", [1, 2])
async def test_staging_reused_layer_skips_gather_and_preserves_lru(
    monkeypatch, gpu_slots
):
    r = runtime(monkeypatch, gpu_slots=gpu_slots)
    stage(r, 1)
    stage(r, 2)
    before = r.store.blocks[1]["K"]
    syncs = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: syncs.append(args))

    gathers = []

    def unexpected(*args, **kwargs):
        gathers.append(args)
        raise AssertionError("immutable cached layer must not be gathered")

    monkeypatch.setattr(torch.Tensor, "index_select", unexpected)
    await r.stage([{"K": ([1], [7])}])
    assert list(r.store.blocks) == [2, 1]
    assert r.store.blocks[1]["K"] is before
    assert r.store.tiers[1] == ("cpu" if gpu_slots == 1 else "gpu")
    assert not gathers
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
    syncs = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: syncs.append(args))
    await r.stage([{"K": ([1], [0]), "V": ([1], [0])}])
    assert r.store.read("K", [1]).tolist() == [[1, -1]]
    assert r.store.read("V", [1]).tolist() == [[1, 1]]
    assert len(syncs) == 1


@pytest.mark.asyncio
async def test_unpublished_snapshot_still_synchronizes(monkeypatch):
    r = runtime(monkeypatch)
    r.store.stage("K", [1], torch.ones(1, 2, dtype=torch.bfloat16))
    syncs = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: syncs.append(args))
    await r.stage([{"K": ([1], [0])}])
    assert len(syncs) == 1


@pytest.mark.asyncio
async def test_skipped_published_layer_then_failure_fences_and_cleans(monkeypatch):
    r = runtime(monkeypatch, gpu_slots=3)
    r.store.block_bytes = 8
    stage(r, 1)
    r.store.stage("K", [2], torch.ones(1, 2, dtype=torch.bfloat16))
    r.caches["V"] = torch.ones_like(r.caches["K"])
    gathers, syncs = [], []

    def fail(value, dim, ids):
        gathers.append(value)
        raise RuntimeError("V gather failed")

    monkeypatch.setattr(torch.Tensor, "index_select", fail)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: syncs.append(args))
    await r.stage([{"K": ([1], [0]), "V": ([1, 2], [0, 1])}])
    assert len(gathers) == 1 and gathers[0] is r.caches["V"]
    assert len(syncs) == 1
    assert list(r.store.blocks) == [1]
    assert r.store.ready == {1}
    assert r.store.read("K", [1]).tolist() == [[1, -1]]
    assert "V" not in r.store.blocks[1]


@pytest.mark.parametrize("block_bytes", [1024, 4096, 8192])
def test_constructor_creates_persistent_slab_views(monkeypatch, block_bytes):
    from .. import gpu_transfer

    # Keep real torch allocations/views on CPU while exercising the CUDA-only
    # constructor; only the cache device guard and NIXL reference are substituted.
    class Cache:
        is_cuda = True
        device = torch.device("cpu")

        def __getitem__(self, index):
            return torch.empty(block_bytes, dtype=torch.uint8)

        def element_size(self):
            return 1

    monkeypatch.setattr(gpu_transfer, "version", lambda name: "0.11.1")
    refs = []

    def buffer_ref(address, buffer):
        ref = SimpleNamespace(address=address, buffer=buffer)
        refs.append(ref)
        return ref

    monkeypatch.setattr(xo, "buffer_ref", buffer_ref)
    actor = SimpleNamespace(
        address="nixl://127.0.0.1:1234", _snapshot_store=SimpleNamespace(capacity=8)
    )
    transfer = GPUTransfer(actor, {"K": Cache()}, block_bytes * 2)
    expected_slab = block_bytes * 64
    assert transfer.slab_bytes == expected_slab
    expected_keys = {expected_slab, min(expected_slab, 262144)}
    assert set(transfer.send_buffers) == expected_keys
    assert set(transfer.recv_buffers) == expected_keys
    assert set(transfer.recv_refs) == expected_keys
    assert len(refs) == len(expected_keys)
    assert transfer.send_buffers[expected_slab] is transfer.send_buffer
    assert transfer.recv_buffers[expected_slab] is transfer.recv_buffer
    assert transfer.recv_refs[expected_slab] is transfer.recv_ref
    for size in expected_keys:
        for buffers, full in (
            (transfer.send_buffers, transfer.send_buffer),
            (transfer.recv_buffers, transfer.recv_buffer),
        ):
            view = buffers[size]
            assert view.numel() == size
            assert view.data_ptr() == full.data_ptr()
            assert (
                view.untyped_storage().data_ptr() == full.untyped_storage().data_ptr()
            )
            assert view.storage_offset() == 0
        ref = transfer.recv_refs[size]
        assert ref.address == actor.address
        assert ref.buffer is transfer.recv_buffers[size]


def test_slab_selection_avoids_alternating_churn_per_peer(monkeypatch):
    r = runtime(monkeypatch)
    r.recv_refs[4] = r.recv_buffer[:4]
    assert [r._select_slab_bytes(0, size) for size in [8, 4] * 5] == [16] * 10
    assert r._select_slab_bytes(1, 4) == 16
    assert r._select_slab_bytes(0, 4) == 4
    assert r._select_slab_bytes(0, 8) == 16
    assert r._select_slab_bytes(1, 4) == 4
    assert r._select_slab_bytes(0, 4) == 16


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
async def test_connector_batch_cancellation_holds_all_leases(
    connector, connector_module
):
    requests = [
        connector_module.XavierLoadRequest(str(i), {}, lease=str(i)) for i in range(2)
    ]
    entered, release = asyncio.Event(), asyncio.Event()

    async def load(entries):
        assert entries == requests
        entered.set()
        await release.wait()

    connector._load_gpu_requests = AsyncMock(side_effect=load)
    connector._release_load_request = AsyncMock()
    task = asyncio.create_task(connector._load_gpu_batch(requests))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    connector._release_load_request.assert_not_awaited()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert connector._release_load_request.await_count == 2


@pytest.mark.asyncio
async def test_connector_batch_attempts_all_lease_releases(connector, connector_module):
    requests = [
        connector_module.XavierLoadRequest(str(i), {}, lease=str(i)) for i in range(2)
    ]
    connector._load_gpu_requests = AsyncMock()
    connector._release_load_request = AsyncMock(
        side_effect=[RuntimeError("release failed"), None]
    )
    with pytest.raises(RuntimeError, match="release failed"):
        await connector._load_gpu_batch(requests)
    assert connector._release_load_request.await_count == 2


@pytest.mark.asyncio
async def test_connector_batch_builds_one_actor_call(
    connector, connector_module, monkeypatch
):
    requests = [
        connector_module.XavierLoadRequest(
            str(i), {0: {123: i}}, local_transfers_by_group={0: {0: {123: i}}}
        )
        for i in range(2)
    ]
    connector._registered_kv_caches = {"layer": torch.zeros(8, 2)}
    transfer = SimpleNamespace(load_gpu_requests_v1=AsyncMock())
    connector._ensure_gpu_cache_mapping = AsyncMock(return_value=transfer)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    await connector._load_gpu_requests(requests)
    transfer.load_gpu_requests_v1.assert_awaited_once_with(
        [{0: {"layer": {123: 0}}}, {0: {"layer": {123: 1}}}]
    )


@pytest.mark.asyncio
async def test_batch_small_requests_keep_small_slab_bound(monkeypatch):
    source, dest = runtime(monkeypatch, gpu_slots=2), runtime(monkeypatch)
    stage(source, 2)
    source.send_buffers[8] = source.send_buffer[:8]
    dest.recv_buffers[8] = dest.recv_buffer[:8]
    dest.recv_refs[8] = dest.recv_buffers[8]

    sizes = []

    async def copy(buffers, refs):
        sizes.append(buffers[0].numel())
        assert buffers[0].numel() == refs[0].numel()
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
    assert sizes == [16, 8, 8]
    assert source.metrics["wire_bytes"] == 32
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
