# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
from types import SimpleNamespace

import pytest
import torch

from ....xavier.backends.torch import shard_gpu_transfer as module
from ....xavier.backends.torch.shard_gpu_transfer import (
    ShardGPUTransfer,
    ShardGPUTransferMixin,
)

# Two caches per rank: 2 x bfloat16 and 1 x float32, an 8-byte row.
LAYOUT = [
    ("a", (2,), torch.bfloat16, 0, 4),
    ("b", (1,), torch.float32, 4, 4),
]
ROW_BYTES = 8


def _caches(blocks=8):
    return {
        "a": torch.zeros(blocks, 2, dtype=torch.bfloat16),
        "b": torch.zeros(blocks, 1, dtype=torch.float32),
    }


def _row(key, shard):
    a = torch.tensor([key, -key], dtype=torch.bfloat16).view(torch.uint8)
    b = torch.tensor([key * 10 + shard], dtype=torch.float32).view(torch.uint8)
    return torch.cat([a, b])


class Network:
    """Replicas whose actor refs, buffer refs and copies stay in-process."""

    def __init__(self, monkeypatch):
        self.runtimes = {}
        self.copies = []
        self.peer_calls = []
        self.fail_send = None
        self.block_send = None
        network = self

        class Peer:
            def __init__(self, rank):
                self.rank = rank

            async def send_gpu_shards_v1(self, keys, refs, slab_class):
                network.peer_calls.append((self.rank, keys, slab_class))
                if network.block_send is not None:
                    await network.block_send.wait()
                if network.fail_send is not None and network.fail_send(keys):
                    raise ConnectionError("peer lost")
                return await network.runtimes[self.rank].send(keys, refs, slab_class)

            async def warm_gpu_shards_v1(self, refs, slab_class):
                return await network.runtimes[self.rank].warm(refs, slab_class)

        async def actor_ref(address, uid):
            return Peer(int(address))

        async def copy_to(buffers, refs):
            self.copies.append([b.numel() for b in buffers])
            for buffer, ref in zip(buffers, refs):
                assert buffer.numel() == ref.numel()
                ref.copy_(buffer)

        monkeypatch.setattr(module.xo, "actor_ref", actor_ref)
        monkeypatch.setattr(module.xo, "copy_to", copy_to)
        monkeypatch.setattr(module.xo, "buffer_ref", lambda address, buffer: buffer)
        monkeypatch.setattr(module, "PREWARM_ATTEMPTS", 0)

    def replica(self, rank, num_shards=2, budget=4 * ROW_BYTES, register=True):
        actor = SimpleNamespace(
            address=f"nixl://replica-{rank}",
            _rank=rank,
            _world_addresses=["0", "1", "2"],
            default_uid=lambda: "vllm-transfer-actor",
        )
        runtime = ShardGPUTransfer(actor, num_shards)
        self.runtimes[rank] = runtime
        if register:
            for shard in range(num_shards):
                runtime.register(shard, _caches(), LAYOUT, ROW_BYTES, budget)
        return runtime


async def _admit(runtime, shard, keys):
    await runtime.admit(shard, keys, [_row(key, shard) for key in keys])


def _mapping(destinations):
    return {"a": dict(destinations), "b": dict(destinations)}


def _assert_rows(runtime, shard, destinations):
    caches = runtime.shards[shard].caches
    for key, destination in destinations.items():
        row = torch.cat(
            [
                caches["a"][destination].view(torch.uint8),
                caches["b"][destination].view(torch.uint8),
            ]
        )
        assert torch.equal(row, _row(key, shard)), (shard, key)


@pytest.mark.asyncio
async def test_matching_ranks_load_cached_rows_and_report_misses(monkeypatch):
    network = Network(monkeypatch)
    source, consumer = network.replica(1), network.replica(2)
    await _admit(source, 0, [11, 22, 33])
    await _admit(source, 1, [11, 22])  # Rank 1 evicted or never cached 33.
    keys = [11, 22, 33]
    destinations = {11: 5, 22: 0, 33: 7}
    loaded = await asyncio.gather(
        *(
            consumer.load(shard, "lease", 1, keys, _mapping(destinations))
            for shard in range(2)
        )
    )
    assert loaded == [[11, 22, 33], [11, 22]]
    _assert_rows(consumer, 0, destinations)
    _assert_rows(consumer, 1, {11: 5, 22: 0})
    assert not consumer.shards[1].caches["b"][7].any()
    # One copy carried both shards' slabs.
    assert len(network.copies) == 1 and len(network.copies[0]) == 2
    assert consumer.metrics["loaded_rows"] == 5
    assert consumer.metrics["missed_rows"] == 1


@pytest.mark.asyncio
async def test_each_rank_scatters_only_its_own_cache_groups(monkeypatch):
    network = Network(monkeypatch)
    source, consumer = network.replica(1), network.replica(2)
    await _admit(source, 0, [11])
    await _admit(source, 1, [11])
    # Cache "b" belongs to a group that did not accept the block.
    mappings = {"a": {11: 3}, "b": {}}
    loaded = await asyncio.gather(
        *(consumer.load(shard, "lease", 1, [11], mappings) for shard in range(2))
    )
    assert loaded == [[11], [11]]
    for shard in range(2):
        caches = consumer.shards[shard].caches
        assert torch.equal(
            caches["a"][3], torch.tensor([11, -11], dtype=torch.bfloat16)
        )
        assert not caches["b"].any()


@pytest.mark.asyncio
async def test_registered_groups_share_one_mapping_per_load(monkeypatch):
    network = Network(monkeypatch)
    source = network.replica(1)
    consumer = network.replica(2, register=False)
    for shard in range(2):
        consumer.register(
            shard, _caches(), LAYOUT, ROW_BYTES, 4 * ROW_BYTES, {"a": 0, "b": 0}
        )
        await _admit(source, shard, [11, 22])
    destinations = {11: 6, 22: 1}
    loaded = await asyncio.gather(
        *(consumer.load(s, "lease", 1, [11, 22], {0: destinations}) for s in range(2))
    )
    assert loaded == [[11, 22], [11, 22]]
    for shard in range(2):
        _assert_rows(consumer, shard, destinations)


@pytest.mark.asyncio
async def test_large_loads_take_several_rounds(monkeypatch):
    monkeypatch.setattr(module, "MAX_SLAB_ROWS", 2)
    network = Network(monkeypatch)
    source = network.replica(1, budget=8 * ROW_BYTES)
    consumer = network.replica(2, budget=8 * ROW_BYTES)
    keys = [1, 2, 3, 4, 5]
    for shard in range(2):
        await _admit(source, shard, keys)
    destinations = {key: key + 1 for key in keys}
    loaded = await asyncio.gather(
        *(consumer.load(s, "lease", 1, keys, _mapping(destinations)) for s in range(2))
    )
    assert loaded == [keys, keys]
    assert [call[1] for call in network.peer_calls] == [
        [[1, 2], [1, 2]],
        [[3, 4], [3, 4]],
        [[5], [5]],
    ]
    for shard in range(2):
        _assert_rows(consumer, shard, destinations)


@pytest.mark.asyncio
async def test_repeated_small_loads_switch_to_a_persistent_small_view(monkeypatch):
    monkeypatch.setattr(module, "SMALL_SLAB_BYTES", 2 * ROW_BYTES)
    monkeypatch.setattr(module, "MAX_SLAB_ROWS", 8)
    network = Network(monkeypatch)
    source = network.replica(1, budget=8 * ROW_BYTES)
    consumer = network.replica(2, budget=8 * ROW_BYTES)
    assert consumer.class_rows == [[8, 8], [2, 2]]
    for shard in range(2):
        await _admit(source, shard, [1, 2])
    for lease in ("first", "second", "third"):
        await asyncio.gather(
            *(consumer.load(s, lease, 1, [1], _mapping({1: 0})) for s in range(2))
        )
    assert [call[2] for call in network.peer_calls] == [0, 1, 1]
    assert network.copies == [[64, 64], [16, 16], [16, 16]]
    # The small class reuses one view per shard, keeping the registration.
    assert consumer.recv_refs[1][0] is consumer.recv_refs[1][0]
    assert (
        consumer.shards[0].recv[1].data_ptr() == consumer.shards[0].recv[0].data_ptr()
    )


@pytest.mark.asyncio
async def test_gpu_copy_keeps_recent_rows_within_its_budget(monkeypatch):
    network = Network(monkeypatch)
    source = network.replica(1, budget=2 * ROW_BYTES)
    await _admit(source, 0, [1, 2])
    await source._gather(source.shards[0], [1], source.shards[0].send[0])
    await _admit(source, 0, [3])
    assert list(source.shards[0].slots) == [1, 3]
    assert source.metrics["evicted_rows"] == 1
    # A batch larger than the budget keeps only what fits.
    await _admit(source, 0, [4, 5, 6])
    assert len(source.shards[0].slots) == 2


@pytest.mark.asyncio
async def test_a_missing_rank_times_out_and_the_lease_reads_from_cpu(monkeypatch):
    monkeypatch.setattr(module, "RENDEZVOUS_SECONDS", 0.01)
    network = Network(monkeypatch)
    source, consumer = network.replica(1), network.replica(2)
    await _admit(source, 0, [1])
    assert await consumer.load(0, "lease", 1, [1], _mapping({1: 0})) == []
    # A rank arriving after the timeout does not start a partial transfer.
    assert await consumer.load(1, "lease", 1, [1], _mapping({1: 0})) == []
    assert network.peer_calls == []
    assert consumer.metrics["rendezvous_timeouts"] == 1
    assert not consumer.rendezvous


@pytest.mark.asyncio
async def test_unregistered_ranks_disable_the_gpu_path(monkeypatch):
    network = Network(monkeypatch)
    consumer = network.replica(2, register=False)
    consumer.register(0, _caches(), LAYOUT, ROW_BYTES, ROW_BYTES)
    assert not consumer.ready
    assert await consumer.load(0, "lease", 1, [1], _mapping({1: 0})) == []
    assert network.peer_calls == []


@pytest.mark.asyncio
async def test_a_failed_round_reports_only_completed_rows(monkeypatch):
    monkeypatch.setattr(module, "MAX_SLAB_ROWS", 1)
    network = Network(monkeypatch)
    source, consumer = network.replica(1), network.replica(2)
    for shard in range(2):
        await _admit(source, shard, [1, 2])
    network.fail_send = lambda keys: keys[0] == [2]
    loaded = await asyncio.gather(
        *(
            consumer.load(s, "lease", 1, [1, 2], _mapping({1: 0, 2: 1}))
            for s in range(2)
        )
    )
    assert loaded == [[1], [1]]
    assert consumer.metrics["failed_loads"] == 1
    assert "peer lost" in consumer.metrics["last_failure"]


@pytest.mark.asyncio
async def test_cancelled_rank_waits_until_its_caches_are_written(monkeypatch):
    network = Network(monkeypatch)
    source, consumer = network.replica(1), network.replica(2)
    for shard in range(2):
        await _admit(source, shard, [1])
    network.block_send = asyncio.Event()
    tasks = [
        asyncio.create_task(consumer.load(s, "lease", 1, [1], _mapping({1: 4})))
        for s in range(2)
    ]
    while not network.peer_calls:
        await asyncio.sleep(0)
    tasks[0].cancel()
    await asyncio.sleep(0.01)
    assert not tasks[0].done()  # The copy into its caches is still pending.
    network.block_send.set()
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]
    assert await tasks[1] == [1]
    _assert_rows(consumer, 0, {1: 4})


@pytest.mark.asyncio
async def test_rank_cancelled_before_the_others_arrive_abandons_the_lease(
    monkeypatch,
):
    network = Network(monkeypatch)
    consumer = network.replica(2)
    waiting = asyncio.create_task(consumer.load(0, "lease", 1, [1], _mapping({1: 0})))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert await consumer.load(1, "lease", 1, [1], _mapping({1: 0})) == []
    assert network.peer_calls == []


@pytest.mark.asyncio
async def test_send_without_cached_rows_copies_nothing(monkeypatch):
    network = Network(monkeypatch)
    network.replica(1)
    consumer = network.replica(2)
    loaded = await asyncio.gather(
        *(consumer.load(s, "lease", 1, [9], _mapping({9: 0})) for s in range(2))
    )
    assert loaded == [[], []]
    assert network.copies == []


@pytest.mark.asyncio
async def test_send_rejects_mismatched_requests(monkeypatch):
    network = Network(monkeypatch)
    source = network.replica(1)
    refs = network.replica(2).recv_refs[0]
    with pytest.raises(ValueError, match="shard count"):
        await source.send([[1]], refs[:1], 0)
    with pytest.raises(ValueError, match="slab classes"):
        await source.send([[1], [1]], refs, 5)
    too_many = list(range(source.class_rows[0][0] + 1))
    with pytest.raises(ValueError, match="exceeds"):
        await source.send([too_many, []], refs, 0)


@pytest.mark.asyncio
async def test_registration_is_idempotent_and_rejects_changes(monkeypatch):
    network = Network(monkeypatch)
    runtime = network.replica(1)
    shard = runtime.shards[0]
    assert runtime.register(0, _caches(), LAYOUT, ROW_BYTES, ROW_BYTES)
    assert runtime.shards[0] is shard
    with pytest.raises(ValueError, match="changed"):
        runtime.register(
            0, _caches(), LAYOUT[:1] + [("b", (1,), torch.int32, 4, 4)], 8, 8
        )
    with pytest.raises(ValueError, match="row size"):
        network.replica(3, register=False).register(0, _caches(), LAYOUT, 9, 8)


def test_requires_a_nixl_pool_and_tensor_parallelism():
    actor = SimpleNamespace(address="127.0.0.1:1234")
    with pytest.raises(RuntimeError, match="NIXL"):
        ShardGPUTransfer(actor, 2)
    with pytest.raises(ValueError, match="TP>1"):
        ShardGPUTransfer(SimpleNamespace(address="nixl://a"), 1)


@pytest.mark.asyncio
async def test_prewarm_copies_from_every_peer_replica(monkeypatch):
    network = Network(monkeypatch)
    monkeypatch.setattr(module, "PREWARM_ATTEMPTS", 2)
    monkeypatch.setattr(module, "PREWARM_INTERVAL_SECONDS", 0)
    source = network.replica(1)
    consumer = network.replica(2, register=False)
    for shard in range(2):
        consumer.register(shard, _caches(), LAYOUT, ROW_BYTES, ROW_BYTES)
    await asyncio.gather(*list(consumer.tasks), *list(source.tasks))
    # Rank 0 is the collective manager and is never a source.
    assert consumer.prewarmed == {1: True}
    assert source.prewarmed == {2: True}
    assert len(network.copies) == 2


@pytest.mark.asyncio
async def test_actor_endpoints_map_caches_and_close(monkeypatch):
    network = Network(monkeypatch)

    class Actor(ShardGPUTransferMixin):
        address = "nixl://replica"
        _rank = 1
        _world_addresses = ["0", "1"]

        @staticmethod
        def default_uid():
            return "vllm-transfer-actor"

    actor = Actor()
    caches = _caches()
    monkeypatch.setattr(
        "torch.multiprocessing.reductions.rebuild_cuda_tensor",
        lambda name: caches[name],
    )
    descriptors = {name: (name,) for name in caches}
    assert actor.get_gpu_shard_stats_v1() == {}
    assert not actor.register_gpu_shard_v1(0, 2, descriptors, LAYOUT, ROW_BYTES, 8)
    assert actor.register_gpu_shard_v1(1, 2, descriptors, LAYOUT, ROW_BYTES, 8)
    assert actor.get_gpu_shard_stats_v1()["ready"]
    with pytest.raises(ValueError, match="shard count"):
        actor.register_gpu_shard_v1(0, 4, descriptors, LAYOUT, ROW_BYTES, 8)
    network.runtimes[1] = actor._shard_gpu_transfer
    await actor.close_gpu_shards_v1()
    assert not actor.get_gpu_shard_stats_v1()["ready"]
    with pytest.raises(module.ShardGPUUnavailable):
        await actor.send_gpu_shards_v1([[], []], [None, None], 0)
