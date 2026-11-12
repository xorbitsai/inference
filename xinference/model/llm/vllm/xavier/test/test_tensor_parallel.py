# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
import asyncio
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from ..request_transfer import LayerRead
from ..snapshot import KVSnapshotStore
from ..transfer import TransferActor


def _groups(*specs):
    return SimpleNamespace(
        num_blocks=8,
        kv_cache_groups=[
            SimpleNamespace(layer_names=list(names), kv_cache_spec=spec)
            for names, spec in specs
        ],
    )


def _spec(prefix_cacheable=True, block_size=16):
    return SimpleNamespace(prefix_cacheable=prefix_cacheable, block_size=block_size)


@pytest.fixture
def shared_config(connector_config):
    # Ordinary replicas: no P/D role, no GPU cache budget.
    connector_config.kv_transfer_config.get_from_extra_config = lambda *a: {
        "rank": 1,
        "role": "hybrid",
    }
    return connector_config


def _worker(connector_module, config, caches, tp_rank, kv_caches):
    instance = connector_module.XavierConnector(config, None, caches)
    # vLLM initializes the TP group in each worker before its connector
    # registers the caches; the scheduler process has no TP group.
    instance._current_tp_rank = lambda: tp_rank
    instance.register_kv_caches(kv_caches)
    return instance


@pytest.mark.parametrize(
    "xavier_config",
    [
        {"rank": 1, "role": "prefill", "gpu_cache_bytes": 0},
        {"rank": 1, "role": "hybrid", "gpu_cache_bytes": 1024},
    ],
)
def test_tensor_parallel_requires_cpu_snapshots(
    connector_module, shared_config, xavier_config
):
    shared_config.parallel_config.tensor_parallel_size = 2
    shared_config.kv_transfer_config.get_from_extra_config = lambda *a: xavier_config
    with pytest.raises(ValueError, match="TP>1"):
        connector_module.XavierConnector(shared_config, None, _groups())


def test_multimodal_requires_shared_snapshots(connector_module, shared_config):
    shared_config.model_config.is_multimodal_model = True
    instance = connector_module.XavierConnector(shared_config, None, _groups())
    instance.shutdown()
    shared_config.kv_transfer_config.get_from_extra_config = lambda *a: {
        "rank": 1,
        "role": "prefill",
        "gpu_cache_bytes": 0,
    }
    with pytest.raises(ValueError, match="multimodal"):
        connector_module.XavierConnector(shared_config, None, _groups())


@pytest.mark.parametrize("name", ["prefix", "cross_engine", "hybrid_gpu"])
def test_specialized_connectors_keep_one_text_worker(name):
    source = (
        __import__("pathlib").Path(__file__).parents[1].joinpath(f"{name}.py")
    ).read_text()
    assert "_shards_snapshots = False" in source


def test_workers_register_rank_qualified_shards(connector_module, shared_config):
    shared_config.parallel_config.tensor_parallel_size = 2
    caches = _groups((["layer"], _spec()))
    workers = [
        _worker(
            connector_module,
            shared_config,
            caches,
            rank,
            {"layer": torch.zeros(8, 2, 4)},
        )
        for rank in range(2)
    ]
    scheduler = connector_module.XavierConnector(shared_config, None, caches)
    try:
        schemas = {i: w.get_handshake_metadata() for i, w in enumerate(workers)}
        # A rank keeps its caches under rank-qualified names and exposes them
        # to the replica as one packed row per block.
        assert [list(w._registered_kv_caches) for w in workers] == [
            ["layer@tp0"],
            ["layer@tp1"],
        ]
        rows = [w._shard_rows for w in workers]
        assert [list(s.layers) for s in schemas.values()] == [
            [rows[0].layer],
            [rows[1].layer],
        ]
        assert rows[0].layer.endswith("@tp0") and rows[1].layer.endswith("@tp1")
        # Ranks that pack the same caches agree on the row's name and size.
        assert rows[0].layer[:-1] == rows[1].layer[:-1]
        assert rows[1].row_bytes == 2 * 4 * 4
        assert rows[1].layout == [("layer@tp1", (2, 4), torch.float32, 0, 32)]
        assert workers[1]._get_layer_group_id("layer@tp1") == 0
        assert workers[1]._shard_args() == (1, 2)
        # The scheduler has no TP rank and addresses the whole replica.
        assert scheduler._shard_args() == ()
        scheduler.set_xfer_handshake_metadata(schemas)
        assert scheduler._kv_schema.layers == {
            rows[0].layer: ((32,), torch.uint8),
            rows[1].layer: ((32,), torch.uint8),
        }
        with pytest.raises(ValueError, match="one worker per TP rank"):
            scheduler.set_xfer_handshake_metadata({0: schemas[0]})
        with pytest.raises(ValueError, match="Inconsistent"):
            scheduler.set_xfer_handshake_metadata({0: schemas[0], 1: schemas[0]})
    finally:
        for instance in (*workers, scheduler):
            instance.shutdown()


def test_single_worker_layer_names_are_unchanged(connector):
    connector.register_kv_caches({"layer": torch.zeros(8, 2, 4)})
    assert list(connector.get_handshake_metadata().layers) == ["layer"]
    assert connector._shard_args() == ()


def test_only_prefix_cacheable_groups_are_shared(connector_module, shared_config):
    # A ring buffer holds one block per request; vLLM rebuilds it and the
    # replayed sliding window after a prefix hit.
    caches = _groups(
        (["ring"], _spec(prefix_cacheable=False, block_size=8)),
        (["latent", "index"], _spec()),
        (["window"], _spec(prefix_cacheable=False, block_size=32)),
    )
    instance = connector_module.XavierConnector(shared_config, None, caches)
    try:
        shared = torch.zeros(8, 2, 4)
        instance.register_kv_caches(
            {
                "ring": torch.zeros(8, 3),
                "latent": shared,
                "alias": shared,  # A layer reading another layer's cache.
                "index": torch.zeros(8, 5),
                "window": torch.zeros(8, 7),
            }
        )
        assert list(instance.get_handshake_metadata().layers) == ["latent", "index"]
        assert instance._block_size == 16

        meta = connector_module.XavierConnectorMetadata()
        instance._add_store_request(meta, "r", list(range(33)), [[9], [1, 2], [0, 5]])
        (store,) = meta.store_requests
        assert store.block_ids == [1, 2]
        assert len(store.block_hashes) == 2
        # Groups that are not shared carry no block IDs; one ring block per
        # request cannot be indexed by hashed block.
        assert store.block_ids_by_group == [[], [1, 2], []]
        (missing,) = instance._missing_store_requests([store], {store.block_hashes[0]})
        assert missing.block_ids == [2]
        assert missing.block_ids_by_group == [[], [2], []]
        assert instance._get_source_block_ids(missing, "index") == [2]

        instance._requests_need_load["r"] = connector_module.XavierLoadRequest(
            "r", {2: {40: 0, 41: 1}}
        )
        blocks = SimpleNamespace(
            get_block_ids=lambda: ([9], [1, 2], [0, 5]),
            blocks=([], [], []),
        )
        instance.update_state_after_alloc(
            SimpleNamespace(request_id="r", kv_transfer_params={}), blocks, 32
        )
        load = instance._requests_need_load["r"]
        assert load.local_transfers_by_group == {1: {2: {40: 1, 41: 2}}}
        assert instance._get_local_transfer_map(load, "index", 2) == {40: 1, 41: 2}
        # Only the prefix-cacheable group identifies the request's blocks.
        instance.request_finished = lambda request, block_ids: block_ids
        assert instance.request_finished_all_groups(None, ([9], [1, 2], [0, 5])) == [
            1,
            2,
        ]
    finally:
        instance.shutdown()


def test_load_maps_only_blocks_the_scheduler_accepted(connector, connector_module):
    # Four 16-token blocks were offered after a 16-token local hit. vLLM
    # accepted 32 external tokens and has allocated three blocks so far.
    connector._requests_need_load["r"] = connector_module.XavierLoadRequest(
        "r", {2: {41: 1, 42: 2, 43: 3, 44: 4}}, local_hit_tokens=16
    )
    blocks = SimpleNamespace(get_block_ids=lambda: ([7, 8, 9],), blocks=([],))
    request = SimpleNamespace(request_id="r", kv_transfer_params={})
    connector.update_state_after_alloc(request, blocks, 32)
    load = connector._requests_need_load["r"]
    assert load.local_transfers_by_group == {0: {2: {41: 8, 42: 9}}}

    # A partial final block of the accepted range is still loaded.
    connector._requests_need_load["r"] = connector_module.XavierLoadRequest(
        "r", {2: {41: 1, 42: 2}}, local_hit_tokens=16
    )
    connector.update_state_after_alloc(request, blocks, 17)
    load = connector._requests_need_load["r"]
    assert load.local_transfers_by_group == {0: {2: {41: 8, 42: 9}}}


@pytest.mark.parametrize("mode", ["tp", "multimodal"])
def test_shards_and_multimodal_keys_need_one_block_size(
    connector_module, shared_config, mode
):
    caches = _groups((["a"], _spec(block_size=16)), (["b"], _spec(block_size=32)))
    # Single-GPU text replicas keep the engine's configured block size.
    instance = connector_module.XavierConnector(shared_config, None, caches)
    assert instance._block_size == 16
    instance.shutdown()
    if mode == "tp":
        shared_config.parallel_config.tensor_parallel_size = 2
    else:
        shared_config.model_config.is_multimodal_model = True
    with pytest.raises(ValueError, match="one block size"):
        connector_module.XavierConnector(shared_config, None, caches)


def _image(offset, length, identifier):
    return SimpleNamespace(
        mm_position=SimpleNamespace(offset=offset, length=length),
        identifier=identifier,
    )


def test_multimodal_items_identify_their_blocks(connector):
    # Placeholder tokens are the same for every image.
    tokens = [7] * 64
    keys = connector._mm_hash_keys
    text = connector._build_xavier_hashes(tokens)
    red = connector._build_xavier_hashes(tokens, None, keys([_image(20, 20, "red")]))
    blue = connector._build_xavier_hashes(tokens, None, keys([_image(20, 20, "blue")]))
    assert red == connector._build_xavier_hashes(
        tokens, None, keys([_image(20, 20, "red")])
    )
    # Block 0 precedes the image; blocks 1-2 contain it; block 3 chains on them.
    assert red[0] == blue[0] == text[0]
    assert all(len({r, b, t}) == 3 for r, b, t in zip(red[1:], blue[1:], text[1:]))
    moved = connector._build_xavier_hashes(tokens, None, keys([_image(21, 20, "red")]))
    assert moved[1] != red[1]
    # An item beyond the hashed prefix does not affect it.
    assert connector._build_xavier_hashes(
        tokens[:16], None, keys([_image(20, 20, "red")])
    ) == connector._build_xavier_hashes(tokens[:16])


def test_multimodal_requests_probe_and_store_with_item_identity(
    connector, connector_module
):
    tokens = [7] * 33
    request = SimpleNamespace(
        request_id="r",
        prompt_token_ids=tokens,
        mm_features=[_image(4, 8, "red")],
        kv_transfer_params={},
    )
    expected = connector._build_xavier_hashes(
        tokens[:32], None, connector._mm_hash_keys(request.mm_features)
    )
    queried = []

    async def query(request_id, hashes):
        queried.extend(hashes)
        return {}

    connector._query_remote_blocks = query
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert queried == expected[:1]

    connector.update_state_after_alloc(request, None, 0)
    meta = connector_module.XavierConnectorMetadata()
    connector._add_store_request(meta, "r", tokens, [[1, 2]])
    assert meta.store_requests[0].block_hashes == [key for key, _ in expected]
    connector._requests_need_load.clear()
    connector.request_finished(request, [1, 2])
    assert "r" not in connector._request_mm_keys


def test_unidentified_multimodal_items_are_rejected(connector):
    request = SimpleNamespace(
        request_id="r", prompt_token_ids=list(range(17)), mm_features=[object()]
    )
    with pytest.raises(ValueError, match="identified multimodal"):
        connector.get_num_new_matched_tokens(request, 0)


def _actor(**fields):
    return SimpleNamespace(_kv_schema_v1=None, **fields)


def test_replica_schema_waits_for_every_shard():
    actor = _actor()
    shard = {"layer@tp0": ((2,), torch.float16)}
    other = {"layer@tp1": ((2,), torch.float16)}
    TransferActor.configure_kv_schema_v1(actor, 16, shard, "auto", 0, 2)
    assert actor._kv_schema_v1 is None
    # Re-registration by a restarted worker is idempotent.
    TransferActor.configure_kv_schema_v1(actor, 16, shard, "auto", 0, 2)
    TransferActor.configure_kv_schema_v1(actor, 16, other, "auto", 1, 2)
    assert actor._kv_schema_v1 == (16, {**shard, **other}, "auto")
    # The scheduler registers the merged schema it collected from the workers.
    TransferActor.configure_kv_schema_v1(actor, 16, {**shard, **other}, "auto")
    with pytest.raises(ValueError, match="schema changed"):
        TransferActor.configure_kv_schema_v1(
            actor, 16, {"layer@tp0": ((4,), torch.float16)}, "auto", 0, 2
        )
    with pytest.raises(ValueError, match="schema changed"):
        TransferActor.configure_kv_schema_v1(actor, 16, shard, "auto")


@pytest.mark.parametrize(
    "shards,error",
    [
        ([(16, "layer", "auto"), (16, "layer", "auto")], "overlapping"),
        ([(16, "a", "auto"), (32, "b", "auto")], "different"),
        ([(16, "a", "auto"), (16, "b", "fp8")], "different"),
    ],
)
def test_inconsistent_shards_are_rejected(shards, error):
    actor = _actor()
    with pytest.raises(ValueError, match=error):
        for index, (block_size, layer, dtype) in enumerate(shards):
            TransferActor.configure_kv_schema_v1(
                actor, block_size, {layer: ((2,), torch.float16)}, dtype, index, 2
            )


def test_blocks_publish_once_all_shards_are_staged():
    store = KVSnapshotStore(4)
    actor = _actor(_snapshot_store=store)
    for rank in range(2):
        TransferActor.configure_kv_schema_v1(
            actor, 16, {f"layer@tp{rank}": ((2,), torch.float32)}, "auto", rank, 2
        )
    store.stage("layer@tp0", [5, 6], torch.ones(2, 2))
    # One shard alone is not a readable block, whatever its worker reports.
    assert TransferActor.publish_blocks_v1(actor, [5, 6], {"layer@tp0"}) == ([], [])
    assert not store.reserve("2:r", [5])
    store.stage("layer@tp1", [5], torch.full((1, 2), 2.0))
    assert TransferActor.publish_blocks_v1(actor, [5, 6], {"layer@tp1"}) == ([5], [])
    assert store.reserve("2:r", [5])
    assert store.read("layer@tp1", [5]).tolist() == [[2.0, 2.0]]


def test_unsharded_publication_uses_reported_layers():
    store = KVSnapshotStore(2)
    actor = _actor(_snapshot_store=store)
    TransferActor.configure_kv_schema_v1(
        actor, 16, {"K": ((2,), torch.float32), "V": ((2,), torch.float32)}, "auto"
    )
    store.stage("K", [5], torch.ones(1, 2))
    assert TransferActor.publish_blocks_v1(actor, [5], {"K"}) == ([5], [])


@pytest.mark.asyncio
async def test_lease_is_released_after_every_shard(monkeypatch):
    import xoscar as xo

    remote = SimpleNamespace(release_blocks_v1=AsyncMock())
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=remote))
    actor = SimpleNamespace(_world_addresses=["a", "b", "c"])
    transfers = {2: {1: 0}}

    await TransferActor.release_remote_blocks_v1(actor, "lease", transfers, 0, 2)
    await TransferActor.release_remote_blocks_v1(actor, "lease", transfers, 0, 2)
    remote.release_blocks_v1.assert_not_awaited()
    await TransferActor.release_remote_blocks_v1(actor, "lease", transfers, 1, 2)
    remote.release_blocks_v1.assert_awaited_once_with("lease")
    assert not actor._shard_releases_v1

    # The scheduler owns the lease and releases it without waiting for workers.
    await TransferActor.release_remote_blocks_v1(actor, "other", transfers, 0, 2)
    await TransferActor.release_remote_blocks_v1(actor, "other", transfers)
    assert remote.release_blocks_v1.await_count == 2
    assert not actor._shard_releases_v1


@pytest.mark.asyncio
async def test_concurrent_shard_reads_do_not_interleave(monkeypatch):
    import xoscar as xo
    from xoscar.collective import xoscar_pygloo as xp

    events = []
    release = threading.Event()

    async def start_send(to_rank, reads):
        events.append(("send", reads[0].layer))

    def recv(context, pointer, size, dtype, from_rank):
        events.append(("recv", size))
        assert release.wait(5)

    monkeypatch.setattr(xp, "recv", recv)
    monkeypatch.setattr(
        xo,
        "actor_ref",
        AsyncMock(
            return_value=SimpleNamespace(start_send_request_blocks_v1=start_send)
        ),
    )
    actor = SimpleNamespace(
        _world_addresses=["sender"], _rank=1, _context=None, get_gloo_dtype=lambda t: t
    )
    reads = [
        [LayerRead(f"K@tp{rank}", [1], [0], (rank + 1,), torch.float32)]
        for rank in range(2)
    ]
    tasks = [
        asyncio.create_task(TransferActor.read_request_blocks_v1(actor, 0, read))
        for read in reads
    ]
    for _ in range(50):
        await asyncio.sleep(0.01)
        if len(events) >= 2:
            break
    # The second shard waits for the first send/receive pair to complete.
    assert events == [("send", "K@tp0"), ("recv", 4)]
    release.set()
    payloads = await asyncio.gather(*tasks)
    assert events == [
        ("send", "K@tp0"),
        ("recv", 4),
        ("send", "K@tp1"),
        ("recv", 8),
    ]
    assert [payload.numel() for payload in payloads] == [4, 8]


@pytest.mark.skipif(sys.platform != "linux", reason="requires /dev/shm")
@pytest.mark.asyncio
async def test_each_shard_reads_through_its_own_shared_buffer():
    actor = SimpleNamespace()
    try:
        metadata = [TransferActor.get_local_read_metadata_v1(actor, i) for i in (0, 1)]
        assert metadata[0]["token"] != metadata[1]["token"]
        # Repeated discovery by a worker reuses its buffer.
        assert TransferActor.get_local_read_metadata_v1(actor, 1) == metadata[1]
        assert getattr(actor, "_local_read_buffer_v1", None) is None

        async def read(from_rank, reads, *, _buffer):
            _buffer.fill_(from_rank)

        actor.read_request_blocks_v1 = read
        # This peer is not on the same host: the buffers receive over Gloo.
        actor._shared_pack_ranks_v1 = {7: False}
        reads = [LayerRead("K", [1], [0], (2,), torch.float32)]
        # Both shards hold a lease at once; neither falls back to tensor RPC.
        leases = [
            await TransferActor.read_request_blocks_local_v1(
                actor, 7, reads, item["token"]
            )
            for item in metadata
        ]
        assert None not in leases and leases[0] != leases[1]
        assert (
            await TransferActor.read_request_blocks_local_v1(
                actor, 7, reads, b"unknown"
            )
            is None
        )
        for lease in leases:
            TransferActor.release_local_read_v1(actor, lease)
        owners = list(actor._local_read_shard_buffers_v1.values())
        assert all(owner._lease is None for owner in owners)
    finally:
        for owner in getattr(actor, "_local_read_shard_buffers_v1", {}).values():
            owner.close()


def test_tp_workers_stage_their_shard_through_host_memory(
    connector_module, shared_config, monkeypatch
):
    # A CUDA IPC export would make the replica's single TransferActor wait on
    # a GPU that a rank may hold in a collective while waiting for the actor.
    shared_config.parallel_config.tensor_parallel_size = 2
    caches = _groups((["layer"], _spec()))
    worker = _worker(
        connector_module, shared_config, caches, 1, {"layer": torch.zeros(8, 2, 4)}
    )
    try:
        assert worker._packed_gather is None and worker._gpu_export_arena is None
        store = connector_module.XavierStoreRequest("r", [4], [222], [[4]])
        assert not worker._can_queue_ipc_export([store])
        worker._get_connector_metadata = (
            lambda: connector_module.XavierConnectorMetadata(store_requests=[store])
        )

        async def filter_requests(pending):
            return pending

        staged = []
        # Without same-host buffers the worker stages through actor RPCs.
        worker._save_shard_blocks = lambda pending: False
        worker._kv_schema_complete = True  # Every rank has registered.
        worker._filter_store_requests = filter_requests
        worker._stage_missing_registered_layers = staged.append
        worker._register_blocks = AsyncMock()
        worker._enqueue_ipc_export = Mock(side_effect=AssertionError("CUDA IPC"))
        worker.wait_for_save()
        assert staged == [[store]]
        worker._register_blocks.assert_awaited_once_with([store])

        # Staging is synchronous, so each step takes a bounded leading part.
        # Every rank computes the same bound and completes the same blocks.
        block_bytes = worker._cpu_export_size([store])
        monkeypatch.setattr(connector_module, "_MAX_SHARD_STAGE_BYTES", 3 * block_bytes)
        first = connector_module.XavierStoreRequest("a", [1, 2], [11, 12], [[1, 2]])
        second = connector_module.XavierStoreRequest(
            "b", [3, 4, 5], [13, 14, 15], [[3, 4, 5]]
        )
        admitted = worker._admit_shard_stages([first, second])
        assert [request.block_hashes for request in admitted] == [[11, 12], [13]]
        assert admitted[1].block_ids_by_group == [[3]]

        # A chunked prefill repeats blocks that still wait for peer shards.
        # They do not use the next step's budget, so this rank keeps up.
        admitted = worker._admit_shard_stages([first, second])
        assert [request.block_hashes for request in admitted] == [[14, 15]]
        assert worker._admit_shard_stages([first, second]) == []
        # If the peers never completed a block, it becomes eligible again.
        later = time.monotonic() + 60
        monkeypatch.setattr(connector_module.time, "monotonic", lambda: later)
        admitted = worker._admit_shard_stages([first])
        assert [request.block_hashes for request in admitted] == [[11, 12]]
    finally:
        worker.shutdown()


def test_rank_loads_peer_rows_into_its_caches(connector_module, shared_config):
    shared_config.parallel_config.tensor_parallel_size = 2
    groups = _groups((["a", "b"], _spec()))
    values = {
        "a": torch.arange(64, dtype=torch.float32).reshape(8, 2, 4),
        "b": torch.arange(24, dtype=torch.int16).reshape(8, 3),
    }
    producer = _worker(connector_module, shared_config, groups, 1, values)
    consumer = _worker(
        connector_module,
        shared_config,
        groups,
        1,
        {name: torch.zeros_like(value) for name, value in values.items()},
    )
    try:
        rows = consumer._shard_rows
        assert rows.layer == producer._shard_rows.layer
        assert rows.row_bytes == 2 * 4 * 4 + 3 * 2
        stored = {44: 4, 66: 6}  # Content key -> block on the producer.
        reads = []

        async def read(transfer, rank, batch, device):
            reads.append((rank, [(part.layer, part.keys) for part in batch]))
            keys = [key for part in batch for key in part.keys]
            sources = {name: [stored[k] for k in keys] for name in rows.caches}
            return producer._gather_shard_rows(sources, len(keys))[0].reshape(-1)

        async def transfer_ref():
            return None

        async def attach(transfer):
            return None  # No same-host buffer: rows arrive as tensors.

        consumer._read_shard_rows = read
        consumer._attach_local_read = attach
        consumer._get_transfer_ref = transfer_ref
        consumer._load_request_blocks(
            connector_module.XavierLoadRequest(
                "r",
                {2: {66: 0, 44: 1}},
                local_transfers_by_group={0: {2: {66: 1, 44: 3}}},
            )
        )
        # One read fetched each row once, for every cache of the rank.
        assert reads == [(2, [(rows.layer, [66, 44])])]
        for name, value in values.items():
            loaded = consumer._registered_kv_caches[f"{name}@tp1"]
            assert torch.equal(loaded[[1, 3]], value[[6, 4]])
            assert not loaded[[0, 2, 4, 5, 6, 7]].any()
    finally:
        producer.shutdown()
        consumer.shutdown()


def _shard_actor(tracker, monkeypatch, shards=2):
    from .. import transfer as transfer_module

    async def actor_ref(**kwargs):
        return tracker

    monkeypatch.setattr(transfer_module.xo, "actor_ref", actor_ref)
    actor = _actor(
        _snapshot_store=KVSnapshotStore(8),
        _snapshot_export_lock_v1=asyncio.Lock(),
        _rank=1,
    )
    actor.publish_blocks_v1 = TransferActor.publish_blocks_v1.__get__(actor)
    return actor


def _close_shard_actor(actor):
    pool = actor.__dict__.pop("_shard_stage_pool_v1", None)
    if pool is not None:
        pool.shutdown()
    for reader in actor.__dict__.get("_shard_stage_readers_v1", {}).values():
        reader.close()


def _handover(connector_module, config, actor, tp_rank, values):
    """A TP worker whose actor calls reach ``actor`` on the worker's loop."""
    caches = _groups((["layer"], _spec()))
    worker = _worker(connector_module, config, caches, tp_rank, {"layer": values})
    calls = []

    class Ref:
        async def stage_shard_blocks_v1(self, *args):
            calls.append(args)
            return await TransferActor.stage_shard_blocks_v1(actor, *args)

    async def transfer_ref():
        return Ref()

    schema = worker._kv_schema
    TransferActor.configure_kv_schema_v1(
        actor, schema.block_size, schema.layers, schema.cache_dtype, tp_rank, 2
    )
    worker._kv_schema_complete = True  # The test registers every rank up front.
    worker._tp_rank = tp_rank
    worker._get_transfer_ref = transfer_ref
    worker._stage_missing_registered_layers = Mock(
        side_effect=AssertionError("staged through actor RPCs")
    )
    return worker, calls


def _settle(worker):
    """Wait for the actor's replies, then let the worker collect them."""
    jobs = worker._shard_stage.jobs
    if jobs:
        worker._call(asyncio.wait([job[0] for job in jobs]))
    worker.get_finished(set())


def _store_step(connector_module, worker, *requests):
    worker._get_connector_metadata = lambda: connector_module.XavierConnectorMetadata(
        store_requests=list(requests)
    )
    worker.wait_for_save()


@pytest.mark.skipif(sys.platform != "linux", reason="requires /dev/shm")
def test_tp_workers_hand_their_shard_over_without_waiting(
    connector_module, shared_config, monkeypatch
):
    shared_config.parallel_config.tensor_parallel_size = 2
    tracker = SimpleNamespace(
        unregister_blocks=AsyncMock(), register_snapshot_blocks=AsyncMock()
    )
    actor = _shard_actor(tracker, monkeypatch)
    values = [
        torch.arange(64, dtype=torch.float32).reshape(8, 2, 4) + 100 * rank
        for rank in (0, 1)
    ]
    workers = [
        _handover(connector_module, shared_config, actor, rank, values[rank])
        for rank in (0, 1)
    ]
    request = connector_module.XavierStoreRequest("r", [4, 6], [44, 66], [[4, 6]])
    store = actor._snapshot_store
    try:
        (first, first_calls), (second, second_calls) = workers
        _store_step(connector_module, first, request)
        # The request was submitted, and the step did not wait for its reply.
        state = first._shard_stage
        assert len(first_calls) == 1 and len(state.jobs) == 1
        assert state.pending == {44, 66}
        assert len(state.free) == connector_module._SHARD_STAGE_BUFFERS - 1
        # A chunked prefill repeats the blocks while the reply is pending.
        _store_step(connector_module, first, request)
        assert len(first_calls) == 1

        _settle(first)
        assert not state.jobs and not state.pending
        assert len(state.free) == connector_module._SHARD_STAGE_BUFFERS
        assert set(state.known) == {44, 66}
        # One shard alone is not readable, and nothing reached the tracker.
        assert not store.reserve("2:r", [44])
        tracker.register_snapshot_blocks.assert_not_awaited()

        _store_step(connector_module, second, request)
        _settle(second)
        tracker.register_snapshot_blocks.assert_awaited_once_with(0, [44, 66], 1)
        assert store.reserve("2:r", [44, 66])
        for rank, (worker, _) in enumerate(workers):
            stored = store.read(worker._shard_rows.layer, [44, 66])
            assert torch.equal(
                stored.view(torch.float32).view(2, 2, 4), values[rank][[4, 6]]
            )

        # Confirmed blocks cost nothing until the trust in them expires. Then
        # the worker only asks, again without waiting.
        _store_step(connector_module, first, request)
        assert len(first_calls) == 1
        later = time.monotonic() + 60
        monkeypatch.setattr(connector_module.time, "monotonic", lambda: later)
        _store_step(connector_module, first, request)
        assert first_calls[-1][1] == [] and first_calls[-1][4] == [44, 66]
        _settle(first)
        assert set(state.known) == {44, 66}

        # A block the store evicted is staged again by the next request.
        later += 60
        store.release("2:r")
        while store._evict_one(set()):
            pass
        _store_step(connector_module, first, request)
        _settle(first)
        assert not state.known
        _store_step(connector_module, first, request)
        assert [keys for _, _, keys in first_calls[-1][1]] == [[44, 66]]
    finally:
        for worker, _ in workers:
            worker.shutdown()
        _close_shard_actor(actor)


@pytest.mark.skipif(sys.platform != "linux", reason="requires /dev/shm")
def test_shard_handover_is_bounded_and_falls_back(
    connector_module, shared_config, monkeypatch
):
    shared_config.parallel_config.tensor_parallel_size = 2
    tracker = SimpleNamespace(
        unregister_blocks=AsyncMock(), register_snapshot_blocks=AsyncMock()
    )
    actor = _shard_actor(tracker, monkeypatch)
    worker, calls = _handover(
        connector_module, shared_config, actor, 0, torch.zeros(8, 2, 4)
    )
    TransferActor.configure_kv_schema_v1(
        actor, 16, {"rows@tp1": ((32,), torch.uint8)}, "auto", 1, 2
    )
    try:
        # Every rank takes the same leading blocks of an oversized step.
        row_bytes = 2 * 4 * 4
        monkeypatch.setattr(connector_module, "_MAX_SHARD_STAGE_BYTES", 3 * row_bytes)
        first = connector_module.XavierStoreRequest("a", [1, 2], [11, 12], [[1, 2]])
        second = connector_module.XavierStoreRequest(
            "b", [3, 4, 5], [13, 14, 15], [[3, 4, 5]]
        )
        _store_step(connector_module, worker, first, second)
        assert [keys for _, _, keys in calls[-1][1]] == [[11, 12, 13]]
        _settle(worker)
        _store_step(connector_module, worker, first, second)
        assert [keys for _, _, keys in calls[-1][1]] == [[14, 15]]
        _settle(worker)

        # An actor that cannot attach the buffers gets tensors instead.
        from ....xavier.backends.torch import local_read

        for reader in actor.__dict__.pop("_shard_stage_readers_v1").values():
            reader.close()
        monkeypatch.setattr(
            local_read.LocalReadBuffer,
            "attach",
            classmethod(lambda cls, metadata, **kwargs: None),
        )
        third = connector_module.XavierStoreRequest("c", [6], [16], [[6]])
        _store_step(connector_module, worker, third)
        _settle(worker)
        assert not worker._shard_stage.buffers

        async def filter_requests(pending):
            return pending

        staged = []
        worker._filter_store_requests = filter_requests
        worker._stage_missing_registered_layers = staged.append
        worker._register_blocks = AsyncMock()
        _store_step(connector_module, worker, third)
        assert [request.block_hashes for request in staged[0]] == [[16]]
    finally:
        worker.shutdown()
        _close_shard_actor(actor)


@pytest.mark.skipif(sys.platform != "linux", reason="requires /dev/shm")
def test_buffers_of_an_unacknowledged_handover_are_never_reused(
    connector_module, shared_config, monkeypatch
):
    # The worker loses the reply while the actor still copies the buffer. A
    # second request must not land in that buffer, or the actor would store
    # its row under the first request's key.
    from concurrent.futures import ThreadPoolExecutor

    shared_config.parallel_config.tensor_parallel_size = 2
    tracker = SimpleNamespace(
        unregister_blocks=AsyncMock(), register_snapshot_blocks=AsyncMock()
    )
    actor = _shard_actor(tracker, monkeypatch)
    values = torch.arange(64, dtype=torch.float32).reshape(8, 2, 4)
    worker, _ = _handover(connector_module, shared_config, actor, 0, values)
    TransferActor.configure_kv_schema_v1(
        actor, 16, {"rows@tp1": ((32,), torch.uint8)}, "auto", 1, 2
    )
    gate = threading.Event()
    pool = ThreadPoolExecutor(1)

    class Held:
        def submit(self, fn, *args):
            return pool.submit(lambda: (gate.wait(), fn(*args))[1])

    monkeypatch.setattr(
        TransferActor, "_shard_stage_executor_v1", lambda self: Held(), raising=False
    )
    copies = []

    class Ref:
        async def stage_shard_blocks_v1(self, *args):
            copy = asyncio.ensure_future(
                TransferActor.stage_shard_blocks_v1(actor, *args)
            )
            copies.append(copy)
            if len(copies) == 1:
                await asyncio.sleep(0)  # The actor takes its view of the buffer.
                raise ConnectionError("connection closed")
            return await copy

    async def transfer_ref():
        return Ref()

    worker._get_transfer_ref = transfer_ref
    store = actor._snapshot_store
    layer = worker._shard_rows.layer
    try:
        state = worker._shard_stage_state()
        buffers = list(state.buffers)
        _store_step(
            connector_module,
            worker,
            connector_module.XavierStoreRequest("a", [2], [44], [[2]]),
        )
        (job,) = state.jobs
        used = job[1][0][0]
        worker._call(asyncio.wait([job[0]]))
        worker.get_finished(set())
        # The buffer left the pool for good and a fresh one took its place.
        assert used not in state.buffers and used not in state.free
        assert len(state.buffers) == len(buffers) == len(state.free)
        assert 44 not in state.known

        _store_step(
            connector_module,
            worker,
            connector_module.XavierStoreRequest("b", [3], [66], [[3]]),
        )
        gate.set()
        worker._call(asyncio.wait(copies))
        _settle(worker)
        assert torch.equal(
            store.blocks[66][layer].view(torch.float32).view(2, 4), values[3]
        )
        # The first request's key holds its own row or nothing at all.
        block = store.blocks.get(44)
        assert block is None or torch.equal(
            block[layer].view(torch.float32).view(2, 4), values[2]
        )
    finally:
        gate.set()
        pool.shutdown()
        worker.shutdown()
        _close_shard_actor(actor)


@pytest.mark.skipif(sys.platform != "linux", reason="requires /dev/shm")
def test_copies_that_no_longer_fit_the_surviving_pool_use_actor_rpcs(
    connector_module, shared_config, monkeypatch
):
    # Lost replies retire buffers, and /dev/shm may have no room to replace
    # them. A copy sized for the old pool must not wait forever for buffers
    # that no longer exist, and later copies have to fit the smaller pool.
    import errno

    from ....xavier.backends.torch import local_read

    shared_config.parallel_config.tensor_parallel_size = 2
    worker = _worker(
        connector_module,
        shared_config,
        _groups((["layer"], _spec())),
        0,
        {"layer": torch.arange(64, dtype=torch.float32).reshape(8, 2, 4)},
    )
    worker._kv_schema_complete = True
    row_bytes = worker._shard_rows.row_bytes
    monkeypatch.setattr(connector_module, "_MAX_SHARD_STAGE_BYTES", 6 * row_bytes)
    lost = []
    handovers = []
    rpc_rows = []
    disconnect = asyncio.Event()

    class Ref:
        async def stage_shard_blocks_v1(self, shard, stages, *args):
            keys = [key for _, _, staged in stages for key in staged]
            if len(lost) < 2:
                lost.append(keys)
                await disconnect.wait()
                raise ConnectionError("connection closed")
            handovers.append([staged for _, _, staged in stages])
            return keys

        async def stage_layer_batches_v1(self, entries, metadata):
            rpc_rows.extend(key for _, keys, _ in entries for key in keys)

        refresh_snapshot_blocks_v1 = AsyncMock(return_value=([], []))

    async def transfer_ref():
        return Ref()

    worker._get_transfer_ref = transfer_ref

    def request(first_key):
        return connector_module.XavierStoreRequest(
            "r", list(range(6)), list(range(first_key, first_key + 6)), [list(range(6))]
        )

    try:
        state = worker._shard_stage_state()
        for buffer in state.buffers:
            buffer.close()
        # Eight slots of two rows each: a step of six rows takes three.
        state.buffers = [local_read.SharedReadBuffer(2 * row_bytes) for _ in range(8)]
        state.free = list(state.buffers)
        _store_step(connector_module, worker, request(100))
        _store_step(connector_module, worker, request(200))
        assert len(state.jobs) == 2 and len(state.free) == 2
        # The third step's copy waits: only two slots are free for now.
        _store_step(connector_module, worker, request(300))
        assert len(state.copies) == 1 and len(state.jobs) == 2 and not rpc_rows

        def no_space(*args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(local_read, "SharedReadBuffer", no_space)
        disconnect.set()
        _settle(worker)
        # Both replies were lost and nothing could be replaced: two slots are
        # left, fewer than the waiting copy needs. It went through the RPCs.
        assert lost == [list(range(100, 106)), list(range(200, 206))]
        assert len(state.buffers) == 2 == len(state.free)
        assert not state.copies and not state.pending
        assert rpc_rows == list(range(300, 306))
        Ref.refresh_snapshot_blocks_v1.assert_awaited_once()
        assert set(state.known) == set(range(300, 306))

        # New steps are sized for the surviving pool and use it again.
        _store_step(connector_module, worker, request(400))
        _settle(worker)
        assert handovers == [[[400, 401], [402, 403]]] or handovers == [
            [[402, 403], [400, 401]]
        ]
        assert set(range(400, 404)) <= set(state.known)
    finally:
        worker.shutdown()


@pytest.mark.skipif(sys.platform != "linux", reason="requires /dev/shm")
@pytest.mark.asyncio
async def test_same_host_peer_fills_a_shard_buffer_directly(monkeypatch):
    from ....xavier.backends.torch.local_read import LocalReadBuffer
    from .. import transfer as transfer_module

    store = KVSnapshotStore(4)
    store.stage("K@tp0", [5, 6], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    store.publish([5, 6], {"K@tp0"})
    producer = _actor(_snapshot_store=store, _rank=1)
    reachable = True

    class Peer:
        async def pack_request_blocks_shared_v1(self, *args):
            if not reachable:
                return False
            return await TransferActor.pack_request_blocks_shared_v1(producer, *args)

    async def actor_ref(**kwargs):
        return Peer()

    monkeypatch.setattr(transfer_module.xo, "actor_ref", actor_ref)
    consumer = SimpleNamespace(_rank=2, _world_addresses={1: "producer"})
    for name in ("_read_request_blocks_shared_v1", "_retire_local_read_buffer_v1"):
        setattr(
            consumer,
            name,
            getattr(TransferActor, name).__get__(consumer),
        )
    consumer.read_request_blocks_v1 = AsyncMock()
    reads = [LayerRead("K@tp0", [6, 5], [0, 1], (2,), torch.float32)]
    local = None
    try:
        metadata = TransferActor.get_local_read_metadata_v1(consumer, 0)
        local = LocalReadBuffer.attach(metadata)
        lease = await TransferActor.read_request_blocks_local_v1(
            consumer, 1, reads, metadata["token"]
        )
        # The peer wrote the blocks itself: no Gloo receive was needed.
        consumer.read_request_blocks_v1.assert_not_awaited()
        payload = local.copy(lease, reads[0].nbytes, pin_memory=False)
        assert payload.view(torch.float32).tolist() == [3.0, 4.0, 1.0, 2.0]
        TransferActor.release_local_read_v1(consumer, lease)

        # A peer that cannot reach the buffer is remembered and uses Gloo.
        reachable = False
        for _ in range(2):
            lease = await TransferActor.read_request_blocks_local_v1(
                consumer, 1, reads, metadata["token"]
            )
            TransferActor.release_local_read_v1(consumer, lease)
        assert consumer.read_request_blocks_v1.await_count == 2
        assert consumer._shared_pack_ranks_v1 == {1: False}

        # After a failed write the peer may still own the buffer: retire it.
        consumer._shared_pack_ranks_v1.clear()
        reachable = True
        store.release_consumer(2)
        reads = [LayerRead("K@tp0", [7], [0], (2,), torch.float32)]
        with pytest.raises(KeyError):
            await TransferActor.read_request_blocks_local_v1(
                consumer, 1, reads, metadata["token"]
            )
        assert not consumer._local_read_shard_buffers_v1
        assert not local.valid()
        assert TransferActor.get_local_read_metadata_v1(consumer, 0) != metadata
    finally:
        if local is not None:
            local.close()
        for owner in getattr(consumer, "_local_read_shard_buffers_v1", {}).values():
            owner.close()
        for writer in producer.__dict__.get("_shared_pack_writers_v1", {}).values():
            writer.close()


def test_ranks_do_not_stage_before_every_shard_is_registered(
    connector_module, shared_config, monkeypatch
):
    # A backend such as LMCache is created from the replica's complete schema.
    # The rank that reaches its first save before its peer has registered must
    # neither stage into the missing store nor publish.
    from ....xavier.backends.torch import backend as backend_module
    from .. import transfer as transfer_module

    shared_config.parallel_config.tensor_parallel_size = 2
    monkeypatch.setattr(
        backend_module,
        "create_snapshot_backend",
        lambda capacity, *args, **kwargs: KVSnapshotStore(capacity),
    )
    tracker = SimpleNamespace(
        unregister_blocks=AsyncMock(), register_snapshot_blocks=AsyncMock()
    )
    actor = _actor(
        _snapshot_store=None,
        _snapshot_capacity=None,
        _cache_backend={"name": "lmcache"},
        _snapshot_export_lock_v1=asyncio.Lock(),
        _rank=1,
    )
    actor.configure_snapshots_v1 = TransferActor.configure_snapshots_v1.__get__(actor)
    actor.publish_blocks_v1 = TransferActor.publish_blocks_v1.__get__(actor)

    class Ref:
        def __getattr__(self, name):
            method = getattr(TransferActor, name)

            async def call(*args, **kwargs):
                result = method(actor, *args, **kwargs)
                return await result if asyncio.iscoroutine(result) else result

            return call

    async def actor_ref(address=None, uid=None):
        is_transfer = uid is not None and uid.startswith(TransferActor.default_uid())
        return Ref() if is_transfer else tracker

    monkeypatch.setattr(transfer_module.xo, "actor_ref", actor_ref)
    request = connector_module.XavierStoreRequest("r", [4], [44], [[4]])
    workers = [
        _worker(
            connector_module,
            shared_config,
            _groups((["layer"], _spec())),
            rank,
            {"layer": torch.full((8, 2, 4), float(rank + 1))},
        )
        for rank in (0, 1)
    ]
    try:
        for worker in workers:
            # Stage through actor RPCs on every platform.
            worker._save_shard_blocks = lambda pending: False
        first, second = workers
        _store_step(connector_module, first, request)
        assert actor._snapshot_store is None and actor._kv_schema_v1 is None
        tracker.register_snapshot_blocks.assert_not_awaited()

        # The second rank completes the schema, which creates the backend.
        _store_step(connector_module, second, request)
        store = actor._snapshot_store
        assert store is not None and not store.ready
        row = store.blocks[44][second._shard_rows.layer]
        assert row.view(torch.float32).eq(2).all()

        # The first rank stages its shard with the next request for the block.
        _store_step(connector_module, first, request)
        assert store.reserve("2:r", [44])
        row = store.read(first._shard_rows.layer, [44])
        assert row.view(torch.float32).eq(1).all()
    finally:
        for worker in workers:
            worker.shutdown()


def test_actor_skips_staging_without_a_store():
    actor = _actor(_snapshot_store=None, _kv_schema_num_shards_v1=2)
    entries = [("layer@tp0", [44], torch.zeros(8, dtype=torch.uint8))]
    TransferActor.stage_layer_batches_v1(actor, entries, [((2,), torch.float32)])
    assert TransferActor.publish_blocks_v1(actor, [44], {"layer@tp0"}) == ([], [])
