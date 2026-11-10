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
        assert [list(s.layers) for s in schemas.values()] == [
            ["layer@tp0"],
            ["layer@tp1"],
        ]
        assert workers[1]._get_layer_group_id("layer@tp1") == 0
        assert workers[1]._shard_args() == (1, 2)
        # The scheduler has no TP rank and addresses the whole replica.
        assert scheduler._shard_args() == ()
        scheduler.set_xfer_handshake_metadata(schemas)
        assert scheduler._kv_schema.layers == {
            "layer@tp0": ((2, 4), torch.float32),
            "layer@tp1": ((2, 4), torch.float32),
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
