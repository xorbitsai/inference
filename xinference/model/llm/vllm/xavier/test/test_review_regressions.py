# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from ..cache_lifecycle import PDCacheLifecycleMixin
from ..snapshot import KVSnapshotStore, block_major_view
from ..transfer import TransferActor


def test_snapshots_survive_slot_reuse_and_pin_eviction():
    store = KVSnapshotStore(2)
    source = torch.tensor([[1.0], [2.0]])
    store.stage("K", [111, 222], source)
    assert not store.reserve("2:r", [111])  # Not all layers have been published.
    store.stage("V", [111, 222], source)
    assert store.publish([111, 222], {"K", "V"}) == [111, 222]
    assert store.reserve("2:r", [111])
    source.fill_(9)  # No borrowed request storage.
    store.stage("K", [111], source[:1])  # Same content key is immutable.
    store.stage("K", [333], source[:1])  # Evicts only the unleased block.
    assert store.read("K", [111]).tolist() == [[1.0]]
    assert 222 not in store.blocks
    assert len(store.blocks) == 2
    store.release("2:r")
    store.release("2:r")
    store.stage("K", [444, 555], source)
    assert not store.reserve(
        "2:next", [111]
    )  # Miss must recompute, never load new content.


def test_snapshot_full_with_readers_skips_new_writes():
    store = KVSnapshotStore(1)
    store.stage("L", [1], torch.tensor([[1.0]]))
    store.publish([1], {"L"})
    assert store.reserve("2:r", [1])
    store.stage("L", [2], torch.tensor([[2.0]]))
    assert store.publish([2], {"L"}) == []
    assert store.read("L", [1]).item() == 1
    store.release_consumer(2)
    assert not store.leases
    store.stage("L", [2], torch.tensor([[2.0]]))
    assert len(store.blocks) == 1


@pytest.mark.parametrize("kv_first", [True, False])
def test_cache_layout_round_trip(connector, connector_module, kv_first):
    canonical = torch.arange(8 * 2 * 16 * 2 * 4).view(8, 2, 16, 2, 4).float()
    source = canonical.movedim(0, 1).contiguous() if kv_first else canonical
    sent = []

    async def stage(request, layer, keys, blocks):
        sent.append((keys, blocks.clone()))

    connector._stage_layer_blocks = stage
    request = connector_module.XavierStoreRequest("r", [2, 5], [111, 222], [[2, 5]])
    connector._stage_kv_layer_for_request(request, "layer", source)
    assert sent[0][0] == [111, 222]
    assert torch.equal(sent[0][1], canonical[[2, 5]])
    destination = torch.zeros_like(source)

    async def read(*args):
        return sent[0][1]

    connector._read_layer_blocks = read
    load = connector_module.XavierLoadRequest(
        "r", {1: {111: 0, 222: 1}}, local_transfers_by_group={0: {1: {111: 3, 222: 6}}}
    )
    connector._load_layer_blocks("layer", destination, load)
    assert torch.equal(block_major_view(destination, 8)[[3, 6]], canonical[[2, 5]])
    assert block_major_view(destination, 8)[0].count_nonzero() == 0


def test_ambiguous_layout_rejected():
    with pytest.raises(ValueError, match="Ambiguous"):
        block_major_view(torch.zeros(2, 2, 16, 2, 4), 2)


@pytest.mark.parametrize(
    "resumed,new,expected", [(False, None, [0]), (True, ([4],), [4])]
)
def test_chunked_prefill_no_new_blocks_and_resume(
    connector, connector_module, resumed, new, expected
):
    connector._chunked_prefill["r"] = ([[0]], list(range(15)))
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        num_scheduled_tokens={"r": 7},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r"],
            num_computed_tokens=[8],
            new_block_ids=[new],
            resumed_req_ids={"r"} if resumed else set(),
        ),
    )
    meta = connector_module.XavierConnectorMetadata()
    connector._build_store_meta(output, meta)
    assert meta.store_requests[0].block_ids == expected
    assert not connector._chunked_prefill


@pytest.mark.parametrize("field", ["lora_config", "multimodal", "tp", "pp"])
def test_unsupported_configs_rejected(connector_module, connector_config, field):
    if field == "lora_config":
        connector_config.lora_config = object()
    elif field == "multimodal":
        connector_config.model_config.is_multimodal_model = True
    else:
        setattr(
            connector_config.parallel_config,
            "tensor_parallel_size" if field == "tp" else "pipeline_parallel_size",
            2,
        )
    with pytest.raises(ValueError):
        connector_module.XavierConnector(
            connector_config, None, SimpleNamespace(kv_cache_groups=[])
        )


@pytest.mark.asyncio
async def test_reused_request_id_publishes_new_content(connector, connector_module):
    tracker = SimpleNamespace(
        register_blocks=AsyncMock(), unregister_blocks=AsyncMock()
    )
    transfer = SimpleNamespace(
        publish_blocks_v1=AsyncMock(side_effect=[([111], []), ([222], [111])])
    )
    connector._get_tracker_ref = AsyncMock(return_value=tracker)
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    connector._registered_kv_caches = {"layer": torch.zeros(8, 2, 16, 2, 4)}
    for key in [111, 222]:
        await connector._register_blocks(
            [connector_module.XavierStoreRequest("r", [1], [key], [[1]])]
        )
    assert tracker.register_blocks.await_count == 2
    tracker.register_blocks.assert_awaited_with(0, [(222, 222)], 1)
    tracker.unregister_blocks.assert_awaited_once_with(0, 1, [111])
    assert not hasattr(connector, "_stored_requests")


def test_missed_reservation_recomputes(connector):
    connector._query_remote_blocks = AsyncMock(return_value={2: {(1, 4, 0)}})
    connector._reserve_load_request = AsyncMock(return_value=False)
    request = SimpleNamespace(request_id="r", prompt_token_ids=list(range(17)))
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert not connector._requests_need_load
    assert not connector._leased_requests


def test_load_failure_releases_snapshot(connector, connector_module):
    request = connector_module.XavierLoadRequest("r", {2: {111: 0}}, lease="1:unique")
    connector._get_connector_metadata = (
        lambda: connector_module.XavierConnectorMetadata(load_requests=[request])
    )
    connector._registered_kv_caches = {"layer": torch.zeros(8, 2, 16, 2, 4)}
    connector._load_request_blocks = Mock(side_effect=RuntimeError("failed"))
    connector._release_load_request = AsyncMock()
    with pytest.raises(RuntimeError):
        connector.start_load_kv(SimpleNamespace())
    connector._release_load_request.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_partial_reservation_rolls_back(monkeypatch):
    import xoscar as xo

    first = SimpleNamespace(
        reserve_blocks_v1=AsyncMock(return_value=True), release_blocks_v1=AsyncMock()
    )
    second = SimpleNamespace(
        reserve_blocks_v1=AsyncMock(return_value=False), release_blocks_v1=AsyncMock()
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(side_effect=[first, second]))
    actor = SimpleNamespace(
        _world_addresses=["zero", "one", "two"],
        _kv_schema_v1=(16, {"layer": ((2,), torch.float32)}, "auto"),
    )
    assert not await TransferActor.reserve_remote_blocks_v1(
        actor, "3:r", {1: {111: 0}, 2: {222: 1}}
    )
    first.reserve_blocks_v1.assert_awaited_once_with("3:r", [111], actor._kv_schema_v1)
    first.release_blocks_v1.assert_awaited_once_with("3:r")


def test_legacy_cleanup_is_targeted_and_idempotent():
    class Parent:
        def free_finished_seq_groups(self):
            self.automatic += 1

    class Scheduler(PDCacheLifecycleMixin, Parent):
        pass

    s = Scheduler()
    s.automatic = 0
    a = SimpleNamespace(request_id="A", is_finished=lambda: True)
    b = SimpleNamespace(request_id="B", is_finished=lambda: True)
    s.running = deque([a, b])
    s._transferring = deque()
    s.waiting = deque()
    s.swapped = deque()
    s._free_finished_seq_group = Mock()
    s.free_seq_cache("A")
    s.free_seq_cache("A")
    s._free_finished_seq_group.assert_called_once_with(a)
    assert list(s.running) == [b]
    for role in ["hybrid", "decode", "prefill"]:
        s._role = role
        s.free_finished_seq_groups()
    assert s.automatic == 2


def test_requery_miss_discards_old_load(connector, connector_module):
    previous = connector_module.XavierLoadRequest("r", {2: {111: 0}}, lease="1:old")
    connector._requests_need_load["r"] = previous
    connector._leased_requests["r"] = previous
    connector._release_load_request = AsyncMock()
    connector._query_remote_blocks = AsyncMock(return_value={})
    assert connector.get_num_new_matched_tokens(
        SimpleNamespace(request_id="r", prompt_token_ids=list(range(17))), 0
    ) == (0, False)
    connector._release_load_request.assert_awaited_once_with(previous)
    assert not connector._requests_need_load
    assert not connector._leased_requests


@pytest.mark.parametrize(
    "field", ["lora_request", "mm_features", "prompt_embeds", "cache_salt"]
)
def test_request_specific_cache_identity_is_rejected(connector, field):
    request = SimpleNamespace(request_id="r", prompt_token_ids=list(range(17)))
    setattr(request, field, [1])
    with pytest.raises(ValueError, match="does not support"):
        connector.get_num_new_matched_tokens(request, 0)


@pytest.mark.asyncio
async def test_snapshot_store_configured_once(connector, monkeypatch):
    import xoscar as xo

    ref = SimpleNamespace(configure_snapshots_v1=AsyncMock())
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=ref))
    assert await connector._get_transfer_ref() is ref
    assert await connector._get_transfer_ref() is ref
    ref.configure_snapshots_v1.assert_awaited_once_with(8)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "count,width,expected_calls",
    [(65, 1, 2), (130, 1, 3), (130, 8192, 5), (3, 262144, 3), (2, 262145, 2)],
)
async def test_batched_reads_bound_payload_and_preserve_order(
    connector_module, count, width, expected_calls
):
    calls = []

    async def read(rank, layer, mapping, shape, dtype):
        calls.append(mapping)
        assert shape == (len(mapping), width)
        assert len(mapping) <= 64
        assert len(mapping) * width * 4 <= 1024 * 1024 or len(mapping) == 1
        return (
            torch.tensor(list(mapping), dtype=dtype)
            .view(-1, 1)
            .expand(shape)
            .contiguous()
        )

    transfer = SimpleNamespace(read_layer_blocks_v1=read)
    connector = SimpleNamespace(_get_transfer_ref=AsyncMock(return_value=transfer))
    mapping = {2000 + i: 5000 - i for i in range(count)}
    result = await connector_module.XavierConnector._read_layer_blocks(
        connector, "layer", 1, mapping, (count, width), torch.float32
    )
    assert len(calls) == expected_calls
    assert [item for batch in calls for item in batch.items()] == list(mapping.items())
    assert result[:, 0].tolist() == list(mapping)


@pytest.mark.asyncio
async def test_later_read_batch_failure_propagates(connector_module):
    read = AsyncMock(side_effect=[torch.zeros(64, 1), KeyError("missing block")])
    connector = SimpleNamespace(
        _get_transfer_ref=AsyncMock(
            return_value=SimpleNamespace(read_layer_blocks_v1=read)
        )
    )
    with pytest.raises(KeyError, match="missing block"):
        await connector_module.XavierConnector._read_layer_blocks(
            connector, "layer", 1, {i: i for i in range(130)}, (130, 1), torch.float32
        )
    assert read.await_count == 2


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_bf16_snapshot_and_connector_preserve_all_bits(
    connector, connector_module, noncontiguous
):
    # Includes infinities, NaNs, subnormals and values above the FP16 range.
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    source = bits.view(torch.bfloat16).reshape(8, 8192)
    if noncontiguous:
        source = source.t().contiguous().t()
        assert not source.is_contiguous()
    actor = SimpleNamespace(_rank=0, _snapshot_store=KVSnapshotStore(8))
    TransferActor.stage_layer_blocks_v1(actor, "r", "layer", list(range(8)), source)
    payload = actor._snapshot_store.read("layer", list(range(8)))
    assert payload.dtype == torch.float16
    assert payload.element_size() == 2
    assert torch.equal(payload.view(torch.int16), bits.reshape(8, 8192))

    async def read(layer, rank, mapping, shape, dtype):
        assert dtype == torch.bfloat16
        # Exercise the NumPy transport representation, without numeric casts.
        result = torch.from_numpy(payload.numpy().copy())
        if noncontiguous:
            result = result.t().contiguous().t()
            assert not result.is_contiguous()
        return result

    connector._read_layer_blocks = read
    destination = torch.empty_like(source)
    request = connector_module.XavierLoadRequest(
        "r",
        {1: {i: i for i in range(8)}},
        local_transfers_by_group={0: {1: {i: i for i in range(8)}}},
    )
    connector._load_layer_blocks("layer", destination, request)
    assert torch.equal(destination.view(torch.int16), source.view(torch.int16))


def test_connector_rejects_unexpected_transport_dtype(connector, connector_module):
    async def read(*args):
        return torch.ones(1, 2, dtype=torch.float32)

    connector._read_layer_blocks = read
    destination = torch.zeros(8, 2, dtype=torch.bfloat16)
    request = connector_module.XavierLoadRequest(
        "r", {1: {1: 0}}, local_transfers_by_group={0: {1: {1: 0}}}
    )
    with pytest.raises(RuntimeError, match="torch.float32.*torch.bfloat16"):
        connector._load_layer_blocks("layer", destination, request)
    assert destination.count_nonzero() == 0


def test_snapshot_logical_dtype_tracks_immutable_blocks_and_eviction():
    store = KVSnapshotStore(1)
    carrier = torch.ones(1, 2, dtype=torch.float16)
    store.stage("K", [1], carrier, logical_dtype=torch.bfloat16)
    store.stage("K", [1], carrier, logical_dtype=torch.float16)
    assert store.logical_dtypes[1]["K"] == torch.bfloat16
    store.stage("K", [2], carrier)
    assert 1 not in store.logical_dtypes
    assert store.logical_dtypes == {2: {"K": torch.float16}}


@pytest.mark.asyncio
async def test_transfer_cleanup_continues_after_gpu_close_error():
    close = AsyncMock(side_effect=RuntimeError("sticky CUDA error"))
    task = Mock()
    context = Mock()
    actor = SimpleNamespace(
        close_gpu_caches_v1=close, _layer_send_tasks_v1={task}, _context=context
    )
    await TransferActor.__pre_destroy__(actor)
    task.cancel.assert_called_once_with()
    context.closeConnections.assert_called_once_with()


@pytest.mark.asyncio
async def test_gpu_connector_mapping_staging_and_load_fences(
    connector, connector_module, monkeypatch
):
    import torch.multiprocessing.reductions as reductions

    calls = []
    cache = torch.zeros(8, 2, 16, 2, 4)
    connector._gpu_budget = 64
    connector._gpu_cache_mapped = False
    connector._registered_kv_caches = {"layer": cache}
    transfer = SimpleNamespace(
        map_gpu_caches_v1=AsyncMock(side_effect=lambda *args: calls.append("map")),
        stage_gpu_requests_v1=AsyncMock(
            side_effect=lambda *args: calls.append("stage")
        ),
        load_gpu_request_v1=AsyncMock(side_effect=lambda *args: calls.append("load")),
        close_gpu_caches_v1=AsyncMock(),
    )
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(
        reductions, "reduce_tensor", lambda tensor: (None, ("descriptor",))
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("fence"))
    request = connector_module.XavierStoreRequest("r", [2, 5], [111, 222], [[2, 5]])
    await connector._stage_gpu_requests([request])
    assert calls == ["fence", "map", "fence", "stage"]
    transfer.stage_gpu_requests_v1.assert_awaited_once_with(
        [{"layer": ([111, 222], [2, 5])}]
    )
    assert not connector._request_staged_layers
    calls.clear()
    load = connector_module.XavierLoadRequest(
        "r", {1: {111: 0}}, local_transfers_by_group={0: {1: {111: 3}}}
    )
    await connector._load_gpu_request(load)
    assert calls == ["fence", "load", "fence"]
    transfer.load_gpu_request_v1.assert_awaited_once_with({1: {"layer": {111: 3}}})
    assert transfer.map_gpu_caches_v1.await_count == 1
    connector._get_connector_metadata = (
        lambda: connector_module.XavierConnectorMetadata(store_requests=[request])
    )
    connector._stage_kv_layer_for_request = Mock(
        side_effect=AssertionError("GPU mode stages once in wait_for_save")
    )
    connector.save_kv_layer("layer", cache, None)
    assert connector._pending_store_requests["r"] is request
