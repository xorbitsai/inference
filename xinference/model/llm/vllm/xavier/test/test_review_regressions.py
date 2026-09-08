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
    connector._load_layer_blocks = Mock(side_effect=RuntimeError("failed"))
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
    actor = SimpleNamespace(_world_addresses=["zero", "one", "two"])
    assert not await TransferActor.reserve_remote_blocks_v1(
        actor, "3:r", {1: {111: 0}, 2: {222: 1}}
    )
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
