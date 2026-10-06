# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from ..snapshot import KVSnapshotStore
from ..tiered_snapshot import TieredKVSnapshotStore
from ..transfer import TransferActor


@pytest.fixture
def snapshot_export(connector):
    store = KVSnapshotStore(8)
    actor = SimpleNamespace(_snapshot_store=store, _rank=1)
    transfer = SimpleNamespace(
        ready_blocks_for_export_v1=AsyncMock(
            side_effect=lambda keys: TransferActor.ready_blocks_for_export_v1(
                actor, keys
            )
        ),
        stage_layer_blocks_v1=AsyncMock(
            side_effect=lambda *args: TransferActor.stage_layer_blocks_v1(actor, *args)
        ),
        publish_blocks_v1=AsyncMock(
            side_effect=lambda *args: TransferActor.publish_blocks_v1(actor, *args)
        ),
    )
    tracker = SimpleNamespace(
        register_blocks=AsyncMock(), unregister_blocks=AsyncMock()
    )
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    connector._get_tracker_ref = AsyncMock(return_value=tracker)
    connector._registered_kv_caches = {
        "K": torch.arange(16).view(8, 2).float(),
        "V": torch.arange(16, 32).view(8, 2).float(),
    }
    connector._layer_group_ids = {"K": 0, "V": 1}
    return store, transfer, tracker


def bind_store_requests(connector, module, *requests):
    metadata = module.XavierConnectorMetadata(store_requests=list(requests))
    connector._get_connector_metadata = lambda: metadata
    return metadata


@pytest.mark.parametrize("remaining", [0, 1, 15, 16])
def test_local_prefix_tail_skips_lookup(connector, remaining):
    connector._query_remote_blocks = AsyncMock(return_value={})
    connector._build_xavier_hashes = Mock(return_value=[(1, 0), (2, 1)])
    request = SimpleNamespace(
        request_id="r", prompt_token_ids=list(range(16 + remaining + 1))
    )
    assert connector.get_num_new_matched_tokens(request, 16) == (0, False)
    assert connector._query_remote_blocks.await_count == (remaining == 16)
    assert connector._build_xavier_hashes.call_count == (remaining == 16)


def test_skipped_tail_releases_previous_lease(connector, connector_module):
    load = connector_module.XavierLoadRequest("r", {2: {111: 0}}, lease="1:old")
    connector._leased_requests["r"] = load
    connector._requests_need_load["r"] = load
    connector._release_load_request = AsyncMock()
    connector._query_remote_blocks = AsyncMock()
    request = SimpleNamespace(request_id="r", prompt_token_ids=list(range(18)))
    assert connector.get_num_new_matched_tokens(request, 16) == (0, False)
    connector._release_load_request.assert_awaited_once_with(load)
    connector._query_remote_blocks.assert_not_awaited()
    assert not connector._requests_need_load and not connector._leased_requests


def test_export_uses_post_forward_data_and_current_metadata(
    connector, connector_module, snapshot_export
):
    store, transfer, tracker = snapshot_export
    capture = connector_module.XavierStoreRequest("capture", [0], [111], [[0], [1]])
    metadata = bind_store_requests(connector, connector_module, capture)
    for name, cache in connector._registered_kv_caches.items():
        connector.save_kv_layer(name, cache, None)
    transfer.stage_layer_blocks_v1.assert_not_awaited()
    transfer.ready_blocks_for_export_v1.assert_not_awaited()
    current = connector_module.XavierStoreRequest("replay", [2], [222], [[2], [3]])
    metadata.store_requests = [current]
    connector._registered_kv_caches["K"][2].fill_(42)
    connector._registered_kv_caches["V"][3].fill_(84)
    # Full CUDA graph replay need not invoke any per-layer Python callback.
    connector.wait_for_save()
    assert store.ready == {222}
    assert store.read("K", [222]).tolist() == [[42, 42]]
    assert store.read("V", [222]).tolist() == [[84, 84]]
    tracker.register_blocks.assert_awaited_once_with(0, [(222, 222)], 1)
    assert not connector._request_staged_layers


def test_repeated_export_skips_payload_but_refreshes_discovery(
    connector, connector_module, snapshot_export
):
    store, transfer, tracker = snapshot_export
    request = connector_module.XavierStoreRequest("r", [2], [111], [[2], [3]])
    bind_store_requests(connector, connector_module, request)
    connector.wait_for_save()
    original = store.read("K", [111]).clone()
    connector._registered_kv_caches["K"][2].fill_(99)
    connector.wait_for_save()
    assert transfer.stage_layer_blocks_v1.await_count == 2
    assert tracker.register_blocks.await_count == 2
    assert torch.equal(store.read("K", [111]), original)


def test_batch_dedup_preserves_positions_in_every_cache_group(
    connector, connector_module, snapshot_export
):
    store, transfer, _ = snapshot_export
    store.stage("K", [111], torch.ones(1, 2))
    store.stage("V", [111], torch.ones(1, 2))
    store.publish([111], {"K", "V"})
    first = connector_module.XavierStoreRequest(
        "a", [0, 2, 4], [111, 222, 333], [[0, 2, 4], [1, 3, 5]]
    )
    second = connector_module.XavierStoreRequest("b", [6], [222], [[6], [7]])
    bind_store_requests(connector, connector_module, first, second)
    connector.wait_for_save()
    assert store.ready == {111, 222, 333}
    assert store.read("K", [222, 333]).tolist() == [[4, 5], [8, 9]]
    assert store.read("V", [222, 333]).tolist() == [[22, 23], [26, 27]]
    assert transfer.stage_layer_blocks_v1.await_count == 2
    assert transfer.stage_layer_blocks_v1.await_args_list[0].args[2] == [222, 333]
    assert first.block_ids == [0, 2, 4]
    assert first.block_ids_by_group == [[0, 2, 4], [1, 3, 5]]


def test_partial_export_failure_retries_unpublished_content(
    connector, connector_module, snapshot_export
):
    store, transfer, _ = snapshot_export
    request = connector_module.XavierStoreRequest("r", [2], [111], [[2], [3]])
    bind_store_requests(connector, connector_module, request)
    stage = transfer.stage_layer_blocks_v1.side_effect

    def fail_v(*args):
        if args[1] == "V":
            raise RuntimeError("copy failed")
        return stage(*args)

    transfer.stage_layer_blocks_v1.side_effect = fail_v
    with pytest.raises(RuntimeError, match="copy failed"):
        connector.wait_for_save()
    assert not store.ready
    assert set(store.blocks[111]) == {"K"}
    assert not connector._request_staged_layers
    transfer.stage_layer_blocks_v1.side_effect = stage
    connector.wait_for_save()
    assert store.ready == {111}
    assert store.read("V", [111]).tolist() == [[22, 23]]


def test_tracker_failure_retries_without_exporting_payload(
    connector, connector_module, snapshot_export
):
    store, transfer, tracker = snapshot_export
    request = connector_module.XavierStoreRequest("r", [2], [111], [[2], [3]])
    bind_store_requests(connector, connector_module, request)
    tracker.register_blocks.side_effect = RuntimeError("tracker unavailable")
    with pytest.raises(RuntimeError, match="tracker unavailable"):
        connector.wait_for_save()
    assert store.ready == {111}
    tracker.register_blocks.side_effect = None
    connector.wait_for_save()
    assert tracker.register_blocks.await_count == 2
    assert transfer.stage_layer_blocks_v1.await_count == 2


def test_evicted_content_is_exported_again(
    connector, connector_module, snapshot_export
):
    store, transfer, tracker = snapshot_export
    store.capacity = 1
    first = connector_module.XavierStoreRequest("a", [2], [111], [[2], [3]])
    second = connector_module.XavierStoreRequest("b", [4], [222], [[4], [5]])
    for request in [first, second, first]:
        bind_store_requests(connector, connector_module, request)
        connector.wait_for_save()
        assert store.ready == set(request.block_hashes)
    assert transfer.stage_layer_blocks_v1.await_count == 6
    assert tracker.unregister_blocks.await_count == 2
    assert store.read("K", [111]).tolist() == [[4, 5]]


@pytest.mark.parametrize("tiered", [False, True])
def test_ready_query_excludes_partial_blocks_and_refreshes_lru(tiered):
    store = (
        TieredKVSnapshotStore(2, 0, 16, torch.device("cpu"))
        if tiered
        else KVSnapshotStore(2)
    )
    actor = SimpleNamespace(_snapshot_store=store)
    for key in [111, 222]:
        store.stage("K", [key], torch.ones(1, 2))
        store.publish([key], {"K"})
    assert TransferActor.ready_blocks_for_export_v1(actor, [111, 111, 333]) == [111]
    store.stage("K", [333], torch.ones(1, 2))
    assert store.ready == {111}
    assert TransferActor.ready_blocks_for_export_v1(actor, [111, 222, 333]) == [111]
    store.publish([333], {"K"})
    assert TransferActor.ready_blocks_for_export_v1(actor, [333]) == [333]


def test_unconfigured_store_has_no_ready_blocks():
    actor = SimpleNamespace(_snapshot_store=None)
    assert TransferActor.ready_blocks_for_export_v1(actor, [111]) == []


@pytest.mark.asyncio
async def test_empty_gpu_export_avoids_mapping_and_fence(connector, monkeypatch):
    connector._ensure_gpu_cache_mapping = AsyncMock()
    fence = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", fence)
    await connector._stage_gpu_requests([])
    connector._ensure_gpu_cache_mapping.assert_not_awaited()
    fence.assert_not_called()
