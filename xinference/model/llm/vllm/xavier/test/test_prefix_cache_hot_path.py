# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
import socket
import weakref
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
        stage_layer_batches_v1=AsyncMock(
            side_effect=lambda *args: TransferActor.stage_layer_batches_v1(actor, *args)
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
    transfer.stage_layer_batches_v1.assert_not_awaited()
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
    assert transfer.stage_layer_batches_v1.await_count == 1
    assert tracker.register_blocks.await_count == 2
    assert torch.equal(store.read("K", [111]), original)


def test_batch_dedup_preserves_positions_in_every_cache_group(
    connector, connector_module, snapshot_export
):
    store, transfer, tracker = snapshot_export
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
    transfer.stage_layer_batches_v1.assert_awaited_once()
    assert transfer.stage_layer_batches_v1.await_args.args[0][0][1] == [222, 333]
    tracker.register_blocks.assert_awaited_once_with(
        0, [(111, 111), (222, 222), (333, 333)], 1
    )
    assert first.block_ids == [0, 2, 4]
    assert first.block_ids_by_group == [[0, 2, 4], [1, 3, 5]]


def test_partial_export_failure_retries_unpublished_content(
    connector, connector_module, snapshot_export
):
    store, transfer, _ = snapshot_export
    request = connector_module.XavierStoreRequest("r", [2], [111], [[2], [3]])
    bind_store_requests(connector, connector_module, request)
    stage = store.stage

    def fail_v(*args, **kwargs):
        if args[0] == "V":
            raise RuntimeError("copy failed")
        return stage(*args, **kwargs)

    store.stage = fail_v
    with pytest.raises(RuntimeError, match="copy failed"):
        connector.wait_for_save()
    assert not store.ready
    assert set(store.blocks[111]) == {"K"}
    assert not connector._request_staged_layers
    store.stage = stage
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
    assert transfer.stage_layer_batches_v1.await_count == 1


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
    assert transfer.stage_layer_batches_v1.await_count == 3
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


def test_export_batches_bound_bytes_across_layers_and_requests(
    connector, connector_module, snapshot_export, monkeypatch
):
    store, transfer, tracker = snapshot_export
    # Each block has two float32 values in each of K and V: 16 bytes total.
    monkeypatch.setattr(connector_module, "_MAX_EXPORT_BYTES", 32)
    first = connector_module.XavierStoreRequest(
        "a", [0, 2, 4], [111, 222, 333], [[0, 2, 4], [1, 3, 5]]
    )
    second = connector_module.XavierStoreRequest("b", [6], [444], [[6], [7]])
    bind_store_requests(connector, connector_module, first, second)
    connector.wait_for_save()
    batches = [call.args[0] for call in transfer.stage_layer_batches_v1.await_args_list]
    assert len(batches) == 2
    assert [batch[0][1] for batch in batches] == [[111, 222], [333, 444]]
    assert all(
        sum(t.numel() * t.element_size() for _, _, t in batch) <= 32
        for batch in batches
    )
    assert store.read("K", [333, 444]).tolist() == [[8, 9], [12, 13]]
    assert store.read("V", [333, 444]).tolist() == [[26, 27], [30, 31]]
    transfer.publish_blocks_v1.assert_awaited_once_with(
        [111, 222, 333, 444], {"K", "V"}
    )
    tracker.register_blocks.assert_awaited_once_with(
        0, [(key, key) for key in [111, 222, 333, 444]], 1
    )


def test_oversized_export_block_keeps_all_layers_together(
    connector, connector_module, snapshot_export, monkeypatch
):
    store, transfer, _ = snapshot_export
    monkeypatch.setattr(connector_module, "_MAX_EXPORT_BYTES", 1)
    request = connector_module.XavierStoreRequest("r", [2], [111], [[2], [3]])
    bind_store_requests(connector, connector_module, request)
    connector.wait_for_save()
    transfer.stage_layer_batches_v1.assert_awaited_once()
    assert store.ready == {111}


def test_batch_larger_than_available_capacity_keeps_complete_blocks(
    connector, connector_module, snapshot_export
):
    store, _, tracker = snapshot_export
    store.capacity = 2
    store.stage("K", [999], torch.zeros(1, 2))
    store.stage("V", [999], torch.zeros(1, 2))
    store.publish([999], {"K", "V"})
    assert store.reserve("lease", [999])
    request = connector_module.XavierStoreRequest(
        "r", [2, 4], [111, 222], [[2, 4], [3, 5]]
    )
    bind_store_requests(connector, connector_module, request)
    connector.wait_for_save()
    assert store.ready == {999, 222}
    assert store.read("V", [222]).tolist() == [[26, 27]]
    tracker.unregister_blocks.assert_awaited_once_with(0, 1, [111])
    tracker.register_blocks.assert_awaited_once_with(0, [(222, 222)], 1)


def test_batched_export_preserves_noncontiguous_bf16_and_snapshot_ownership(
    connector, connector_module, snapshot_export
):
    store, _, _ = snapshot_export
    values = torch.arange(32, dtype=torch.int16).view(8, 4)
    tensor = values.view(torch.bfloat16)[:, ::2]
    connector._registered_kv_caches = {"K": tensor}
    request = connector_module.XavierStoreRequest("r", [2], [111], [[2]])
    bind_store_requests(connector, connector_module, request)
    connector.wait_for_save()
    assert store.logical_dtypes[111]["K"] == torch.bfloat16
    original = tensor[2].view(torch.int16).clone()
    tensor[2].zero_()
    assert torch.equal(store.read("K", [111]).view(torch.int16)[0], original)


@pytest.mark.asyncio
async def test_batched_registration_keeps_distinct_layer_requirements(
    connector, connector_module
):
    connector._registered_kv_caches = {}
    connector._request_staged_layers = {"a": {"K"}, "b": {"K", "V"}}
    transfer = SimpleNamespace(
        publish_blocks_v1=AsyncMock(side_effect=[([111], [222]), ([222], [])])
    )
    tracker = SimpleNamespace(
        register_blocks=AsyncMock(), unregister_blocks=AsyncMock()
    )
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    connector._get_tracker_ref = AsyncMock(return_value=tracker)
    request_type = connector_module.XavierStoreRequest
    await connector._register_blocks(
        [
            request_type("a", [0], [111], [[0]]),
            request_type("b", [1], [222], [[1]]),
        ]
    )
    assert [call.args for call in transfer.publish_blocks_v1.await_args_list] == [
        ([111], {"K"}),
        ([222], {"K", "V"}),
    ]
    tracker.unregister_blocks.assert_awaited_once_with(0, 1, [222])
    tracker.register_blocks.assert_awaited_once_with(0, [(111, 111), (222, 222)], 1)


@pytest.mark.asyncio
async def test_export_failure_drains_next_copy_without_publishing(
    connector, snapshot_export, monkeypatch
):
    _, transfer, tracker = snapshot_export
    events = [Mock(), Mock()]
    batches = [([], [torch.ones(1)], [event]) for event in events]
    monkeypatch.setattr(connector, "_cpu_export_batches", lambda _: iter(batches))
    transfer.stage_layer_batches_v1.side_effect = RuntimeError("actor failed")
    with pytest.raises(RuntimeError, match="actor failed"):
        await connector._stage_cpu_requests([Mock()])
    for event in events:
        event.synchronize.assert_called_once()
    transfer.publish_blocks_v1.assert_not_awaited()
    tracker.register_blocks.assert_not_awaited()


@pytest.mark.asyncio
async def test_export_releases_previous_buffers_before_preparing_next_batch(
    connector, snapshot_export, monkeypatch
):
    owners = []

    def batches(_):
        for i in range(4):
            # Only the preceding batch may still own GPU storage at this point.
            assert sum(ref() is not None for ref in owners) <= 1
            owner = torch.ones(1)
            owners.append(weakref.ref(owner))
            yield [("K", [i], torch.ones(1))], [owner], []
            del owner

    monkeypatch.setattr(connector, "_cpu_export_batches", batches)
    await connector._stage_cpu_requests([Mock()])
    assert all(ref() is None for ref in owners)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_export_fences_current_stream_and_owns_snapshot(
    connector, connector_module, snapshot_export, monkeypatch, dtype
):
    store, transfer, _ = snapshot_export
    connector._registered_kv_caches = {
        name: tensor.to(device="cuda", dtype=dtype)
        for name, tensor in connector._registered_kv_caches.items()
    }
    # Force two batches so the second copy is in flight during the first RPC.
    monkeypatch.setattr(connector_module, "_MAX_EXPORT_BYTES", 8)
    request = connector_module.XavierStoreRequest(
        "r", [2, 4], [111, 222], [[2, 4], [3, 5]]
    )
    bind_store_requests(connector, connector_module, request)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        connector._registered_kv_caches["K"][2].fill_(42)
        connector._registered_kv_caches["V"][5].fill_(84)
        connector.wait_for_save()
        for tensor in connector._registered_kv_caches.values():
            tensor.zero_()
    stream.synchronize()
    assert store.ready == {111, 222}
    assert store.read("K", [111, 222]).view(dtype).tolist() == [[42, 42], [8, 9]]
    assert store.read("V", [111, 222]).view(dtype).tolist() == [[22, 23], [84, 84]]
    assert transfer.stage_layer_batches_v1.await_count == 2
    assert all(
        t.is_pinned()
        for call in transfer.stage_layer_batches_v1.await_args_list
        for _, _, t in call.args[0]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
async def test_batch_export_socket_backpressure_preserves_dtype_and_bytes(
    connector, connector_module, snapshot_export, dtype
):
    from xoscar.backends.communication.socket import SocketChannel
    from xoscar.serialization.aio import AioSerializer

    store, transfer, _ = snapshot_export
    source = torch.arange(8 * 32768).view(8, 32768).to(dtype)
    connector._registered_kv_caches = {f"layer{i}": source for i in range(4)}
    actor = SimpleNamespace(_snapshot_store=store, _rank=1)
    received = asyncio.get_running_loop().create_future()

    async def receive(reader, writer):
        try:
            # Force the sender to queue tensor buffers, rather than writing
            # the entire message immediately into the socket's send buffer.
            await asyncio.sleep(0.01)
            channel = SocketChannel(reader, writer)
            entries, metadata = await channel.recv()
            TransferActor.stage_layer_batches_v1(actor, entries, metadata)
            await channel.send(True)
            received.set_result(None)
        except Exception as exc:
            received.set_exception(exc)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(receive, "127.0.0.1", 0)
    reader, writer = await asyncio.open_connection(*server.sockets[0].getsockname())
    writer.get_extra_info("socket").setsockopt(
        socket.SOL_SOCKET, socket.SO_SNDBUF, 4096
    )
    channel = SocketChannel(reader, writer)

    async def send(entries, metadata):
        buffers = await AioSerializer((entries, metadata)).run()
        assert all(len(buf) == getattr(buf, "nbytes", len(buf)) for buf in buffers)
        await channel.send((entries, metadata))
        assert await channel.recv() is True

    transfer.stage_layer_batches_v1.side_effect = send
    request = connector_module.XavierStoreRequest("r", [2, 4], [111, 222], [[2, 4]])
    try:
        await asyncio.wait_for(connector._stage_cpu_requests([request]), timeout=20)
        await asyncio.wait_for(received, timeout=20)
        for name in connector._registered_kv_caches:
            actual = store.read(name, [111, 222]).view(dtype)
            assert torch.equal(actual, source[[2, 4]])
            assert store.logical_dtypes[111][name] == dtype
    finally:
        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()
