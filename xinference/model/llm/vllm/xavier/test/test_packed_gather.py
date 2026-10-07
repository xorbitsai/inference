# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch


def test_fused_export_queues_owned_payload_without_repacking(
    connector, connector_module, monkeypatch
):
    from torch.multiprocessing import reductions

    packed = torch.arange(32, dtype=torch.uint8).view(2, 16)
    layers = [("K", (4,), torch.float32, 0, 16)]

    class Gather:
        device = torch.device("cpu")

        def __call__(self, sources):
            assert sources == {"K": [2, 4]}
            return packed, []

    gather = Gather()
    gather.layers = layers
    connector._packed_gather = gather
    transfer = SimpleNamespace(enqueue_snapshot_export_v1=AsyncMock())
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    monkeypatch.setattr(reductions, "reduce_tensor", lambda value: (None, (value,)))
    monkeypatch.setattr(torch, "cat", lambda *a, **kw: pytest.fail("payload repacked"))
    batch = connector._prepare_cpu_export({}, [111, 222], {"K": [2, 4]}, gpu_only=True)
    connector._call(connector._queue_ipc_export([batch], 32))
    ticket, descriptors, _, _ = transfer.enqueue_snapshot_export_v1.await_args.args
    keys, (payload,), schema = descriptors[0]
    assert keys == [111, 222] and payload is packed and schema == layers
    assert connector._ipc_export_jobs[ticket][2] == [packed]
    connector._ipc_export_jobs.clear()


@pytest.mark.parametrize("outcome", ["recovery", "lost-ack", "failed-copy"])
def test_arena_slots_survive_recovery_and_release_only_after_completion(
    connector, connector_module, outcome
):
    packed = torch.zeros(2, 16, dtype=torch.uint8)
    arena = SimpleNamespace(
        source="source", metadata=Mock(return_value=["fresh"]), release=Mock()
    )
    connector._gpu_export_arena = arena
    connector._gpu_export_registered = outcome == "recovery"
    connector._exported_keys = {999}
    connector._export_refresh_at = 100
    transfer = SimpleNamespace(
        register_snapshot_export_source_v1=AsyncMock(),
        enqueue_snapshot_export_v1=AsyncMock(),
        poll_snapshot_exports_v1=AsyncMock(return_value={}),
    )
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    entries = connector_module._PackedExportEntries(
        [111, 222], packed, [("K", (4,), torch.float32, 0, 16)], 0
    )
    if outcome == "recovery":
        transfer.enqueue_snapshot_export_v1.side_effect = [False, None]
    elif outcome == "lost-ack":
        transfer.enqueue_snapshot_export_v1.side_effect = RuntimeError("lost ack")
    try:
        if outcome == "lost-ack":
            with pytest.raises(RuntimeError, match="lost ack"):
                connector._call(connector._queue_ipc_export([(entries, [], [])], 32))
            assert not connector._gpu_export_registered
        else:
            connector._call(connector._queue_ipc_export([(entries, [], [])], 32))
        ticket = next(iter(connector._ipc_export_jobs))
        assert connector._gpu_export_tickets[ticket] == [0]
        connector._poll_ipc_exports(wait=True)
        arena.release.assert_not_called()
        assert ticket in connector._ipc_export_jobs
        transfer.register_snapshot_export_source_v1.assert_awaited_once_with(
            "source", ["fresh"]
        )
        if outcome == "recovery":
            assert not connector._exported_keys and connector._export_refresh_at == 0
        if outcome == "failed-copy":
            transfer.poll_snapshot_exports_v1.return_value = {
                ticket: (None, "copy failed")
            }
            connector._poll_ipc_exports(wait=True)
        else:
            transfer.poll_snapshot_exports_v1.return_value = {ticket: ([111, 222], [])}
            connector._poll_ipc_exports(wait=True)
        arena.release.assert_called_once_with([0])
        assert not connector._ipc_export_jobs and not connector._gpu_export_tickets
    finally:
        connector._gpu_export_arena = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.uint8])
def test_fused_gather_preserves_mixed_group_bits_and_stream_order(dtype):
    from ....xavier.backends.torch.packed_gather import PackedGather

    base = torch.arange(2 * 8 * 16 * 2 * 64, device="cuda", dtype=torch.float32)
    k = base.remainder(197).to(dtype).view(2, 8, 16, 2, 64).permute(1, 0, 2, 3, 4)
    v = torch.randn(8, 1, 8, 4, 64, device="cuda", dtype=torch.float32)
    caches = {"K": k, "V": v}
    sources = {"K": [5, 0, 6, 2], "V": [1, 7, 3, 0]}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gather = PackedGather.try_create(caches, {"K": 0, "V": 1})
        assert gather is not None
        gather.warmup()
        packed, owners = gather(sources)
        expected = torch.cat(
            [
                cache.index_select(0, torch.tensor(sources[name], device="cuda"))
                .view(torch.uint8)
                .reshape(4, -1)
                for name, cache in caches.items()
            ],
            dim=1,
        )
        # A later model step can overwrite every original source slot.
        k.zero_()
        v.zero_()
    stream.synchronize()
    assert torch.equal(packed, expected)
    assert owners and packed.dtype == torch.uint8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unsupported_geometry_uses_per_layer_fallback():
    from ....xavier.backends.torch.packed_gather import PackedGather

    assert (
        PackedGather.try_create(
            {"K": torch.ones(8, 3, device="cuda", dtype=torch.uint8)}, {"K": 0}
        )
        is None
    )
    assert (
        PackedGather.try_create(
            {"K": torch.ones(8, 16, device="cuda")[:, ::2]}, {"K": 0}
        )
        is None
    )
    assert PackedGather.try_create({"K": torch.ones(8, 16)}, {"K": 0}) is None
