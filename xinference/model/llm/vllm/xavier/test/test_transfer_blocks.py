# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ..transfer import TransferActor


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2, 5])
async def test_receive_all_chunks(monkeypatch, count):
    from xoscar.collective import xoscar_pygloo as xp

    buffer = torch.empty(2, 2, 2, 1)
    received = []

    def get_buffer(index, size):
        view = buffer.flatten()[: 4 * size].view(2, 2, size, 1)
        received.append(view)
        return view

    def recv(*args):
        received[-1].fill_(len(received))

    monkeypatch.setattr(xp, "recv", recv)
    actor = SimpleNamespace(
        _get_swap_block_ids=lambda mapping, is_sender: list(mapping.values()),
        get_buffer_index=Mock(return_value=7),
        free_buffer_index=Mock(),
        get_swap_buffer=get_buffer,
        get_gloo_dtype=lambda dtype: dtype,
        transfer_block_num=2,
        _context=None,
    )
    result, ids, index = await TransferActor.read_blocks(
        actor, 1, {i: i + 10 for i in range(count)}
    )
    assert ids == list(range(10, 10 + count))
    assert index == 7
    expected = torch.tensor([i // 2 + 1 for i in range(count)])
    assert torch.equal(result, expected.view(1, 1, count, 1).expand(2, 2, count, 1))
    actor.free_buffer_index.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [True, False])
async def test_receive_failure_releases_buffer(monkeypatch, empty):
    from xoscar.collective import xoscar_pygloo as xp

    monkeypatch.setattr(
        xp, "recv", Mock(side_effect=[None, RuntimeError("receive failed")])
    )
    actor = SimpleNamespace(
        _get_swap_block_ids=lambda mapping, is_sender: list(mapping.values()),
        get_buffer_index=Mock(return_value=7),
        free_buffer_index=Mock(),
        get_swap_buffer=lambda index, size: torch.empty(1, 2, size, 1),
        get_gloo_dtype=lambda dtype: dtype,
        transfer_block_num=2,
        _context=None,
    )
    with pytest.raises(ValueError if empty else RuntimeError):
        await TransferActor.read_blocks(
            actor, 1, {} if empty else {0: 10, 1: 11, 2: 12}
        )
    if empty:
        actor.get_buffer_index.assert_not_called()
    else:
        actor.free_buffer_index.assert_called_once_with(7)


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
async def test_v1_multiblock_actor_transfer_preserves_order(monkeypatch, dtype):
    import asyncio
    import ctypes
    import queue
    from types import MethodType
    from unittest.mock import AsyncMock

    import xoscar as xo
    from xoscar.collective import xoscar_pygloo as xp

    from ..snapshot import KVSnapshotStore

    wire = queue.Queue()
    sizes = []

    def send(context, pointer, count, dtype, rank):
        sizes.append(count)
        wire.put(ctypes.string_at(pointer, count * dtype.itemsize))

    def recv(context, pointer, count, dtype, rank):
        data = wire.get(timeout=5)
        assert len(data) == count * dtype.itemsize
        ctypes.memmove(pointer, data, len(data))

    monkeypatch.setattr(xp, "send", send)
    monkeypatch.setattr(xp, "recv", recv)
    sender = SimpleNamespace(
        _rank=0,
        _context=None,
        _snapshot_store=KVSnapshotStore(8),
        _layer_send_tasks_v1=set(),
        get_gloo_dtype=lambda dtype: dtype,
    )
    for name in (
        "_get_staged_layer_blocks_v1",
        "do_send_layer_blocks_v1",
        "start_send_layer_blocks_v1",
        "has_layer_blocks_v1",
    ):
        setattr(sender, name, MethodType(getattr(TransferActor, name), sender))
    blocks = torch.arange(12, dtype=dtype).reshape(3, 2, 2)
    TransferActor.stage_layer_blocks_v1(sender, "r", "layer", [11, 22, 33], blocks)
    TransferActor.publish_blocks_v1(sender, [11, 22, 33], ["layer"])
    ref = SimpleNamespace(
        has_layer_blocks_v1=AsyncMock(side_effect=sender.has_layer_blocks_v1),
        start_send_layer_blocks_v1=AsyncMock(
            side_effect=sender.start_send_layer_blocks_v1
        ),
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=ref))
    receiver = SimpleNamespace(
        _rank=1,
        _context=None,
        _world_addresses=["sender", "receiver"],
        get_gloo_dtype=lambda dtype: dtype,
    )
    receiver.do_recv_layer_blocks_v1 = MethodType(
        TransferActor.do_recv_layer_blocks_v1, receiver
    )
    result = await TransferActor.read_layer_blocks_v1(
        receiver, 0, "layer", {33: 7, 11: 2, 22: 5}, (3, 2, 2), dtype
    )
    await asyncio.gather(*sender._layer_send_tasks_v1)
    assert sizes == [12]
    assert result.shape == (3, 2, 2)
    if dtype == torch.bfloat16:
        assert result.dtype == torch.float16
        result = result.view(torch.bfloat16)
    assert torch.equal(result, blocks[[2, 0, 1]])
    ref.has_layer_blocks_v1.assert_awaited_once_with("layer", [33, 11, 22], dtype)
    ref.start_send_layer_blocks_v1.assert_awaited_once_with(1, "layer", [33, 11, 22])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_dtype, destination_dtype",
    [(torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16)],
)
async def test_mismatched_logical_dtype_rejected_before_send(
    monkeypatch, source_dtype, destination_dtype
):
    from unittest.mock import AsyncMock

    import xoscar as xo

    from ..snapshot import KVSnapshotStore

    sender = SimpleNamespace(_rank=0, _snapshot_store=KVSnapshotStore(1))
    TransferActor.stage_layer_blocks_v1(
        sender, "r", "layer", [1], torch.ones(1, 2, dtype=source_dtype)
    )
    TransferActor.publish_blocks_v1(sender, [1], ["layer"])
    ref = SimpleNamespace(
        has_layer_blocks_v1=AsyncMock(
            side_effect=lambda *args: TransferActor.has_layer_blocks_v1(sender, *args)
        ),
        start_send_layer_blocks_v1=AsyncMock(),
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=ref))
    receiver = SimpleNamespace(_rank=1, _world_addresses=["sender"])
    with pytest.raises(KeyError, match="No staged Xavier"):
        await TransferActor.read_layer_blocks_v1(
            receiver, 0, "layer", {1: 0}, (1, 2), destination_dtype
        )
    ref.start_send_layer_blocks_v1.assert_not_awaited()
