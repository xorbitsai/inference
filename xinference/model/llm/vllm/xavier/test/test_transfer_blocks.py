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
