# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
import multiprocessing
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from ..request_transfer import LayerRead
from ..transfer import TransferActor

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux tmpfs required")


def _copy_in_child(metadata, lease, queue):
    from ....xavier.backends.torch.local_read import LocalReadBuffer

    view = LocalReadBuffer.attach(metadata)
    queue.put(view.copy(lease, 8, pin_memory=False).tolist())
    view.close()


def test_shared_receive_is_leased_and_copied_before_slot_reuse():
    from ....xavier.backends.torch.local_read import LocalReadBuffer, SharedReadBuffer

    owner = SharedReadBuffer(64)
    view = LocalReadBuffer.attach(owner.metadata())
    try:
        lease, payload = owner.acquire(8)
        payload.copy_(torch.arange(8, dtype=torch.uint8))
        expected = view.copy(lease, 8, pin_memory=False)
        assert owner.acquire(8) is None
        owner.release(b"wrong")
        assert owner.acquire(8) is None
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        child = ctx.Process(
            target=_copy_in_child, args=(owner.metadata(), lease, queue)
        )
        child.start()
        assert queue.get(timeout=30) == list(range(8))
        child.join(30)
        assert child.exitcode == 0
        owner.release(lease)
        next_lease, replacement = owner.acquire(8)
        replacement.fill_(99)
        assert expected.tolist() == list(range(8))
        assert view.copy(next_lease, 8, pin_memory=False).tolist() == [99] * 8
        with pytest.raises(RuntimeError, match="lease"):
            view.copy(lease, 8, pin_memory=False)
        owner.release(next_lease)
        assert owner.acquire(65) is None
        assert owner.acquire(0) is None
        del payload, replacement
        owner.close()
        assert not view.valid()
    finally:
        view.close()
        owner.close()
    view.close()


@pytest.mark.parametrize("changed", ["boot_id", "token", "path", "capacity"])
def test_foreign_stale_or_invalid_receive_metadata_falls_back(changed):
    from ....xavier.backends.torch.local_read import LocalReadBuffer, SharedReadBuffer

    owner = SharedReadBuffer(64)
    try:
        metadata = owner.metadata()
        metadata[changed] = {
            "boot_id": "foreign",
            "token": bytes(16),
            "path": "/proc/0/fd/0",
            "capacity": 65,
        }[changed]
        assert LocalReadBuffer.attach(metadata) is None
    finally:
        owner.close()


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.asyncio
async def test_receive_failure_or_cancel_fences_native_write_before_release(cancel):
    from ....xavier.backends.torch.local_read import SharedReadBuffer

    owner = SharedReadBuffer(64)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def receive(*_, _buffer):
        entered.set()
        await finish.wait()
        _buffer.fill_(99)
        if not cancel:
            raise RuntimeError("receive failed")

    actor = SimpleNamespace(
        _local_read_buffer_v1=owner,
        _local_read_tasks_v1=set(),
        read_request_blocks_v1=receive,
    )
    reads = [LayerRead("K", [1], [2], (8,), torch.uint8)]
    task = asyncio.create_task(
        TransferActor.read_request_blocks_local_v1(actor, 0, reads, owner.token)
    )
    try:
        await entered.wait()
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        assert owner.acquire(8) is None and not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await task
        assert not actor._local_read_tasks_v1
        acquired = owner.acquire(8)
        assert acquired is not None and acquired[1].tolist() == [99] * 8
        owner.release(acquired[0])
        del acquired
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
        owner.close()


@pytest.mark.parametrize("copy_failure", [False, True])
def test_connector_owns_receive_bytes_and_releases_even_on_copy_failure(
    connector, monkeypatch, copy_failure
):
    from ....xavier.backends.torch.local_read import LocalReadBuffer, SharedReadBuffer

    owner = SharedReadBuffer(64)
    actor = SimpleNamespace(_local_read_buffer_v1=owner)

    async def read_local(*_):
        acquired = owner.acquire(8)
        acquired[1].fill_(7)
        return acquired[0]

    transfer = SimpleNamespace(
        get_local_read_metadata_v1=AsyncMock(return_value=owner.metadata()),
        read_request_blocks_local_v1=AsyncMock(side_effect=read_local),
        release_local_read_v1=AsyncMock(
            side_effect=MethodType(TransferActor.release_local_read_v1, actor)
        ),
    )
    reads = [LayerRead("K", [1], [2], (8,), torch.uint8)]
    try:
        if copy_failure:
            monkeypatch.setattr(
                LocalReadBuffer,
                "copy",
                lambda *_, **__: (_ for _ in ()).throw(RuntimeError("copy failed")),
            )
            with pytest.raises(RuntimeError, match="copy failed"):
                connector._call(
                    connector._read_request_payload(
                        transfer, 0, reads, pin_memory=False
                    )
                )
        else:
            copied = connector._call(
                connector._read_request_payload(transfer, 0, reads, pin_memory=False)
            )
            assert copied.tolist() == [7] * 8
        transfer.release_local_read_v1.assert_awaited_once()
        lease, payload = owner.acquire(8)
        payload.fill_(99)
        if not copy_failure:
            assert copied.tolist() == [7] * 8
        owner.release(lease)
        del payload
    finally:
        connector.shutdown()
        owner.close()


def test_shared_receive_unavailable_uses_normal_tensor_rpc(connector, monkeypatch):
    from ....xavier.backends.torch import local_read

    monkeypatch.setattr(
        local_read,
        "SharedReadBuffer",
        lambda: (_ for _ in ()).throw(OSError("no shared memory")),
    )
    assert TransferActor.get_local_read_metadata_v1(SimpleNamespace()) is None
    expected = torch.arange(8, dtype=torch.uint8)
    transfer = SimpleNamespace(
        get_local_read_metadata_v1=AsyncMock(return_value=None),
        read_request_blocks_v1=AsyncMock(return_value=expected),
    )
    result = connector._call(
        connector._read_request_payload(transfer, 0, [], pin_memory=False)
    )
    assert result is expected
    transfer.read_request_blocks_v1.assert_awaited_once_with(0, [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pinned_receive_retains_exact_bits_after_shared_slot_is_reused(dtype):
    from ....xavier.backends.torch.local_read import LocalReadBuffer, SharedReadBuffer

    owner = SharedReadBuffer(64)
    view = LocalReadBuffer.attach(owner.metadata())
    values = torch.tensor([1e30, -1e30, 3, -4]).to(dtype)
    bits = values.view(torch.uint8)
    try:
        lease, payload = owner.acquire(bits.numel())
        payload.copy_(bits)
        copied = view.copy(lease, bits.numel(), pin_memory=True)
        assert copied.is_pinned()
        owner.release(lease)
        new_lease, replacement = owner.acquire(bits.numel())
        replacement.zero_()
        result = copied.to("cuda", non_blocking=True).cpu()
        assert torch.equal(result, bits)
        owner.release(new_lease)
        del payload, replacement
    finally:
        view.close()
        owner.close()
