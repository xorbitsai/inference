# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ..native import NativeSourceReleaser


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_fails_first", [False, True])
async def test_native_release_waits_for_cancel_and_unpin_without_receiver(
    cancel_fails_first,
):
    source = SimpleNamespace(
        py_kv_transfer_timed_out=False, is_disagg_context_complete_state=False
    )
    other = SimpleNamespace(py_kv_transfer_timed_out=False)
    executor = SimpleNamespace(
        ctx_in_transmission_requests={123: (source, 7, 1), 124: (other, 8, 1)},
        cancel_request=Mock(),
        _check_disagg_ctx_cache_transfer_status=Mock(),
    )
    unpin = Mock()
    polls = []

    def terminate():
        if source.py_kv_transfer_timed_out:
            polls.append(123)
            if cancel_fails_first and len(polls) == 1:
                return  # Native cancellation is still draining its GPU reads.
            executor.ctx_in_transmission_requests.pop(123)
            unpin(7)

    executor._terminate_disagg_ctx_finished_requests = terminate
    check_status = executor._check_disagg_ctx_cache_transfer_status
    native = NativeSourceReleaser(executor)
    tasks = [
        asyncio.create_task(asyncio.to_thread(native.release, 123)) for _ in range(2)
    ]
    await asyncio.sleep(0.02)
    assert all(not t.done() for t in tasks)
    executor._terminate_disagg_ctx_finished_requests()
    if cancel_fails_first:
        assert all(not t.done() for t in tasks)
        unpin.assert_not_called()
        executor._terminate_disagg_ctx_finished_requests()
    await asyncio.wait_for(asyncio.gather(*tasks), 2)
    unpin.assert_called_once_with(7)
    assert executor.ctx_in_transmission_requests == {124: (other, 8, 1)}
    assert not other.py_kv_transfer_timed_out
    assert executor.cancel_request.call_args_list
    assert all(c.args == (123,) for c in executor.cancel_request.call_args_list)
    native.release(123)  # Already completed: no extra cancel/unpin.
    unpin.assert_called_once()
    assert native._cancelled[123] is source
    executor._check_disagg_ctx_cache_transfer_status(1)
    check_status.assert_called_once_with(0)
    assert executor.ctx_in_transmission_requests == {124: (other, 8, 1)}
    native.engine_stopped()
    assert not native._cancelled
    assert executor._terminate_disagg_ctx_finished_requests is terminate
    assert executor._check_disagg_ctx_cache_transfer_status is check_status


@pytest.mark.asyncio
async def test_native_cancel_error_is_observed_by_release():
    executor = SimpleNamespace(
        ctx_in_transmission_requests={
            123: (SimpleNamespace(is_disagg_context_complete_state=False), 7, 1)
        },
        _terminate_disagg_ctx_finished_requests=Mock(
            side_effect=RuntimeError("NIXL cancel failed")
        ),
        cancel_request=Mock(),
        _check_disagg_ctx_cache_transfer_status=Mock(),
    )
    native = NativeSourceReleaser(executor)
    task = asyncio.create_task(asyncio.to_thread(native.release, 123))
    await asyncio.sleep(0.02)
    with pytest.raises(RuntimeError, match="NIXL cancel failed"):
        executor._terminate_disagg_ctx_finished_requests()
    with pytest.raises(RuntimeError, match="NIXL cancel failed"):
        await task
    assert 123 in executor.ctx_in_transmission_requests


@pytest.mark.asyncio
async def test_native_finished_transmission_does_not_retain_metadata():
    source = SimpleNamespace(py_client_id=17, is_disagg_context_complete_state=True)
    executor = SimpleNamespace(
        ctx_in_transmission_requests={123: (source, 7, 1)},
        cancel_request=Mock(),
        _check_disagg_ctx_cache_transfer_status=Mock(),
    )
    executor._terminate_disagg_ctx_finished_requests = lambda: (
        executor.ctx_in_transmission_requests.pop(123)
    )
    native = NativeSourceReleaser(executor)
    assert native.find_source(17) == 123
    assert native.find_source(18) is None
    task = asyncio.create_task(asyncio.to_thread(native.release, 123))
    await asyncio.sleep(0.02)
    executor._terminate_disagg_ctx_finished_requests()
    await asyncio.wait_for(task, 2)
    assert native.find_source(17) is None
    assert not native._cancelled
    native.release(123)
