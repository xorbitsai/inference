# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
import multiprocessing
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ..block_tracker import VLLMBlockTracker

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux memfd required")


def _read_in_child(metadata, key, queue):
    from ....xavier.local_directory import LocalBlockDirectory

    directory = LocalBlockDirectory.attach(metadata)
    queue.put(None if directory is None else directory.contains(key))
    if directory is not None:
        directory.close()


def test_directory_publication_collisions_eviction_and_recovery():
    from ....xavier.local_directory import LocalBlockDirectory

    tracker = VLLMBlockTracker()
    collision = 1 << 51  # Both filter positions coincide with key zero.
    tracker.register_snapshot_blocks(0, [0, collision], 1)
    directory = LocalBlockDirectory.attach(tracker.get_snapshot_directory(0))
    assert directory is not None
    try:
        assert directory.contains(0) is True and directory.contains(999) is False
        tracker.register_snapshot_blocks(0, [0], 2)
        tracker.unregister_blocks(0, 1, [0, collision, collision])
        assert directory.contains(0) is True
        tracker.unregister_rank(2)
        tracker.register_rank(2)
        assert directory.contains(0) is False
        # Publication is visible immediately, without a refresh interval.
        tracker.register_snapshot_blocks(0, [999], 2)
        assert directory.contains(999) is True
        tracker.register_blocks(0, [(123, 999)], 2)
        assert directory.contains(123) is None and not directory.valid
        assert tracker.get_snapshot_directory(0) is None
    finally:
        directory.close()
        asyncio.run(tracker.__pre_destroy__())


def test_busy_writer_falls_back_and_owner_close_invalidates_reader():
    from ....xavier.local_directory import LocalBlockDirectory, SharedBlockDirectory

    owner = SharedBlockDirectory()
    directory = LocalBlockDirectory.attach(owner.metadata())
    assert directory is not None
    try:
        with owner.update():
            assert directory.contains(111) is None and directory.valid
            owner.add(111)
        assert directory.contains(111) is True
        owner.close()
        assert directory.contains(111) is None and not directory.valid
    finally:
        owner.close()
        directory.close()


def test_spawned_reader_and_foreign_or_reused_owner_rejection():
    from ....xavier.local_directory import LocalBlockDirectory, SharedBlockDirectory

    owner = SharedBlockDirectory()
    try:
        with owner.update():
            owner.add(111)
        metadata = owner.metadata()
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_read_in_child, args=(metadata, 111, queue))
        proc.start()
        try:
            assert queue.get(timeout=30) is True
            proc.join(timeout=30)
            assert proc.exitcode == 0
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            queue.close()
        assert (
            LocalBlockDirectory.attach({**metadata, "boot_id": "foreign-host"}) is None
        )
        assert LocalBlockDirectory.attach({**metadata, "token": b"wrong-owner"}) is None
        assert (
            LocalBlockDirectory.attach(
                {**metadata, "path": f"/proc/{os.getpid()}/fd/999999"}
            )
            is None
        )
    finally:
        owner.close()


def test_negative_shortcut_and_invalid_owner_reconnect(connector):
    directory = SimpleNamespace(
        contains=Mock(return_value=False), valid=True, close=Mock()
    )
    connector._local_directory = directory
    connector._query_remote_blocks = AsyncMock(return_value={})
    request = SimpleNamespace(request_id="r", prompt_token_ids=list(range(33)))
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    connector._query_remote_blocks.assert_not_awaited()
    directory.contains.return_value = None
    directory.valid = False
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    directory.close.assert_called_once()
    assert connector._local_directory is None
    connector._query_remote_blocks.assert_awaited_once()


def _owner_in_child(queue, stop):
    from ....xavier.local_directory import SharedBlockDirectory

    owner = SharedBlockDirectory()
    with owner.update():
        owner.add(111)
    queue.put(owner.metadata())
    queue.close()
    queue.join_thread()
    stop.wait(30)
    os._exit(0)  # Simulate owner death without publishing its closed flag.


def test_owner_process_death_invalidates_retained_mapping():
    from ....xavier.local_directory import LocalBlockDirectory

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    stop = ctx.Event()
    proc = ctx.Process(target=_owner_in_child, args=(queue, stop))
    directory = None
    proc.start()
    try:
        directory = LocalBlockDirectory.attach(queue.get(timeout=30))
        assert directory is not None and directory.contains(111) is True
        stop.set()
        proc.join(timeout=30)
        assert proc.exitcode == 0
        assert directory.contains(111) is None and not directory.valid
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        if directory is not None:
            directory.close()
        queue.close()


@pytest.mark.asyncio
async def test_shared_memory_unavailable_keeps_authoritative_query(
    connector, monkeypatch
):
    from ....xavier import local_directory

    def unavailable():
        raise OSError("shared memory unavailable")

    monkeypatch.setattr(local_directory, "SharedBlockDirectory", unavailable)
    tracker = VLLMBlockTracker()
    tracker.register_snapshot_blocks(0, [111], 2)
    ref = SimpleNamespace(
        get_snapshot_directory=AsyncMock(side_effect=tracker.get_snapshot_directory),
        query_blocks=AsyncMock(side_effect=tracker.query_blocks),
    )
    connector._get_tracker_ref = AsyncMock(return_value=ref)
    assert await connector._query_remote_blocks("r", [(111, 0)]) == {2: {(111, 111, 0)}}
    assert await connector._query_remote_blocks("s", [(222, 0)]) == {}
    assert ref.get_snapshot_directory.await_count == 1
    assert ref.query_blocks.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [None, {"boot_id": "foreign-host"}])
async def test_permanently_unavailable_directory_stops_discovery(connector, metadata):
    ref = SimpleNamespace(
        get_snapshot_directory=AsyncMock(return_value=metadata),
        query_blocks=AsyncMock(return_value={}),
    )
    connector._get_tracker_ref = AsyncMock(return_value=ref)
    for _ in range(3):
        connector._directory_retry_at = 0
        assert await connector._query_remote_blocks("r", [(111, 0)]) == {}
    assert connector._directory_disabled
    ref.get_snapshot_directory.assert_awaited_once_with(0)
    assert ref.query_blocks.await_count == 3


@pytest.mark.asyncio
async def test_transient_directory_error_retries_without_losing_query(connector):
    ref = SimpleNamespace(
        get_snapshot_directory=AsyncMock(side_effect=[OSError("disconnected"), None]),
        query_blocks=AsyncMock(return_value={2: {(111, 111, 0)}}),
    )
    connector._get_tracker_ref = AsyncMock(return_value=ref)
    assert await connector._query_remote_blocks("r", [(111, 0)]) == {2: {(111, 111, 0)}}
    assert not connector._directory_disabled
    connector._directory_retry_at = 0
    assert await connector._query_remote_blocks("s", [(111, 0)]) == {2: {(111, 111, 0)}}
    assert connector._directory_disabled
    assert ref.get_snapshot_directory.await_count == 2
