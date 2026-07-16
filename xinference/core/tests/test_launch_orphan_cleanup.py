# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for B2 v5: GPU orphan cleanup helpers.

Tests cover:
- _parse_gpu_indices: int / list / str / None formats
- _snapshot_gpu_occupying_pids: pynvml unavailable / mock
- _snapshot_gpu_free_ratio: pynvml unavailable / mock
- _kill_orphan_gpu_pids: diff-only / no-new-pids
- _wait_pids_dead: all-dead / timeout
"""

from unittest.mock import MagicMock, patch

import pytest


def test_parse_gpu_indices_int():
    from xinference.core.worker import _parse_gpu_indices

    assert _parse_gpu_indices(0) == [0]
    assert _parse_gpu_indices(3) == [3]


def test_parse_gpu_indices_list():
    from xinference.core.worker import _parse_gpu_indices

    assert _parse_gpu_indices([0, 1]) == [0, 1]
    assert _parse_gpu_indices([]) == []


def test_parse_gpu_indices_str():
    from xinference.core.worker import _parse_gpu_indices

    assert _parse_gpu_indices("0,1") == [0, 1]
    assert _parse_gpu_indices("2, 3, 4") == [2, 3, 4]
    assert _parse_gpu_indices("") == []


def test_parse_gpu_indices_none():
    from xinference.core.worker import _parse_gpu_indices

    assert _parse_gpu_indices(None) == []


def test_snapshot_gpu_occupying_pids_no_pynvml():
    with patch.dict("sys.modules", {"pynvml": None}):
        from xinference.core.worker import _snapshot_gpu_occupying_pids

        assert _snapshot_gpu_occupying_pids([0]) == set()


def test_snapshot_gpu_occupying_pids_with_mock():
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle"
    mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [
        MagicMock(pid=123),
        MagicMock(pid=456),
    ]
    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        from xinference.core.worker import _snapshot_gpu_occupying_pids

        pids = _snapshot_gpu_occupying_pids([0])
        assert pids == {123, 456}


def test_snapshot_gpu_free_ratio_no_pynvml():
    with patch.dict("sys.modules", {"pynvml": None}):
        from xinference.core.worker import _snapshot_gpu_free_ratio

        assert _snapshot_gpu_free_ratio([0]) == -1


def test_snapshot_gpu_free_ratio_with_mock():
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.return_value = None
    handle = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = handle
    mem_info = MagicMock()
    mem_info.free = 22 * 1024 * 1024 * 1024  # 22 GiB
    mem_info.total = 24 * 1024 * 1024 * 1024  # 24 GiB
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info
    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        from xinference.core.worker import _snapshot_gpu_free_ratio

        ratio = _snapshot_gpu_free_ratio([0])
        assert 0.9 <= ratio <= 0.95  # 22/24 ≈ 0.917


@pytest.mark.asyncio
async def test_kill_orphan_gpu_pids_diff_only():
    from xinference.core.worker import _kill_orphan_gpu_pids

    # psutil is imported locally inside _kill_orphan_gpu_pids, so we patch
    # sys.modules. Without this, psutil.Process(300) on a non-existent PID
    # raises NoSuchProcess and the kill is skipped before p.kill() is reached.
    # Each PID gets its own Process mock so we can attribute kill() calls.
    procs = {}

    def _make_proc(pid):
        if pid not in procs:
            m = MagicMock()
            m.is_running.return_value = True
            m.status.return_value = "running"
            m.cmdline.return_value = ["python", "-m", "vllm"]
            procs[pid] = m
        return procs[pid]

    mock_psutil = MagicMock()
    mock_psutil.Process.side_effect = _make_proc
    mock_psutil.NoSuchProcess = ProcessLookupError
    mock_psutil.AccessDenied = PermissionError
    mock_psutil.STATUS_ZOMBIE = "zombie"

    with patch("xinference.core.worker._snapshot_gpu_occupying_pids") as mock_snap:
        # _kill_orphan_gpu_pids calls _snapshot_gpu_occupying_pids once (for post_pids).
        # pre_pids is passed as argument, not from a snapshot call.
        mock_snap.return_value = {100, 200, 300}
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            await _kill_orphan_gpu_pids([0], {100, 200}, grace=0.01)
            killed_pids = {pid for pid, m in procs.items() if m.kill.called}
            assert 300 in killed_pids
            # PIDs in pre_pids must not be touched
            assert 100 not in killed_pids
            assert 200 not in killed_pids


@pytest.mark.asyncio
async def test_kill_orphan_gpu_pids_no_new():
    from xinference.core.worker import _kill_orphan_gpu_pids

    with patch("xinference.core.worker._snapshot_gpu_occupying_pids") as mock_snap:
        mock_snap.side_effect = [{100}, {100}]
        with patch("os.kill") as mock_kill:
            await _kill_orphan_gpu_pids([0], {100}, grace=0.01)
            assert mock_kill.call_count == 0


@pytest.mark.asyncio
async def test_wait_pids_dead_all_dead():
    from xinference.core.worker import _wait_pids_dead

    mock_psutil = MagicMock()
    mock_psutil.NoSuchProcess = ProcessLookupError
    mock_psutil.AccessDenied = PermissionError
    mock_psutil.Process.side_effect = ProcessLookupError("no such process")
    with patch.dict("sys.modules", {"psutil": mock_psutil}):
        import time

        _start = time.monotonic()
        await _wait_pids_dead({123, 456}, timeout=3.0)
        _elapsed = time.monotonic() - _start
        assert _elapsed < 1.0


@pytest.mark.asyncio
async def test_wait_pids_dead_timeout():
    from xinference.core.worker import _wait_pids_dead

    mock_psutil = MagicMock()
    mock_psutil.NoSuchProcess = ProcessLookupError
    mock_psutil.AccessDenied = PermissionError
    mock_proc = MagicMock()
    mock_proc.is_running.return_value = True
    mock_proc.status.return_value = "running"
    mock_psutil.Process.return_value = mock_proc
    with patch.dict("sys.modules", {"psutil": mock_psutil}):
        import time

        _start = time.monotonic()
        await _wait_pids_dead({123}, timeout=0.5)
        _elapsed = time.monotonic() - _start
        assert _elapsed >= 0.4


@pytest.mark.asyncio
async def test_kill_orphan_gpu_pids_model_uid_filter():
    """When model_uid is given, only processes with model_uid in cmdline
    (own or ancestor) should be killed."""
    from xinference.core.worker import _kill_orphan_gpu_pids

    procs = {}

    def _make_proc(pid, cmdline, parent=None):
        m = MagicMock()
        m.is_running.return_value = True
        m.status.return_value = "running"
        m.cmdline.return_value = cmdline
        m.parent.return_value = parent
        procs[pid] = m
        return m

    # PID 300: vLLM worker with model_uid in cmdline → should be killed
    _make_proc(300, ["Xinf", "vLLM", "worker:", "0", "[test-model-1]"])
    # PID 400: EngineCore, no model_uid in own cmdline, but parent has it
    parent_400 = _make_proc(401, ["Xinf", "vLLM", "worker:", "0", "[test-model-1]"])
    _make_proc(400, ["vllm::EngineCore"], parent=parent_400)
    # PID 500: unrelated process, no model_uid anywhere → should be skipped
    _make_proc(500, ["python", "some_other_app.py"])

    mock_psutil = MagicMock()

    def _process_side_effect(pid):
        if pid in procs:
            return procs[pid]
        raise ProcessLookupError(f"no such process {pid}")

    mock_psutil.Process.side_effect = _process_side_effect
    mock_psutil.NoSuchProcess = ProcessLookupError
    mock_psutil.AccessDenied = PermissionError
    mock_psutil.STATUS_ZOMBIE = "zombie"

    with patch("xinference.core.worker._snapshot_gpu_occupying_pids") as mock_snap:
        mock_snap.return_value = {300, 400, 500}
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            killed = await _kill_orphan_gpu_pids(
                [0], set(), model_uid="test-model-1", grace=0.01
            )
            assert 300 in killed
            assert 400 in killed
            assert 500 not in killed


@pytest.mark.asyncio
async def test_kill_orphan_gpu_pids_model_uid_skips_other_model():
    """When model_uid is given, processes belonging to a different model
    (different uid in cmdline) should be skipped."""
    from xinference.core.worker import _kill_orphan_gpu_pids

    procs = {}

    def _make_proc(pid, cmdline, parent=None):
        m = MagicMock()
        m.is_running.return_value = True
        m.status.return_value = "running"
        m.cmdline.return_value = cmdline
        m.parent.return_value = parent
        procs[pid] = m
        return m

    # PID 300: vLLM worker for a DIFFERENT model → should be skipped
    _make_proc(300, ["Xinf", "vLLM", "worker:", "0", "[other-model]"])

    mock_psutil = MagicMock()

    def _process_side_effect(pid):
        if pid in procs:
            return procs[pid]
        raise ProcessLookupError(f"no such process {pid}")

    mock_psutil.Process.side_effect = _process_side_effect
    mock_psutil.NoSuchProcess = ProcessLookupError
    mock_psutil.AccessDenied = PermissionError
    mock_psutil.STATUS_ZOMBIE = "zombie"

    with patch("xinference.core.worker._snapshot_gpu_occupying_pids") as mock_snap:
        mock_snap.return_value = {300}
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            killed = await _kill_orphan_gpu_pids(
                [0], set(), model_uid="test-model-1", grace=0.01
            )
            assert killed == []
