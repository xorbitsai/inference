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

# CI: linux
"""Unit tests for B2 v5: GPU orphan cleanup helpers.

Tests cover:
- _parse_gpu_indices: int / list / str / None formats
- _snapshot_gpu_occupying_pids: pynvml unavailable / mock
- _snapshot_gpu_free_ratio: pynvml unavailable / mock
- _kill_orphan_gpu_pids: diff-only / no-new-pids
- _wait_pids_dead: all-dead / timeout
- _wait_vram_reclaim_after_terminate: shared GPU / replica exit / timeout
- _snapshot_replica_gpu_children: GPU-owning child / no PID visibility
- _terminate_model: waits for a reparented child that still owns VRAM
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


def _spawn_sleeper():
    import subprocess
    import sys

    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def _stop(proc):
    proc.kill()
    proc.wait()


@pytest.mark.asyncio
async def test_wait_vram_reclaim_returns_at_once_on_shared_gpu():
    """The replica has exited; the GPU stays below the ready ratio because
    other processes use it. There is nothing left to wait for."""
    import time

    from xinference.core.worker import _wait_vram_reclaim_after_terminate

    replica = _spawn_sleeper()
    _stop(replica)

    with patch(
        "xinference.core.worker._snapshot_gpu_free_ratio", return_value=0.74
    ) as mock_ratio:
        _start = time.monotonic()
        reclaimed = await _wait_vram_reclaim_after_terminate(
            [0], {replica.pid}, model_uid="m-0", timeout=30, interval=0.05
        )
        _elapsed = time.monotonic() - _start

    assert reclaimed is True
    assert _elapsed < 1.0
    assert mock_ratio.call_count == 0


@pytest.mark.asyncio
async def test_wait_vram_reclaim_waits_for_replica_exit():
    """A replica process that is still alive is waited for, and the wait
    ends when it exits even though the device-wide ratio stays low."""
    from xinference.core.worker import _wait_vram_reclaim_after_terminate

    replica = _spawn_sleeper()
    polls = []

    def _ratio(devices):
        polls.append(devices)
        if len(polls) == 3:
            _stop(replica)
        return 0.74

    try:
        with patch("xinference.core.worker._snapshot_gpu_free_ratio", _ratio):
            reclaimed = await _wait_vram_reclaim_after_terminate(
                [0], {replica.pid}, model_uid="m-0", timeout=30, interval=0.05
            )
    finally:
        _stop(replica)

    assert reclaimed is True
    assert len(polls) == 3


@pytest.mark.asyncio
async def test_wait_vram_reclaim_stops_when_ratio_recovers():
    from xinference.core.worker import _wait_vram_reclaim_after_terminate

    replica = _spawn_sleeper()
    ratios = iter([0.5, 0.6, 0.95])
    try:
        with patch(
            "xinference.core.worker._snapshot_gpu_free_ratio",
            side_effect=lambda devices: next(ratios),
        ) as mock_ratio:
            reclaimed = await _wait_vram_reclaim_after_terminate(
                [0], {replica.pid}, model_uid="m-0", timeout=30, interval=0.05
            )
    finally:
        _stop(replica)

    assert reclaimed is True
    assert mock_ratio.call_count == 3


@pytest.mark.asyncio
async def test_wait_vram_reclaim_times_out_while_replica_holds_vram():
    """The bounded wait still covers a replica that does not exit."""
    import time

    from xinference.core.worker import _wait_vram_reclaim_after_terminate

    replica = _spawn_sleeper()
    try:
        with patch(
            "xinference.core.worker._snapshot_gpu_free_ratio", return_value=0.74
        ):
            _start = time.monotonic()
            reclaimed = await _wait_vram_reclaim_after_terminate(
                [0], {replica.pid}, model_uid="m-0", timeout=0.3, interval=0.05
            )
            _elapsed = time.monotonic() - _start
    finally:
        _stop(replica)

    assert reclaimed is False
    assert _elapsed >= 0.3


def _spawn_replica_with_child():
    """Start a stand-in replica process that owns one long-lived child."""
    import subprocess
    import sys

    import psutil

    script = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    replica = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    child = psutil.Process(int(replica.stdout.readline()))
    return replica, child


def _stop_replica_with_child(replica, child):
    import psutil

    _stop(replica)
    replica.stdout.close()
    try:
        child.kill()
        child.wait(timeout=5)
    except psutil.NoSuchProcess:
        pass


def test_snapshot_replica_gpu_children():
    from xinference.core.worker import _snapshot_replica_gpu_children

    replica, child = _spawn_replica_with_child()
    try:
        with patch("xinference.core.worker._snapshot_gpu_occupying_pids") as snap:
            # The child owns VRAM next to an unrelated process.
            snap.return_value = {child.pid, 4242}
            assert _snapshot_replica_gpu_children({replica.pid}, [0]) == {child.pid}
            # Only the replica itself is on the GPU.
            snap.return_value = {replica.pid, 4242}
            assert _snapshot_replica_gpu_children({replica.pid}, [0]) == set()
            # NVML cannot attribute anything to the replica (e.g. container
            # PID namespace): every descendant may be the one holding VRAM.
            snap.return_value = {4242}
            assert _snapshot_replica_gpu_children({replica.pid}, [0]) == {child.pid}
            snap.return_value = set()
            assert _snapshot_replica_gpu_children({replica.pid}, [0]) == {child.pid}
    finally:
        _stop_replica_with_child(replica, child)


@pytest.mark.asyncio
async def test_terminate_waits_for_reparented_gpu_child(monkeypatch):
    """The replica exits first while a child it started still owns VRAM on a
    shared GPU. The child cannot be attributed to the model once it has been
    reparented, so it has to be recorded before the pool is destroyed."""
    from unittest.mock import AsyncMock

    import xinference.core.worker as worker_module
    from xinference.core.worker import WorkerActor

    replica, child = _spawn_replica_with_child()
    uid = "model-x-0"
    polls = []

    class _Pool:
        sub_processes: dict = {}

        async def remove_sub_pool(self, address, force=False):
            _stop(replica)  # only the replica dies; its child lives on

    def _ratio(devices):
        # Another tenant keeps the GPU below the ready ratio throughout.
        polls.append(child.is_running())
        if len(polls) == 3:
            child.kill()
            child.wait(timeout=5)
        return 0.74

    async def _no_check(_worker):
        return None

    async def _no_orphans(*args, **kwargs):
        return []  # the reparented child no longer matches the model UID

    monkeypatch.setattr(worker_module, "check_control_session", _no_check)
    monkeypatch.setattr(worker_module, "_kill_orphan_gpu_pids", _no_orphans)
    monkeypatch.setattr(worker_module, "_snapshot_gpu_free_ratio", _ratio)
    monkeypatch.setattr(
        worker_module,
        "_snapshot_gpu_occupying_pids",
        lambda devices: {child.pid, 4242} if child.is_running() else {4242},
    )
    monkeypatch.setattr(WorkerActor, "_finish_launch_progress", AsyncMock())

    worker = MagicMock()
    worker._control_fence = None
    worker._model_uid_launching_guard = {}
    worker._model_uid_to_model_status = {}
    worker._model_uid_to_model = {uid: None}
    worker._model_uid_to_subpool_addresses = {uid: {"addr-1"}}
    worker._model_uid_to_addr = {uid: "addr-1"}
    worker._model_uid_to_launch_args = {uid: {"model_uid": uid, "gpu_idx": [0]}}
    worker._model_uid_to_model_spec = {uid: {}}
    worker._model_uid_to_pid = {uid: replica.pid}
    worker._model_uid_to_subpool_pids = {uid: {replica.pid}}
    worker._main_pool = _Pool()
    worker._event_collector_ref = None
    worker._status_guard_ref = MagicMock(update_instance_info=AsyncMock())
    worker._update_model_state = AsyncMock()
    worker.get_supervisor_ref = AsyncMock()

    try:
        await WorkerActor._terminate_model(worker, uid)
        # The unload returned only after the child released the GPU: the
        # first poll saw it alive, and it was gone when the wait ended.
        assert polls[:3] == [True, True, True]
        assert not child.is_running()
        assert replica.poll() is not None
    finally:
        _stop_replica_with_child(replica, child)
