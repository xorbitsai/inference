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

"""Unit tests for sub-pool creation hardening (H1/H2/H3/C/G).

Covers:
- _create_subpool normal path (env injection, VRAM check skipped, no timeout)
- _create_subpool timeout (C): raises asyncio.TimeoutError, release_devices
  called, leftover sub_pool child killed
- _create_subpool serialization (H3): concurrent calls do not overlap
  append_sub_pool
- _create_subpool env injection (G): XINFERENCE_MODEL_UID in env, os.environ
  not mutated
- _kill_gpu_orphans_by_ppid (H2/H1 helper): PPID==1 + vllm cmdline killed;
  live-worker child (PPID != 1) and non-vllm cmdline spared
- _cleanup_gpu_orphans_on_startup (H1): nvml failure degrades gracefully;
  no orphans returns cleanly; orphan present triggers SIGKILL path
- _nvml_init_with_timeout: success path on the current platform
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _WorkerStub:
    """Plain stand-in for WorkerActor.

    Instantiating WorkerActor directly is unsafe: xoscar's
    StatelessActor.__new__ forbids construction outside an actor pool
    (``object.__new__(WorkerActor)`` raises TypeError). We bind the real
    WorkerActor methods onto this stub so the code under test runs verbatim,
    with only the attributes they read populated.
    """

    pass


def _make_worker():
    """Build a _WorkerStub with the real WorkerActor methods bound and the
    attributes they read populated."""
    from collections import defaultdict

    from xinference.core.worker import WorkerActor

    self = _WorkerStub()
    self._subpool_creation_lock = asyncio.Lock()
    self._main_pool = MagicMock()
    self._ensure_subpool_monitor = AsyncMock()
    self.release_devices = MagicMock()
    self.address = "test:1"
    # allocate_devices_with_gpu_idx reads these; populate enough state for the
    # gpu_idx=[...] path to succeed without real device accounting.
    self._total_gpu_devices = list(range(8))
    self._gpu_to_model_uids = defaultdict(set)
    self._user_specified_gpu_to_model_uids = defaultdict(set)
    self._allow_multi_replica_per_gpu = True
    # Bind the real methods so the code under test runs verbatim.
    self.allocate_devices_with_gpu_idx = (
        WorkerActor.allocate_devices_with_gpu_idx.__get__(self)
    )
    self._allocate_subpool_devices = WorkerActor._allocate_subpool_devices.__get__(self)
    self._spawn_subpool = WorkerActor._spawn_subpool.__get__(self)
    self._create_subpool = WorkerActor._create_subpool.__get__(self)
    self._append_sub_pool_protected = WorkerActor._append_sub_pool_protected.__get__(
        self
    )
    self._cleanup_gpu_orphans_on_startup = (
        WorkerActor._cleanup_gpu_orphans_on_startup.__get__(self)
    )
    return self


async def _hang_coro(*args, **kwargs):
    # A coroutine that never completes, used to drive the timeout path.
    await asyncio.Event().wait()


@pytest.fixture
def patched_subpool_deps(monkeypatch):
    """Patch the module-level GPU/timeout dependencies of _create_subpool so
    tests need no GPU and run in milliseconds."""
    import xinference.core.worker as w

    monkeypatch.setattr(w, "_snapshot_gpu_free_ratio", lambda devs: 0.95)
    monkeypatch.setattr(w, "_kill_gpu_orphans_by_ppid", _async_return([]))
    # Bound the timeout tightly for the hang test; harmless for the fast path.
    monkeypatch.setattr(w, "XINFERENCE_SUBPOOL_LAUNCH_TIMEOUT", 0.2)
    # Use asyncio.wait_for instead of xo.wait_for so tests do not depend on
    # xoscar actor-call cancellation internals.
    monkeypatch.setattr(w.xo, "wait_for", asyncio.wait_for)


def _async_return(value):
    async def _coro(*args, **kwargs):
        return value

    return _coro


# ---------------------------------------------------------------------------
# C: timeout
# ---------------------------------------------------------------------------


async def test_create_subpool_timeout_releases_devices_and_kills_leftover(
    patched_subpool_deps,
):
    self = _make_worker()
    self._main_pool.append_sub_pool = _hang_coro

    # Mock psutil so the leftover-child kill path is exercised without real
    # processes. One child matches (PPID == my pid, start_sub_pool cmdline),
    # one does not (PPID != my pid).
    my_pid = os.getpid()
    matching = MagicMock()
    matching.info = {
        "pid": 9991,
        "ppid": my_pid,
        "cmdline": ["python", "-m", "xoscar", "start_sub_pool", "-sn", "x"],
    }
    other = MagicMock()
    other.info = {"pid": 9992, "ppid": my_pid + 1, "cmdline": ["unrelated"]}
    mock_psutil = MagicMock()
    mock_psutil.process_iter.return_value = [matching, other]
    mock_psutil.NoSuchProcess = ProcessLookupError
    mock_psutil.AccessDenied = PermissionError

    with patch.dict("sys.modules", {"psutil": mock_psutil}):
        with pytest.raises(asyncio.TimeoutError):
            await self._create_subpool("model-1", model_type="LLM", gpu_idx=[0])

    # GPU allocation table must be released so the next launch is not blocked.
    self.release_devices.assert_called_once_with(model_uid="model-1")
    # Only the matching child is killed.
    matching.kill.assert_called_once()
    other.kill.assert_not_called()


# ---------------------------------------------------------------------------
# H3: serialization
# ---------------------------------------------------------------------------


async def test_create_subpool_serializes_append_sub_pool(patched_subpool_deps):
    self = _make_worker()

    in_flight = 0
    max_in_flight = 0
    call_order: list = []

    async def _tracked_append(env=None, start_python=None):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        call_order.append(("start", in_flight))
        await asyncio.sleep(0.05)
        in_flight -= 1
        call_order.append(("end", in_flight))
        return f"addr:{len(call_order)}"

    self._main_pool.append_sub_pool = _tracked_append

    await asyncio.gather(
        self._create_subpool("m-a", model_type="LLM", gpu_idx=[0]),
        self._create_subpool("m-b", model_type="LLM", gpu_idx=[1]),
        self._create_subpool("m-c", model_type="LLM", gpu_idx=[2]),
    )

    # append_sub_pool must never run concurrently.
    assert max_in_flight == 1


# ---------------------------------------------------------------------------
# G: env injection
# ---------------------------------------------------------------------------


async def test_create_subpool_injects_model_uid_env(patched_subpool_deps):
    self = _make_worker()
    captured: dict = {}

    async def _capture_append(env=None, start_python=None):
        captured["env"] = dict(env)
        return "addr:1"

    self._main_pool.append_sub_pool = _capture_append

    os.environ.pop("XINFERENCE_MODEL_UID", None)
    await self._create_subpool("my-model", model_type="LLM", gpu_idx=[0])

    # The sub-pool env dict carries the tag...
    assert captured["env"]["XINFERENCE_MODEL_UID"] == "my-model"
    # ...and the worker main process's os.environ is left untouched.
    assert "XINFERENCE_MODEL_UID" not in os.environ


# ---------------------------------------------------------------------------
# Normal path
# ---------------------------------------------------------------------------


async def test_create_subpool_normal_path_skips_vram_check(patched_subpool_deps):
    self = _make_worker()
    self._main_pool.append_sub_pool = _async_return("addr:1")

    addr, devices = await self._create_subpool(
        "model-1", model_type="LLM", gpu_idx=[0, 1]
    )
    assert addr == "addr:1"
    assert devices == ["0", "1"]
    self._ensure_subpool_monitor.assert_awaited_once()
    self.release_devices.assert_not_called()


async def test_create_subpool_vram_low_triggers_orphan_kill(monkeypatch):
    import xinference.core.worker as w

    monkeypatch.setattr(w, "XINFERENCE_SUBPOOL_LAUNCH_TIMEOUT", 5)
    monkeypatch.setattr(w.xo, "wait_for", asyncio.wait_for)
    killed_calls: list = []
    monkeypatch.setattr(w, "_kill_gpu_orphans_by_ppid", _track_kill(killed_calls))

    # First snapshot (initial check) returns low ratio to trigger the cleanup
    # path; all subsequent calls (poll loop + post-loop log) return healthy.
    calls = {"n": 0}

    def _ratio(devs):
        calls["n"] += 1
        return 0.10 if calls["n"] == 1 else 0.95

    monkeypatch.setattr(w, "_snapshot_gpu_free_ratio", _ratio)

    self = _make_worker()
    self._main_pool.append_sub_pool = _async_return("addr:1")
    await self._create_subpool("model-1", model_type="LLM", gpu_idx=[0])
    assert killed_calls, "expected _kill_gpu_orphans_by_ppid to be invoked"


# ---------------------------------------------------------------------------
# _append_sub_pool_protected: the single protected append_sub_pool entrypoint.
# All three call sites (_create_subpool, need_create_pools gather,
# launch_rank0_model) route through it, so H3/C coverage of the helper itself
# also covers those sites.
# ---------------------------------------------------------------------------


async def test_append_sub_pool_protected_serializes_concurrent_calls(
    patched_subpool_deps,
):
    """The gather pattern used by need_create_pools must not let
    append_sub_pool run concurrently — exactly the H3 deadlock scenario."""
    self = _make_worker()

    in_flight = 0
    max_in_flight = 0

    async def _tracked_append(env=None, start_python=None):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return f"addr:{max_in_flight}"

    self._main_pool.append_sub_pool = _tracked_append

    # Mirror the need_create_pools gather: multiple concurrent appends.
    addrs = await asyncio.gather(
        self._append_sub_pool_protected(model_uid="m-a"),
        self._append_sub_pool_protected(model_uid="m-b"),
        self._append_sub_pool_protected(model_uid="m-c"),
    )
    assert max_in_flight == 1, "append_sub_pool must be serialized"
    assert len(addrs) == 3


async def test_append_sub_pool_protected_timeout_kills_leftover_and_reraises(
    patched_subpool_deps,
):
    """C: on timeout the helper SIGKILLs leftover start_sub_pool children
    (PPID == this worker) and re-raises so callers run their own cleanup."""
    self = _make_worker()
    self._main_pool.append_sub_pool = _hang_coro

    my_pid = os.getpid()
    matching = MagicMock()
    matching.info = {
        "pid": 7701,
        "ppid": my_pid,
        "cmdline": ["python", "-m", "xoscar", "start_sub_pool", "-sn", "y"],
    }
    other = MagicMock()
    other.info = {"pid": 7702, "ppid": my_pid + 1, "cmdline": ["unrelated"]}
    mock_psutil = MagicMock()
    mock_psutil.process_iter.return_value = [matching, other]
    mock_psutil.NoSuchProcess = ProcessLookupError
    mock_psutil.AccessDenied = PermissionError

    with patch.dict("sys.modules", {"psutil": mock_psutil}):
        with pytest.raises(asyncio.TimeoutError):
            await self._append_sub_pool_protected(model_uid="rank0-model")

    matching.kill.assert_called_once()
    other.kill.assert_not_called()


async def test_append_sub_pool_protected_does_not_log_env_values(
    patched_subpool_deps, caplog
):
    """env may carry secrets; the timeout log must emit keys only, never
    values. Uses the timeout path because that is where env is logged."""
    import logging

    self = _make_worker()
    self._main_pool.append_sub_pool = _hang_coro
    mock_psutil = MagicMock()
    mock_psutil.process_iter.return_value = []
    mock_psutil.NoSuchProcess = ProcessLookupError
    mock_psutil.AccessDenied = PermissionError

    secret_env = {"CUDA_VISIBLE_DEVICES": "0", "OPENAI_API_KEY": "sk-secret-xyz"}
    with patch.dict("sys.modules", {"psutil": mock_psutil}):
        with caplog.at_level(logging.ERROR, logger="xinference.core.worker"):
            with pytest.raises(asyncio.TimeoutError):
                await self._append_sub_pool_protected(
                    env=secret_env, model_uid="m-secret"
                )

    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "sk-secret-xyz" not in blob, "env value leaked in timeout log"
    assert "OPENAI_API_KEY" in blob, "env key name should be logged"


def _track_kill(sink):
    async def _coro(devices, model_uid=""):
        sink.append((tuple(devices), model_uid))
        return [4242]

    return _coro


# ---------------------------------------------------------------------------
# H2/H1 helper: _kill_gpu_orphans_by_ppid
# ---------------------------------------------------------------------------


async def test_kill_gpu_orphans_by_ppid_filters_by_ppid_and_cmdline(monkeypatch):
    import xinference.core.worker as w

    procs: dict = {}

    def _make(pid, ppid, cmdline):
        m = MagicMock()
        m.pid = pid  # plain int: production code appends p.pid to the killed list
        m.ppid.return_value = ppid
        m.is_running.return_value = True
        m.status.return_value = "running"
        m.cmdline.return_value = cmdline
        procs[pid] = m
        return m

    # 100: orphan vLLM (PPID==1, vllm cmdline) → killed
    _make(100, 1, ["python", "-m", "vllm"])
    # 200: live worker child (PPID != 1) → spared
    _make(200, 9999, ["python", "-m", "vllm"])
    # 300: orphan but non-vllm cmdline → spared
    _make(300, 1, ["python", "unrelated_app.py"])

    mock_psutil = MagicMock()

    def _process(pid):
        if pid in procs:
            return procs[pid]
        raise ProcessLookupError(pid)

    mock_psutil.Process.side_effect = _process
    mock_psutil.NoSuchProcess = ProcessLookupError
    mock_psutil.AccessDenied = PermissionError
    mock_psutil.STATUS_ZOMBIE = "zombie"

    with patch("xinference.core.worker._snapshot_gpu_occupying_pids") as mock_snap:
        mock_snap.return_value = {100, 200, 300}
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            killed = await w._kill_gpu_orphans_by_ppid([0])

    assert killed == [100]
    # pid 100 (orphan vLLM) gets terminate() then kill(); 200 and 300 are spared.
    procs[100].terminate.assert_called_once()
    procs[100].kill.assert_called_once()
    procs[200].kill.assert_not_called()
    procs[300].kill.assert_not_called()


async def test_kill_gpu_orphans_by_ppid_no_orphans_returns_empty(monkeypatch):
    import xinference.core.worker as w

    with patch("xinference.core.worker._snapshot_gpu_occupying_pids") as mock_snap:
        mock_snap.return_value = set()
        killed = await w._kill_gpu_orphans_by_ppid([0])
    assert killed == []


# ---------------------------------------------------------------------------
# H1: _cleanup_gpu_orphans_on_startup
# ---------------------------------------------------------------------------


async def test_startup_cleanup_nvml_failure_degrades(monkeypatch):
    import xinference.core.worker as w

    monkeypatch.setattr(w, "_nvml_init_with_timeout", lambda timeout=10: False)
    self = _make_worker()
    # Must not raise and must not attempt any snapshot/kill.
    with patch("xinference.core.worker._snapshot_gpu_occupying_pids") as mock_snap:
        await self._cleanup_gpu_orphans_on_startup()
    mock_snap.assert_not_called()


async def test_startup_cleanup_no_orphans_returns_cleanly(monkeypatch):
    import xinference.core.worker as w

    monkeypatch.setattr(w, "_nvml_init_with_timeout", lambda timeout=10: True)
    monkeypatch.setattr(
        "xinference.core.worker._snapshot_gpu_occupying_pids", lambda devs: set()
    )
    monkeypatch.setattr(
        "xinference.core.worker._snapshot_gpu_free_ratio", lambda devs: 0.95
    )
    self = _make_worker()
    await self._cleanup_gpu_orphans_on_startup()  # must not raise


async def test_startup_cleanup_kills_orphans(monkeypatch):
    import xinference.core.worker as w

    monkeypatch.setattr(w, "_nvml_init_with_timeout", lambda timeout=10: True)

    # nvmlDeviceGetCount inside the method (pynvml is imported locally).
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    monkeypatch.setattr(
        "xinference.core.worker._snapshot_gpu_occupying_pids", lambda devs: {500}
    )
    monkeypatch.setattr(
        "xinference.core.worker._snapshot_gpu_free_ratio", lambda devs: 0.95
    )

    orphan = MagicMock()
    orphan.pid = 500  # plain int: production code appends p.pid to the killed list
    orphan.ppid.return_value = 1
    orphan.is_running.return_value = True
    orphan.status.return_value = "running"
    orphan.cmdline.return_value = ["python", "-m", "vllm"]
    mock_psutil = MagicMock()
    mock_psutil.Process.return_value = orphan
    mock_psutil.NoSuchProcess = ProcessLookupError
    mock_psutil.AccessDenied = PermissionError
    mock_psutil.STATUS_ZOMBIE = "zombie"

    self = _make_worker()
    with patch.dict("sys.modules", {"pynvml": mock_pynvml, "psutil": mock_psutil}):
        await self._cleanup_gpu_orphans_on_startup()

    # orphan (pid 500) gets terminate() then kill() via _kill_gpu_orphans_by_ppid.
    orphan.terminate.assert_called_once()
    orphan.kill.assert_called_once()


# ---------------------------------------------------------------------------
# _nvml_init_with_timeout
# ---------------------------------------------------------------------------


def test_nvml_init_with_timeout_success():
    import xinference.core.worker as w

    mock_pynvml = MagicMock()
    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        assert w._nvml_init_with_timeout(timeout=2) is True
    mock_pynvml.nvmlInit.assert_called_once()


def test_nvml_init_with_timeout_failure_returns_false():
    import xinference.core.worker as w

    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.side_effect = RuntimeError("driver down")
    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        assert w._nvml_init_with_timeout(timeout=2) is False


def test_nvml_init_with_timeout_background_thread_falls_back():
    """signal.signal only works in the main thread; a background-thread caller
    must fall back to a direct nvmlInit instead of raising ValueError."""
    import threading

    import xinference.core.worker as w

    mock_pynvml = MagicMock()
    result: dict = {}

    def _run():
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result["ok"] = w._nvml_init_with_timeout(timeout=2)

    t = threading.Thread(target=_run)
    t.start()
    t.join()
    assert result["ok"] is True
    mock_pynvml.nvmlInit.assert_called_once()
