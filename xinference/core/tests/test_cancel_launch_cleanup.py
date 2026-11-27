# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# CI: linux
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

"""Cancelling a launch mid-download must not leak worker resources.

Devices are reserved before the download starts and ``launch_info.sub_pools``
is populated only afterwards, so a cancel that bypassed the launch cleanup
handler would leave GPU capacity reserved and the launching guard occupied
until the worker restarts — the next launch of the same uid then fails with
"<uid> is running".
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest


class _WorkerStub:
    """Stand-in for WorkerActor; see test_subpool_hardening for the rationale
    (xoscar forbids constructing a StatelessActor outside an actor pool)."""

    pass


def _make_worker():
    from xinference.core.worker import WorkerActor

    self = _WorkerStub()
    self.address = "test:1"
    self._main_pool = MagicMock()
    self._main_pool.remove_sub_pool = AsyncMock()
    self._model_uid_launching_guard = {}
    self._cache_uid_to_download_info = {}
    self._model_uid_to_model = {}
    self._virtual_env_usages = {}
    self._model_uid_to_virtual_env_path = {}
    self._virtual_env_usage_lock = threading.Lock()
    self._launch_active = 0
    self._launch_waiting = 0
    self.release_devices = MagicMock()
    self._update_model_state = AsyncMock()
    self._get_progressor = AsyncMock(return_value=MagicMock())
    self._launch_semaphore = asyncio.Semaphore(1)
    self._check_model_is_valid = MagicMock()
    self._create_virtual_env_manager = MagicMock(return_value=None)
    self._spawn_subpool = AsyncMock(return_value="test:2")
    self._allocate_subpool_devices = AsyncMock(return_value=({}, [0]))
    self._upload_download_progress = MagicMock()
    self._download_model_files = WorkerActor._download_model_files.__get__(self)
    self.update_cache_status = AsyncMock()
    self._prepare_virtual_env = AsyncMock()
    self._release_virtual_env_usage = WorkerActor._release_virtual_env_usage.__get__(
        self
    )
    self.launch_builtin_model = WorkerActor.launch_builtin_model.__get__(self)
    self.cancel_launch_model = WorkerActor.cancel_launch_model.__get__(self)
    self.get_model_launch_status = WorkerActor.get_model_launch_status.__get__(self)
    return self


def _raise(exc):
    def _fn(*args, **kwargs):
        raise exc()

    return _fn


@pytest.mark.asyncio
async def test_cancelled_download_releases_devices_and_guard(monkeypatch):
    """A CancelledError from the download phase must run the same cleanup as
    an ordinary failure, so devices and the launching guard are freed."""
    worker = _make_worker()
    monkeypatch.setattr("xinference.core.worker.XINFERENCE_MODEL_DOWNLOAD_WORKERS", 6)
    monkeypatch.setattr(
        "xinference.core.worker.create_model_instance", _raise(asyncio.CancelledError)
    )

    with pytest.raises(asyncio.CancelledError):
        await worker.launch_builtin_model(
            model_uid="m-rep0",
            model_name="m",
            model_size_in_billions=None,
            model_format=None,
            quantization=None,
            model_engine=None,
        )

    worker.release_devices.assert_called_once_with(model_uid="m-rep0")
    assert (
        worker._allocate_subpool_devices.await_args.kwargs["env"][
            "HF_HUB_DOWNLOAD_WORKERS"
        ]
        == "6"
    )
    assert "m-rep0" not in worker._model_uid_launching_guard
    assert worker.get_model_launch_status("m-rep0") is None


@pytest.mark.asyncio
async def test_ordinary_failure_still_releases_devices(monkeypatch):
    """Regression guard: widening the handler must not change the Exception path."""
    worker = _make_worker()
    monkeypatch.setattr(
        "xinference.core.worker.create_model_instance", _raise(RuntimeError)
    )

    with pytest.raises(RuntimeError):
        await worker.launch_builtin_model(
            model_uid="m-rep0",
            model_name="m",
            model_size_in_billions=None,
            model_format=None,
            quantization=None,
            model_engine=None,
        )

    worker.release_devices.assert_called_once_with(model_uid="m-rep0")
    assert "m-rep0" not in worker._model_uid_launching_guard


@pytest.mark.asyncio
async def test_cancel_dependency_install_reaches_uv():
    from xinference.core.worker import LaunchInfo

    worker = _make_worker()
    worker._status_guard_ref = None
    worker._cancel_download = lambda info: info.cancel_event.set()
    manager = MagicMock()
    launch_info = LaunchInfo(virtual_env_manager=manager)
    worker._model_uid_launching_guard["m-rep0"] = launch_info

    def cancel_install():
        assert launch_info.cancel_event.is_set()

    manager.cancel_install.side_effect = cancel_install
    asyncio.get_running_loop().call_later(
        0.25, worker._model_uid_launching_guard.pop, "m-rep0"
    )
    await worker.cancel_launch_model("m-rep0")
    manager.cancel_install.assert_called_once_with()
    assert "m-rep0" not in worker._model_uid_launching_guard


def test_cancelled_dependency_setup_skips_install():
    from xinference.core.worker import WorkerActor

    cancel_event = threading.Event()
    cancel_event.set()
    manager = MagicMock()
    with pytest.raises(RuntimeError, match="Launch cancelled during dependency setup"):
        WorkerActor._prepare_virtual_env(
            manager, None, ["example"], None, cancel_event=cancel_event
        )
    manager.install_packages.assert_not_called()


def test_cancel_while_waiting_for_environment_lock(tmp_path):
    from xinference.core.worker import _exclusive_venv_path_lock

    path = str(tmp_path / "env")
    cancelled = threading.Event()
    with _exclusive_venv_path_lock(path):
        cancelled.set()
        with pytest.raises(RuntimeError, match="Launch cancelled"):
            with _exclusive_venv_path_lock(path, cancelled):
                pytest.fail("cancelled waiter acquired the environment")


@pytest.mark.asyncio
async def test_cancel_creation_waiter_before_environment_write(tmp_path, monkeypatch):
    from xinference.core import worker as module

    worker = _make_worker()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()

    def create(enabled, name, path, cancel_event):
        loop.call_soon_threadsafe(entered.set)
        return module.WorkerActor._create_virtual_env_manager(
            enabled, name, path, cancel_event
        )

    worker._create_virtual_env_manager = create
    factory = MagicMock()
    monkeypatch.setattr("xoscar.virtualenv.get_virtual_env_manager", factory)
    monkeypatch.setattr(module, "XINFERENCE_VIRTUAL_ENV_DIR", str(tmp_path))
    import platform

    env = tmp_path / "v4" / "m" / "default" / platform.python_version()
    with module._exclusive_venv_path_lock(str(env)):
        launch = asyncio.create_task(
            worker.launch_builtin_model(
                model_uid="m-rep0",
                model_name="m",
                model_size_in_billions=None,
                model_format=None,
                quantization=None,
                model_engine=None,
                enable_virtual_env=True,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            launch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(launch, 2)
        finally:
            launch.cancel()
            await asyncio.gather(launch, return_exceptions=True)
    factory.assert_not_called()
    assert "m-rep0" not in worker._model_uid_launching_guard


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_directory", [False, True])
async def test_cancel_running_uv_venv(tmp_path, monkeypatch, existing_directory):
    import os
    import sys

    from xoscar.virtualenv.uv import UVVirtualEnvManager

    from xinference.core.worker import WorkerActor, _exclusive_venv_path_lock

    env = tmp_path / "env"
    existing_file = env / "existing.txt"
    if existing_directory:
        env.mkdir()
        existing_file.write_text("keep")
    started = tmp_path / "started"
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys, time\n"
        "env = pathlib.Path(sys.argv[2])\n"
        "env.mkdir(exist_ok=True)\n"
        "(env / 'pyvenv.cfg').touch()\n"
        f"pathlib.Path({str(started)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    uv.chmod(0o755)
    monkeypatch.setattr(
        UVVirtualEnvManager, "_get_uv_path", staticmethod(lambda: str(uv))
    )
    cancelled = threading.Event()
    creation = asyncio.create_task(
        asyncio.to_thread(
            WorkerActor._create_virtual_env_manager, True, "uv", str(env), cancelled
        )
    )

    async def wait_for_uv():
        while not started.exists() or not started.read_text():
            if creation.done():
                await creation
                pytest.fail("uv venv exited before cancellation")
            await asyncio.sleep(0.01)
        return int(started.read_text())

    try:
        pid = await asyncio.wait_for(wait_for_uv(), 5)
        cancelled.set()
        with pytest.raises(RuntimeError, match="cancelled during virtual environment"):
            await asyncio.wait_for(asyncio.shield(creation), 5)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        with _exclusive_venv_path_lock(str(env)):
            assert env.exists() == existing_directory
            assert not UVVirtualEnvManager(env).exists_env()
            if existing_directory:
                assert existing_file.read_text() == "keep"

        uv.write_text(
            f"#!{sys.executable}\n"
            "import pathlib, sys\n"
            "env = pathlib.Path(sys.argv[2])\n"
            "assert not env.exists() or '--allow-existing' in sys.argv\n"
            "(env / 'bin').mkdir(parents=True, exist_ok=True)\n"
            "(env / 'bin' / 'python').symlink_to(sys.executable)\n"
            "(env / 'pyvenv.cfg').touch()\n"
        )
        manager = await asyncio.to_thread(
            WorkerActor._create_virtual_env_manager,
            True,
            "uv",
            str(env),
            threading.Event(),
        )
        assert manager.exists_env()
        assert os.path.exists(manager.get_python_path())
        if existing_directory:
            assert existing_file.read_text() == "keep"
    finally:
        cancelled.set()
        await asyncio.gather(creation, return_exceptions=True)
