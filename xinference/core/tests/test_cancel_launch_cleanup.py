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
