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

import asyncio
import contextlib
import os
import threading

import pytest

from ..supervisor import SupervisorActor
from ..virtual_env_manager import VirtualEnvConflictError


class _WorkerWithVirtualEnv:
    def __init__(self, error=None):
        self._error = error
        self.remove_calls = 0

    async def list_virtual_envs(self, model_name, model_engine):
        return [{"model_name": model_name, "model_engine": model_engine}]

    async def remove_virtual_env(self, model_name, model_engine, python_version):
        self.remove_calls += 1
        if self._error is not None:
            raise self._error
        return True


@pytest.mark.asyncio
async def test_supervisor_preserves_virtual_env_conflict_from_worker():
    conflict = VirtualEnvConflictError("environment is used by qwen-rep0")
    busy_worker = _WorkerWithVirtualEnv(conflict)
    idle_worker = _WorkerWithVirtualEnv()
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._worker_address_to_worker = {
        "worker-0": busy_worker,
        "worker-1": idle_worker,
    }

    with pytest.raises(VirtualEnvConflictError, match="qwen-rep0"):
        await SupervisorActor.remove_virtual_env(
            supervisor,
            model_name="Qwen3.8-Flash-Next",
            model_engine="vllm",
            python_version="3.12",
        )

    assert busy_worker.remove_calls == 1
    assert idle_worker.remove_calls == 1


def test_worker_serializes_virtual_env_deletion_with_concurrent_launch(
    tmp_path, monkeypatch
):
    from xinference import constants

    from .. import virtual_env_manager as virtual_env_manager_mod
    from .. import worker as worker_mod
    from ..virtual_env_manager import VirtualEnvManager
    from ..worker import WorkerActor

    virtual_env_root = tmp_path / "virtualenv"
    env_path = virtual_env_root / "v4" / "Qwen3.8-Flash-Next" / "vllm" / "3.12"
    env_path.mkdir(parents=True)
    monkeypatch.setattr(constants, "XINFERENCE_VIRTUAL_ENV_DIR", str(virtual_env_root))
    monkeypatch.setattr(worker_mod, "XINFERENCE_VIRTUAL_ENV_DIR", str(virtual_env_root))
    monkeypatch.setattr(
        virtual_env_manager_mod,
        "XINFERENCE_VIRTUAL_ENV_DIR",
        str(virtual_env_root),
    )

    worker = WorkerActor.__new__(WorkerActor)
    worker._virtual_env_manager = VirtualEnvManager("worker-0")
    worker._virtual_env_usages = {}
    worker._model_uid_to_virtual_env_path = {}
    worker._virtual_env_usage_lock = threading.Lock()

    launch_reserved = threading.Event()
    release_launch = threading.Event()
    deletion_waiting = threading.Event()
    deletion_finished = threading.Event()
    deletion_errors = []
    original_path_lock = worker_mod._exclusive_venv_path_lock

    @contextlib.contextmanager
    def observed_path_lock(path):
        if threading.current_thread().name == "delete-virtual-env":
            deletion_waiting.set()
        with original_path_lock(path):
            yield

    monkeypatch.setattr(worker_mod, "_exclusive_venv_path_lock", observed_path_lock)

    def launch_model():
        with original_path_lock(str(env_path)):
            worker._reserve_virtual_env_usage(
                str(env_path), "fingerprint-a", "qwen-rep0", setup_required=True
            )
            launch_reserved.set()
            assert release_launch.wait(timeout=5)

    def delete_virtual_env():
        try:
            asyncio.run(
                WorkerActor.remove_virtual_env(
                    worker,
                    model_name="Qwen3.8-Flash-Next",
                    model_engine="vllm",
                    python_version="3.12",
                )
            )
        except Exception as exc:
            deletion_errors.append(exc)
        finally:
            deletion_finished.set()

    launch_thread = threading.Thread(target=launch_model, name="launch-model")
    deletion_thread = threading.Thread(
        target=delete_virtual_env, name="delete-virtual-env"
    )
    try:
        launch_thread.start()
        assert launch_reserved.wait(timeout=5)
        deletion_thread.start()
        assert deletion_waiting.wait(timeout=5)
        assert not deletion_finished.is_set()
    finally:
        release_launch.set()
        launch_thread.join(timeout=5)
        if deletion_thread.ident is not None:
            deletion_thread.join(timeout=5)

    assert not launch_thread.is_alive()
    assert not deletion_thread.is_alive()
    assert len(deletion_errors) == 1
    assert isinstance(deletion_errors[0], VirtualEnvConflictError)
    assert "qwen-rep0" in str(deletion_errors[0])
    assert os.path.isdir(env_path)
