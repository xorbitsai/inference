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
from unittest.mock import AsyncMock, MagicMock

import pytest

from xinference.core.supervisor import ReplicaInfo, SupervisorActor


class _DummySupervisor:
    _load_autostart_entries = SupervisorActor._load_autostart_entries

    def __init__(self, entries):
        self._autostart_store_lock = asyncio.Lock()
        self._launch_history_store = MagicMock()
        self._launch_history_store.list_autostart.return_value = entries

    async def _run_in_executor(self, func, *args):
        return func(*args)


class _DummyAutostartRunner:
    _autostart_one_model = SupervisorActor._autostart_one_model

    def __init__(self, model_status, attempts: int):
        self._model_status = model_status
        self._autostart_model_states = {
            "uid-1": {"attempts": attempts, "last_error": "previous failure"}
        }
        self.launched: list = []

    async def _get_autostart_model_status(self, model_uid: str):
        return self._model_status

    def _autostart_waiting_for_worker(self, launch):
        return False

    async def _launch_autostart_model(self, launch):
        self.launched.append(launch)
        return launch["model_uid"]


@pytest.mark.asyncio
async def test_autostart_status_falls_back_without_status_guard():
    class DummySupervisor:
        _get_autostart_model_status = SupervisorActor._get_autostart_model_status

        def __init__(self):
            self._status_guard_ref = None
            self._model_uid_to_replica_info = {"uid-1": object()}

    assert await DummySupervisor()._get_autostart_model_status("uid-1") == "READY"


@pytest.mark.asyncio
async def test_load_autostart_entries_reads_sqlite_store_and_normalizes():
    supervisor = _DummySupervisor(
        [
            {
                "priority": "5",
                "max_retries": "2",
                "retry_interval_seconds": "9",
                "launch": {
                    "model_name": "llama",
                    "model_uid": "uid-1",
                    "replica": "2",
                },
            }
        ]
    )

    entries = await supervisor._load_autostart_entries()

    assert entries == [
        {
            "enabled": True,
            "priority": 5,
            "max_retries": 2,
            "retry_interval_seconds": 9,
            "launch": {
                "model_name": "llama",
                "model_uid": "uid-1",
                "model_type": "LLM",
                "replica": 2,
            },
        }
    ]
    supervisor._launch_history_store.list_autostart.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_autostart_ready_model_resets_retry_attempts():
    supervisor = _DummyAutostartRunner(model_status="READY", attempts=3)
    entry = {
        "max_retries": 3,
        "retry_interval_seconds": 30,
        "launch": {"model_name": "llama", "model_uid": "uid-1"},
    }

    retry_delay = await supervisor._autostart_one_model(entry)

    assert retry_delay is None
    assert supervisor.launched == []
    assert supervisor._autostart_model_states["uid-1"] == {
        "attempts": 0,
        "status": "active",
        "message": "Model is already active.",
        "last_error": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_status", ["CREATING", "LOADING", "UPDATING", "TERMINATING"]
)
async def test_autostart_transitional_model_preserves_retry_attempts(model_status):
    supervisor = _DummyAutostartRunner(model_status=model_status, attempts=2)
    entry = {
        "max_retries": 3,
        "retry_interval_seconds": 30,
        "launch": {"model_name": "llama", "model_uid": "uid-1"},
    }

    retry_delay = await supervisor._autostart_one_model(entry)

    assert retry_delay == 30
    assert supervisor.launched == []
    assert supervisor._autostart_model_states["uid-1"] == {
        "attempts": 2,
        "status": "waiting_model",
        "message": f"Model is {model_status.lower()}.",
        "last_error": "previous failure",
    }


class _DummyReplicaDeathSupervisor:
    mark_replica_dead = SupervisorActor.mark_replica_dead
    _get_model_uid_and_replica_index = staticmethod(
        SupervisorActor._get_model_uid_and_replica_index
    )
    _refresh_replica_scheduler = staticmethod(
        SupervisorActor._refresh_replica_scheduler
    )

    def __init__(self, remaining_after_evict: int):
        self._unexpected_down_replicas: dict = {}
        self._replica_model_uid_to_worker: dict = {"uid-1-rep0": object()}
        self._replica_model_uid_to_worker_shards: dict = {}
        self._model_uid_to_replica_info = {
            "uid-1": ReplicaInfo(replica=1, scheduler=iter([]), active_replica_ids=[0])
        }
        self._status_guard_ref = MagicMock()
        self._status_guard_ref.get_instance_info = AsyncMock(return_value=[])
        self._status_guard_ref.remove_replica_status = AsyncMock(
            return_value=remaining_after_evict
        )
        self._status_guard_ref.update_instance_info = AsyncMock()
        self.autostart_scheduled = False

    def _schedule_autostart(self, delay: float = 0.0):
        self.autostart_scheduled = True

    async def _cleanup_distributed_actors(self, base_uid, terminate_rank0_on_worker):
        return None


@pytest.mark.asyncio
async def test_mark_replica_dead_reschedules_autostart_on_last_replica():
    # A dead last replica must be handed back to Autostart, not left
    # TERMINATED forever.
    supervisor = _DummyReplicaDeathSupervisor(remaining_after_evict=0)

    await supervisor.mark_replica_dead("uid-1-rep0")

    assert supervisor.autostart_scheduled is True
    supervisor._status_guard_ref.update_instance_info.assert_called_once_with(
        "uid-1", {"status": "TERMINATED"}
    )


@pytest.mark.asyncio
async def test_mark_replica_dead_does_not_reschedule_autostart_when_degraded():
    # A healthy replica remains after eviction: the model stays READY and
    # Autostart, whose job is only to relaunch a fully-dead model, must not
    # be woken.
    supervisor = _DummyReplicaDeathSupervisor(remaining_after_evict=1)

    await supervisor.mark_replica_dead("uid-1-rep0")

    assert supervisor.autostart_scheduled is False


@pytest.mark.asyncio
async def test_autostart_successful_launch_resets_retry_attempts():
    supervisor = _DummyAutostartRunner(model_status=None, attempts=2)
    launch = {"model_name": "llama", "model_uid": "uid-1"}
    entry = {
        "max_retries": 3,
        "retry_interval_seconds": 30,
        "launch": launch,
    }

    retry_delay = await supervisor._autostart_one_model(entry)

    assert retry_delay is None
    assert supervisor.launched == [launch]
    state = supervisor._autostart_model_states["uid-1"]
    assert state["attempts"] == 0
    assert state["status"] == "active"
    assert state["model_uid"] == "uid-1"
    assert state["message"] == "Model is ready."
    assert state["last_error"] is None
    assert isinstance(state["last_attempt_ts"], int)
    assert isinstance(state["last_started_ts"], int)
