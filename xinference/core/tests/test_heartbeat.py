# Copyright 2022-2026 XProbe Inc.
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

import time

import pytest

from xinference.core.resource import GPUStatus, ResourceStatus
from xinference.core.supervisor import SupervisorActor, WorkerStatus


def _build_worker_status(*, gpu_utils=None):
    if gpu_utils is None:
        gpu_utils = []

    from typing import Dict, Union

    status: Dict[str, Union[ResourceStatus, GPUStatus]] = {
        "cpu": ResourceStatus(
            usage=0.25,
            total=32,
            memory_used=128,
            memory_available=384,
            memory_total=512,
        )
    }
    for idx, gpu_util in enumerate(gpu_utils):
        status[f"gpu-{idx}"] = GPUStatus(
            name=f"GPU-{idx}",
            mem_total=1000,
            mem_free=400,
            mem_used=600,
            mem_usage=0.6,
            gpu_util=gpu_util,
        )
    return status


class DummySupervisorWithHeartbeat:
    receive_heartbeat = SupervisorActor.receive_heartbeat

    def __init__(self, worker_status):
        self._worker_status = worker_status


@pytest.mark.asyncio
async def test_supervisor_receive_heartbeat_new_worker():
    """Test that heartbeat creates initial status for new workers."""
    supervisor = DummySupervisorWithHeartbeat({})

    await supervisor.receive_heartbeat("new-worker-1")

    assert "new-worker-1" in supervisor._worker_status
    status = supervisor._worker_status["new-worker-1"]
    assert status.status == {}  # Empty status, waiting for full report
    assert (
        status.failure_remaining_count == 5
    )  # XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD


@pytest.mark.asyncio
async def test_supervisor_receive_heartbeat_existing_worker():
    """Test that heartbeat only updates timestamp for existing workers."""
    supervisor = DummySupervisorWithHeartbeat(
        {
            "existing-worker": WorkerStatus(
                update_time=100.0,
                failure_remaining_count=3,
                status=_build_worker_status(gpu_utils=[50]),
            ),
        }
    )

    original_status = supervisor._worker_status["existing-worker"].status.copy()
    await supervisor.receive_heartbeat("existing-worker")

    # Timestamp should be updated
    assert supervisor._worker_status["existing-worker"].update_time > 100.0
    # Status should remain unchanged
    assert supervisor._worker_status["existing-worker"].status == original_status
    assert supervisor._worker_status["existing-worker"].failure_remaining_count == 3


@pytest.mark.asyncio
async def test_supervisor_receive_heartbeat_multiple_calls():
    """Test that multiple heartbeat calls only update timestamp."""
    supervisor = DummySupervisorWithHeartbeat({})

    await supervisor.receive_heartbeat("worker-1")
    first_time = supervisor._worker_status["worker-1"].update_time

    time.sleep(0.01)
    await supervisor.receive_heartbeat("worker-1")
    second_time = supervisor._worker_status["worker-1"].update_time

    assert second_time > first_time
    # Status should still be empty
    assert supervisor._worker_status["worker-1"].status == {}
