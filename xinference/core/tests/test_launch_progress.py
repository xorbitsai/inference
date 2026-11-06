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

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ..progress_tracker import Progressor
from ..rpc_context import rpc_context
from ..worker import WorkerActor


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_worker_progress_waits_for_engine_and_state_update(fail):
    loading = asyncio.Event()
    ready = asyncio.Event()

    async def load():
        loading.set()
        await ready.wait()
        if fail:
            raise RuntimeError("CUDA out of memory")

    progressor = SimpleNamespace(finish_launch=AsyncMock())
    worker = SimpleNamespace(
        _model_uid_to_model={"model": SimpleNamespace(wait_for_load=load)},
        _model_uid_to_model_status={"model": "loading"},
        _launch_progressors={"model": progressor},
        _update_model_state=AsyncMock(),
        terminate_model=AsyncMock(),
    )
    worker._clean_up_failed_launch = (
        lambda uid, ref: WorkerActor._clean_up_failed_launch(worker, uid, ref)
    )
    task = asyncio.create_task(WorkerActor.wait_for_load(worker, "model"))
    await asyncio.wait_for(loading.wait(), timeout=2)
    progressor.finish_launch.assert_not_awaited()
    worker._update_model_state.assert_not_awaited()
    ready.set()
    if fail:
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            await task
        worker.terminate_model.assert_awaited_once_with("model", is_model_die=True)
        worker._update_model_state.assert_not_awaited()
        progressor.finish_launch.assert_awaited_once_with(False)
    else:
        await task
        worker._update_model_state.assert_awaited_once_with("model", "ready")
        progressor.finish_launch.assert_awaited_once_with(True)
    assert worker._launch_progressors == {}


@pytest.mark.asyncio
async def test_worker_progress_is_best_effort_and_ignores_replaced_generation():
    old = SimpleNamespace(finish_launch=AsyncMock())
    current = SimpleNamespace(
        finish_launch=AsyncMock(side_effect=RuntimeError("RPC lost"))
    )
    worker = SimpleNamespace(_launch_progressors={"model": current})
    await WorkerActor._finish_launch_progress(worker, "model", True, old)
    current.finish_launch.assert_not_awaited()
    assert worker._launch_progressors["model"] is current
    await WorkerActor._finish_launch_progress(worker, "model", True, current)
    current.finish_launch.assert_awaited_once_with(True)
    assert not worker._launch_progressors


@pytest.mark.asyncio
async def test_new_worker_can_report_to_legacy_tracker():
    class LegacyTracker:
        def __init__(self):
            self.progress = 0

        @rpc_context
        async def start(self, request_id):
            self.progress = 0

        @rpc_context
        async def set_progress(self, request_id, progress, info=None, details=None):
            self.progress = progress

    tracker = LegacyTracker()
    progressor = Progressor(
        "launch", tracker, asyncio.get_running_loop(), generation="new"
    )
    await progressor.start()
    with progressor:
        pass
    await asyncio.sleep(0.01)
    assert tracker.progress == 0.99
    await progressor.finish_launch(True)
    assert tracker.progress == 1
