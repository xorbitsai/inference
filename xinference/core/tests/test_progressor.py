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
import uuid
from types import SimpleNamespace

import pytest
import xoscar as xo

from ..progress_tracker import Progressor, ProgressTrackerActor


@pytest.fixture
def timed_tracker(monkeypatch):
    from .. import progress_tracker

    clock = [100.0]
    monkeypatch.setattr(
        progress_tracker,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: clock[0]),
    )
    return ProgressTrackerActor(), clock


def test_generation_timing_and_output_processing(timed_tracker):
    tracker, clock = timed_tracker
    tracker.start("image")
    assert tracker.get_progress_snapshot("image") == {
        "progress": 0.0,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
    }
    clock[0] += 10
    tracker.set_progress("image", 0.25)
    assert tracker.get_progress_snapshot("image")["eta_seconds"] is None
    clock[0] += 10
    tracker.set_progress("image", 0.5)
    assert tracker.get_progress_snapshot("image") == {
        "progress": 0.5,
        "elapsed_seconds": 20.0,
        "eta_seconds": 20.0,
    }
    clock[0] += 10
    tracker.set_progress("image", 1.0)
    # Finishing the reported steps does not estimate decoding or saving time.
    assert tracker.get_progress_snapshot("image")["eta_seconds"] is None
    clock[0] += 5
    assert tracker.get_progress_snapshot("image")["elapsed_seconds"] == 35


def test_generation_eta_smoothing_stalls_and_resets(timed_tracker):
    tracker, clock = timed_tracker
    tracker.start("video")
    clock[0] += 10
    tracker.set_progress("video", 0.25)
    clock[0] += 20
    tracker.set_progress("video", 0.5)
    assert tracker.get_progress_snapshot("video")["eta_seconds"] == pytest.approx(26)
    clock[0] += 61
    assert tracker.get_progress_snapshot("video")["eta_seconds"] is None
    tracker.set_progress("video", 0.2)
    assert tracker.get_progress_snapshot("video")["eta_seconds"] is None
    clock[0] += 10
    tracker.set_progress("video", 0.3)
    assert tracker.get_progress_snapshot("video")["eta_seconds"] is None
    clock[0] += 10
    tracker.set_progress("video", 0.4)
    assert tracker.get_progress_snapshot("video")["eta_seconds"] == pytest.approx(60)


def test_generation_without_step_updates_has_no_eta_or_added_logs(
    timed_tracker, caplog
):
    tracker, clock = timed_tracker
    with caplog.at_level("INFO", logger="xinference.core.progress_tracker"):
        tracker.start("coarse")
        clock[0] += 10
        assert tracker.get_progress_snapshot("coarse") == {
            "progress": 0.0,
            "elapsed_seconds": 10.0,
            "eta_seconds": None,
        }
        tracker.set_progress("coarse", 0.5)
        assert tracker.get_progress("coarse") == 0.5
    assert not caplog.records


@pytest.mark.asyncio
async def test_finished_progress_is_cleared_after_every_check():
    tracker = ProgressTrackerActor(to_remove_interval=0, check_interval=0.01)
    task = asyncio.create_task(tracker._clear_finished())
    try:
        for request_id in ("first", "second"):
            tracker.start(request_id)
            tracker.set_progress(request_id, 1.0)
            await asyncio.sleep(0.1)
            assert request_id not in tracker._request_id_to_progress
        assert not task.done()
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_progressor():
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        progress_tracker_ref = await xo.create_actor(
            ProgressTrackerActor,
            to_remove_interval=0,
            check_interval=1,
            address=pool.external_address,
            uid=ProgressTrackerActor.default_uid(),
        )
        request_id = str(uuid.uuid4())

        progressor = Progressor(
            request_id, progress_tracker_ref, asyncio.get_running_loop(), upload_span=0
        )
        await progressor.start()

        with progressor:
            progressor.split_stages(2)

            with progressor:
                progressor.set_progress(0.5)

                await asyncio.sleep(0.1)
                assert await progress_tracker_ref.get_progress(request_id) == 0.25

            await asyncio.sleep(0.1)
            assert await progress_tracker_ref.get_progress(request_id) == 0.5

            progressor.activate_stage()
            progressor.set_progress(0.0, details={"stage": "installing_dependencies"})
            await asyncio.sleep(0.1)
            progress, _, details = await progress_tracker_ref.get_progress_details(
                request_id
            )
            assert progress == 0.5
            assert details == {"stage": "installing_dependencies"}

            with progressor:
                progressor.split_stages(2)

                with progressor:
                    progressor.set_progress(0.8)

                    await asyncio.sleep(0.1)
                    assert (
                        await progress_tracker_ref.get_progress(request_id)
                        == 0.5 + 0.25 * 0.8
                    )

                await asyncio.sleep(0.1)
                assert await progress_tracker_ref.get_progress(request_id) == 0.75

                with pytest.raises(ValueError):
                    with progressor:
                        raise ValueError

                await asyncio.sleep(0.1)
                assert await progress_tracker_ref.get_progress(request_id) == 1.0

        await asyncio.sleep(0.1)
        assert await progress_tracker_ref.get_progress(request_id) == 1.0
