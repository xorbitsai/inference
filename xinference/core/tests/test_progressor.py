# Copyright 2022-2023 XProbe Inc.
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
import uuid

import pytest
import xoscar as xo

from ..progress_tracker import Progressor, ProgressTrackerActor


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
