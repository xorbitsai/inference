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

import xoscar as xo

from ..actor.service import ControllerActor, RESTAPIActor


async def _start_controller(address: str):
    pool = await xo.create_actor_pool(address=address, n_process=0)
    controller_ref = await xo.create_actor(ControllerActor, address=address, uid=ControllerActor.uid())
    rest_ref = await xo.create_actor(RESTAPIActor, address=address, uid="restful", addr = "0.0.0.0:8000")
    # TODO: start Gradio actor
    await pool.join()


def main(address: str):
    loop = asyncio.get_event_loop()
    task = loop.create_task(_start_controller(address))

    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        task.cancel()
        loop.run_until_complete(task)
        # avoid displaying exception-unhandled warnings
        task.exception()
