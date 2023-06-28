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

from ..actor.gradio import GradioActor
from ..actor.service import ControllerActor, RESTAPIActor


async def start_controller_components(address: str, share: bool, host: str, port: int):
    await xo.create_actor(ControllerActor, address=address, uid=ControllerActor.uid())
    await xo.create_actor(RESTAPIActor, address=address, uid="restful", host="0.0.0.0", port=8000)
    gradio = await xo.create_actor(
        GradioActor,
        xoscar_endpoint=address,
        share=share,
        host=host,
        port=port,
        address=address,
        uid=GradioActor.default_uid(),
    )
    await gradio.launch()


async def _start_controller(address: str, host: str, port: int, share: bool):
    from .utils import create_actor_pool

    pool = await create_actor_pool(address=address, n_process=0)
    await start_controller_components(
        address=address, host=host, port=port, share=share
    )
    await pool.join()


def main(*args, **kwargs):
    loop = asyncio.get_event_loop()
    task = loop.create_task(_start_controller(*args, **kwargs))

    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        task.cancel()
        loop.run_until_complete(task)
        # avoid displaying exception-unhandled warnings
        task.exception()