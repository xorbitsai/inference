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

from ..actor.gradio import GradioApp
from ..actor.service import ControllerActor, WorkerActor


def run():
    async def start_gradio():
        pool = await xo.create_actor_pool(address="127.0.0.1", n_process=4)
        controller = await xo.create_actor(
            ControllerActor, address=pool.external_address, uid=ControllerActor.uid()
        )
        await xo.create_actor(
            WorkerActor,
            address=pool.external_address,
            uid=WorkerActor.uid(),
            controller_address=pool.external_address,
        )
        await controller.launch_builtin_model("x1", "vicuna-uncensored")
        await controller.launch_builtin_model("x2", "wizardlm")
        app = GradioApp(xoscar_endpoint=pool.external_address)
        demo = app.build()
        demo.queue(concurrency_count=20)
        demo.launch()

    loop = asyncio.get_event_loop()
    task = loop.create_task(start_gradio())
    try:
        loop.run_until_complete(start_gradio())
    except KeyboardInterrupt:
        task.cancel()


if __name__ == "__main__":
    run()
