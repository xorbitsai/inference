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

from .supervisor import start_supervisor_components
from .worker import start_worker_components


async def _start_local_cluster(
    address: str,
    model_name: str,
    size_in_billions: int,
    model_format: str,
    quantization: str,
    host: str,
    port: int,
):
    from .utils import create_actor_pool

    pool = None
    try:
        pool = await create_actor_pool(address=address, n_process=0)
        await start_supervisor_components(address, host, port)
        await start_worker_components(address=address, supervisor_address=address)

        await pool.join()
    except asyncio.CancelledError:
        if pool is not None:
            await pool.stop()


def main(*args, **kwargs):
    loop = asyncio.get_event_loop()
    task = loop.create_task(_start_local_cluster(*args, **kwargs))

    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        task.cancel()
        loop.run_until_complete(task)
        # avoid displaying exception-unhandled warnings
        task.exception()
