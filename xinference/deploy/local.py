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
import logging
import multiprocessing
import signal
import sys
from typing import Dict, Optional

import xoscar as xo
from xoscar.utils import get_next_port

from ..core.supervisor import SupervisorActor
from .worker import start_worker_components

logger = logging.getLogger(__name__)


async def _start_local_cluster(
    address: str,
    logging_conf: Optional[Dict] = None,
):
    from .utils import create_worker_actor_pool

    logging.config.dictConfig(logging_conf)  # type: ignore

    pool = None
    try:
        pool = await create_worker_actor_pool(
            address=address, logging_conf=logging_conf
        )
        await xo.create_actor(
            SupervisorActor, address=address, uid=SupervisorActor.uid()
        )
        await start_worker_components(
            address=address, supervisor_address=address, main_pool=pool
        )
        await pool.join()
    except asyncio.CancelledError:
        if pool is not None:
            await pool.stop()


def run(address: str, logging_conf: Optional[Dict] = None):
    def sigterm_handler(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    loop = asyncio.get_event_loop()
    task = loop.create_task(
        _start_local_cluster(address=address, logging_conf=logging_conf)
    )
    loop.run_until_complete(task)


def run_in_subprocess(
    address: str, logging_conf: Optional[Dict] = None
) -> multiprocessing.Process:
    p = multiprocessing.Process(target=run, args=(address, logging_conf))
    p.start()
    return p


def main(host: str, port: int, logging_conf: Optional[Dict] = None):
    supervisor_address = f"{host}:{get_next_port()}"
    local_cluster = run_in_subprocess(supervisor_address, logging_conf)
    # TODO: cluster health check

    try:
        from ..api import restful_api

        restful_api.run(
            supervisor_address=supervisor_address,
            host=host,
            port=port,
            logging_conf=logging_conf,
        )
    finally:
        local_cluster.terminate()
