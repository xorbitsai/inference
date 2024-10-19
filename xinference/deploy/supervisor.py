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

from ..constants import (
    XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
    XINFERENCE_HEALTH_CHECK_INTERVAL,
)
from ..core.supervisor import SupervisorActor
from .utils import health_check

logger = logging.getLogger(__name__)


async def _start_supervisor(address: str, logging_conf: Optional[Dict] = None):
    logging.config.dictConfig(logging_conf)  # type: ignore

    pool = None
    try:
        pool = await xo.create_actor_pool(
            address=address, n_process=0, logging_conf={"dict": logging_conf}
        )
        await xo.create_actor(
            SupervisorActor, address=address, uid=SupervisorActor.default_uid()
        )
        await pool.join()
    except asyncio.exceptions.CancelledError:
        if pool is not None:
            await pool.stop()


def run(address: str, logging_conf: Optional[Dict] = None):
    def sigterm_handler(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    loop = asyncio.get_event_loop()
    task = loop.create_task(
        _start_supervisor(address=address, logging_conf=logging_conf)
    )
    loop.run_until_complete(task)


def run_in_subprocess(
    address: str, logging_conf: Optional[Dict] = None
) -> multiprocessing.Process:
    p = multiprocessing.Process(target=run, args=(address, logging_conf))
    p.start()
    return p


def main(
    host: str,
    port: int,
    supervisor_port: Optional[int],
    logging_conf: Optional[Dict] = None,
    auth_config_file: Optional[str] = None,
):
    supervisor_address = f"{host}:{supervisor_port or get_next_port()}"
    local_cluster = run_in_subprocess(supervisor_address, logging_conf)

    if not health_check(
        address=supervisor_address,
        max_attempts=XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
        sleep_interval=XINFERENCE_HEALTH_CHECK_INTERVAL,
    ):
        raise RuntimeError("Supervisor is not available after multiple attempts")

    try:
        from ..api import restful_api

        restful_api.run(
            supervisor_address=supervisor_address,
            host=host,
            port=port,
            logging_conf=logging_conf,
            auth_config_file=auth_config_file,
        )
    finally:
        local_cluster.kill()
