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

import asyncio
import logging
import multiprocessing
import signal
import sys
import traceback
from multiprocessing.connection import Connection
from typing import Dict, Optional

import xoscar as xo
from xoscar.utils import get_next_port

from ..constants import (
    XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
    XINFERENCE_HEALTH_CHECK_INTERVAL,
    XINFERENCE_HEALTH_CHECK_TIMEOUT,
)
from ..core.supervisor import SupervisorActor
from .utils import health_check
from .worker import start_worker_components

logger = logging.getLogger(__name__)


READY = "ok"


async def _start_local_cluster(
    address: str,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[Dict] = None,
    conn: Optional[Connection] = None,
):
    from .utils import create_worker_actor_pool

    if logging_conf:
        logging.config.dictConfig(logging_conf)  # type: ignore

    pool = None
    try:
        pool = await create_worker_actor_pool(
            address=address, logging_conf=logging_conf
        )
        await xo.create_actor(
            SupervisorActor, address=address, uid=SupervisorActor.default_uid()
        )
        await start_worker_components(
            address=address,
            supervisor_address=address,
            main_pool=pool,
            metrics_exporter_host=metrics_exporter_host,
            metrics_exporter_port=metrics_exporter_port,
        )
        if conn:
            try:
                conn.send(READY)
            except BrokenPipeError:
                # connection may be gc collected,
                # just ignore this error
                pass
        await pool.join()
    except asyncio.CancelledError:
        if pool is not None:
            await pool.stop()


def run(
    address: str,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[Dict] = None,
    conn: Optional[Connection] = None,
):
    def sigterm_handler(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        loop = asyncio.get_event_loop()
        task = loop.create_task(
            _start_local_cluster(
                address=address,
                metrics_exporter_host=metrics_exporter_host,
                metrics_exporter_port=metrics_exporter_port,
                logging_conf=logging_conf,
                conn=conn,
            )
        )
        loop.run_until_complete(task)
    except:
        tb = traceback.format_exc()
        if conn:
            try:
                conn.send(f"error: {tb}")
            except BrokenPipeError:
                # connection may be gc collected,
                # just ignore this error
                pass
        # raise again in subprocess
        raise


def run_in_subprocess(
    address: str,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[Dict] = None,
) -> multiprocessing.Process:
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(
        target=run,
        args=(address, metrics_exporter_host, metrics_exporter_port, logging_conf),
        kwargs={"conn": child_conn},
    )
    # Since Xoscar 0.7, we do not uses multiprocessing to create subpool any more,
    # we should be able to use daemon here
    p.daemon = True
    p.start()
    if parent_conn.poll(timeout=XINFERENCE_HEALTH_CHECK_TIMEOUT):
        msg = parent_conn.recv()
        if msg != READY:
            raise RuntimeError(
                f"Start service process failed during startup:\n{msg}"  # noqa: E231
            )
    else:
        logger.info(
            "No response from process after %s seconds", XINFERENCE_HEALTH_CHECK_TIMEOUT
        )

    return p


def main(
    host: str,
    port: int,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[Dict] = None,
    auth_config_file: Optional[str] = None,
):
    # force to set spawn,
    # cuda may be inited in xoscar virtualenv
    # which will raise error after sub pool is created
    multiprocessing.set_start_method("spawn")

    supervisor_address = f"{host}:{get_next_port()}"  # noqa: E231
    local_cluster = run_in_subprocess(
        supervisor_address, metrics_exporter_host, metrics_exporter_port, logging_conf
    )

    if not health_check(
        address=supervisor_address,
        max_attempts=XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
        sleep_interval=XINFERENCE_HEALTH_CHECK_INTERVAL,
    ):
        raise RuntimeError("Cluster is not available after multiple attempts")

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
