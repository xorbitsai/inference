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
import socket
from typing import Dict, Optional

import xoscar as xo
from xoscar.utils import get_next_port

from ..constants import XINFERENCE_DEFAULT_ENDPOINT_PORT
from ..core.restful_api import RESTfulAPIActor
from ..core.supervisor import SupervisorActor

logger = logging.getLogger("xinference")


async def start_supervisor_components(address: str, host: str, port: int):
    await xo.create_actor(SupervisorActor, address=address, uid=SupervisorActor.uid())
    # create a socket for RESTful API
    try:
        sockets = []
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, port))
        sockets.append(sock)
    except OSError:
        # compare the reference to differentiate between the cases where the user specify the
        # default port and the user does not specify the port.
        if port is XINFERENCE_DEFAULT_ENDPOINT_PORT:
            while True:
                try:
                    sockets = []
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    port = get_next_port()
                    sock.bind((host, port))
                    sockets.append(sock)
                    break
                except OSError:
                    logger.warning("Failed to create socket with port %d", port)
        else:
            raise

    if host == "0.0.0.0":
        host = "localhost"

    restful_actor = await xo.create_actor(
        RESTfulAPIActor,
        address=address,
        uid=RESTfulAPIActor.uid(),
        sockets=sockets,
        endpoint=f"http://{host}:{port}",
    )
    await restful_actor.serve()
    url = f"http://{host}:{port}"
    logger.info(f"Xinference successfully started. Endpoint: {url}")
    return url


async def _start_supervisor(
    address: str, host: str, port: int, logging_conf: Optional[Dict] = None
):
    pool = None
    try:
        pool = await xo.create_actor_pool(
            address=address, n_process=0, logging_conf=logging_conf
        )
        await start_supervisor_components(address=address, host=host, port=port)
        await pool.join()
    except asyncio.exceptions.CancelledError:
        if pool is not None:
            await pool.stop()


def main(*args, **kwargs):
    loop = asyncio.get_event_loop()
    task = loop.create_task(_start_supervisor(*args, **kwargs))

    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        task.cancel()
        loop.run_until_complete(task)
        # avoid displaying exception-unhandled warnings
        task.exception()
