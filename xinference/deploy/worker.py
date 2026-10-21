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

import asyncio
import contextlib
import logging
import multiprocessing
import os
from typing import Any, Optional

import xoscar as xo
from xoscar import MainActorPoolType

from ..core.worker import WorkerActor
from ..device_utils import get_available_device_env_name, gpu_count

logger = logging.getLogger(__name__)


async def start_worker_components(
    address: str,
    supervisor_address: str,
    supervisor_endpoint: Optional[str],
    main_pool: MainActorPoolType,
    metrics_exporter_host: Optional[str],
    metrics_exporter_port: Optional[int],
    standby: bool = False,
    relaunch_operation_id: Optional[str] = None,
    logging_conf: Optional[dict] = None,
):
    gpu_device_indices = []
    env_name = get_available_device_env_name()
    cuda_visible_devices = os.environ.get(env_name) if env_name else None

    if cuda_visible_devices and cuda_visible_devices != "-1":
        gpu_device_indices.extend([int(i) for i in cuda_visible_devices.split(",")])
    else:
        gpu_device_indices = list(range(gpu_count()))

    return await xo.create_actor(
        WorkerActor,
        address=address,
        uid=WorkerActor.default_uid(),
        supervisor_address=supervisor_address,
        supervisor_endpoint=supervisor_endpoint,
        main_pool=main_pool,
        gpu_devices=gpu_device_indices,
        metrics_exporter_host=metrics_exporter_host,
        metrics_exporter_port=metrics_exporter_port,
        standby=standby,
        relaunch_operation_id=relaunch_operation_id,
        logging_conf=logging_conf,
    )


async def _start_worker(
    address: str,
    supervisor_address: str,
    supervisor_endpoint: Optional[str] = None,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Any = None,
    replace_worker: Optional[str] = None,
    relaunch_operation_id: Optional[str] = None,
):
    from .utils import create_worker_actor_pool, update_all_formatter_addresses

    update_all_formatter_addresses("worker", address)
    if logging_conf and "formatters" in logging_conf:
        for formatter in logging_conf["formatters"].values():
            formatter["address"] = address

    pool = await create_worker_actor_pool(address=address, logging_conf=logging_conf)
    ref = await start_worker_components(
        address=address,
        supervisor_address=supervisor_address,
        supervisor_endpoint=supervisor_endpoint,
        main_pool=pool,
        metrics_exporter_host=metrics_exporter_host,
        metrics_exporter_port=metrics_exporter_port,
        standby=replace_worker is not None,
        relaunch_operation_id=relaunch_operation_id,
        logging_conf=logging_conf,
    )
    if replace_worker is not None:
        from ..core.worker_handoff import HandoffPending

        async def report(status, error=""):
            if relaunch_operation_id is not None:
                from ..core.supervisor import SupervisorActor

                supervisor = await xo.actor_ref(
                    address=supervisor_address, uid=SupervisorActor.default_uid()
                )
                await xo.wait_for(
                    supervisor.report_worker_relaunch(
                        relaunch_operation_id,
                        address,
                        await ref.worker_incarnation(),
                        status,
                        error,
                    ),
                    5,
                )

        reported_started = False
        while True:
            try:
                if not reported_started:
                    with contextlib.suppress(Exception):
                        await report("transferring")
                        reported_started = True
                await ref.replace_worker(replace_worker)
                logger.info("Worker %s has taken over from %s", address, replace_worker)
                break
            except HandoffPending as error:
                logger.exception("Worker handoff needs to finish; retrying")
                with contextlib.suppress(Exception):
                    await report("pending", str(error) or type(error).__name__)
                await asyncio.sleep(1)
            except BaseException as error:
                if await ref.worker_replacement_safe_to_stop():
                    with contextlib.suppress(Exception):
                        await report("failed", str(error) or type(error).__name__)
                    await pool.stop()
                    raise
                logger.exception(
                    "Transfer outcome needs confirmation; keeping replacement alive"
                )
                with contextlib.suppress(Exception):
                    await report("pending", str(error) or type(error).__name__)
                await asyncio.sleep(1)
        with contextlib.suppress(Exception):
            await report("succeeded")
    await pool.join()


def main(
    address: str,
    supervisor_address: str,
    supervisor_endpoint: Optional[str] = None,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[dict] = None,
    replace_worker: Optional[str] = None,
):
    # force to set spawn,
    # cuda may be inited in xoscar virtualenv
    # which will raise error after sub pool is created
    multiprocessing.set_start_method("spawn")

    try:
        asyncio.run(
            _start_worker(
                address,
                supervisor_address,
                supervisor_endpoint,
                metrics_exporter_host,
                metrics_exporter_port,
                logging_conf,
                **({"replace_worker": replace_worker} if replace_worker else {}),
            )
        )
    except KeyboardInterrupt:
        pass
