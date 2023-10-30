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

import logging

import pytest_asyncio
import xoscar as xo

from xinference.core.supervisor import SupervisorActor

TEST_LOGGING_CONF = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "formatter": {
            "format": "%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s",
        },
    },
    "handlers": {
        "stream_handler": {
            "class": "logging.StreamHandler",
            "formatter": "formatter",
            "level": "DEBUG",
            "stream": "ext://sys.stderr",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["stream_handler"],
    },
}


def health_check(endpoint: str, max_attempts: int, sleep_interval: int = 3):
    import time

    import requests

    attempts = 0
    while attempts < max_attempts:
        time.sleep(sleep_interval)
        try:
            response = requests.get(f"{endpoint}/status")
            if response.status_code == 200:
                return True
        except requests.RequestException as e:
            print(f"Error while checking endpoint: {e}")

        attempts += 1
        if attempts < max_attempts:
            print(
                f"Endpoint not available, will try {max_attempts - attempts} more times"
            )

    return False


@pytest_asyncio.fixture
async def setup():
    from .api.restful_api import run_in_subprocess as run_restful_api
    from .deploy.utils import create_worker_actor_pool
    from .deploy.worker import start_worker_components

    logging.config.dictConfig(TEST_LOGGING_CONF)  # type: ignore

    pool = await create_worker_actor_pool(
        address=f"test://127.0.0.1:{xo.utils.get_next_port()}",
        logging_conf=TEST_LOGGING_CONF,
    )
    print(f"Pool running on localhost:{pool.external_address}")

    await xo.create_actor(
        SupervisorActor, address=pool.external_address, uid=SupervisorActor.uid()
    )
    await start_worker_components(
        address=pool.external_address,
        supervisor_address=pool.external_address,
        main_pool=pool,
    )
    print("Supervisor and worker has been started successfully")

    port = xo.utils.get_next_port()
    restful_api_proc = run_restful_api(
        pool.external_address,
        host="localhost",
        port=port,
        logging_conf=TEST_LOGGING_CONF,
    )
    endpoint = f"http://localhost:{port}"
    if not health_check(endpoint, max_attempts=3, sleep_interval=3):
        raise RuntimeError("Endpoint is not available after multiple attempts")

    async with pool:
        yield f"http://localhost:{port}", pool.external_address

    restful_api_proc.terminate()
