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
import signal
import sys
from typing import Dict, Optional

import pytest
import xoscar as xo

# skip health checking for CI
if os.environ.get("GITHUB_ACTIONS"):
    os.environ["XINFERENCE_DISABLE_HEALTH_CHECK"] = "1"

from .constants import XINFERENCE_LOG_BACKUP_COUNT, XINFERENCE_LOG_MAX_BYTES
from .core.supervisor import SupervisorActor
from .deploy.utils import create_worker_actor_pool, get_log_file, get_timestamp_ms
from .deploy.worker import start_worker_components

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
    "loggers": {
        "xinference": {
            "handlers": ["stream_handler"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
}

TEST_LOG_FILE_PATH = get_log_file(f"test_{get_timestamp_ms()}")
if os.name == "nt":
    TEST_LOG_FILE_PATH = TEST_LOG_FILE_PATH.encode("unicode-escape").decode()


TEST_FILE_LOGGING_CONF = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "formatter": {
            "format": "%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s"
        },
    },
    "handlers": {
        "stream_handler": {
            "class": "logging.StreamHandler",
            "formatter": "formatter",
            "level": "DEBUG",
            "stream": "ext://sys.stderr",
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "formatter",
            "level": "DEBUG",
            "filename": TEST_LOG_FILE_PATH,
            "mode": "a",
            "maxBytes": XINFERENCE_LOG_MAX_BYTES,
            "backupCount": XINFERENCE_LOG_BACKUP_COUNT,
            "encoding": "utf8",
        },
    },
    "loggers": {
        "xinference": {
            "handlers": ["stream_handler", "file_handler"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
}


def api_health_check(endpoint: str, max_attempts: int, sleep_interval: int = 3):
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


async def _start_test_cluster(
    address: str,
    logging_conf: Optional[Dict] = None,
):
    logging.config.dictConfig(logging_conf)  # type: ignore
    pool = None
    try:
        pool = await create_worker_actor_pool(
            address=f"test://{address}", logging_conf=logging_conf
        )
        await xo.create_actor(
            SupervisorActor, address=address, uid=SupervisorActor.default_uid()
        )
        await start_worker_components(
            address=address,
            supervisor_address=address,
            supervisor_endpoint=None,
            main_pool=pool,
            metrics_exporter_host=None,
            metrics_exporter_port=None,
        )
        await pool.join()
    except asyncio.CancelledError:
        if pool is not None:
            await pool.stop()


def run_test_cluster(address: str, logging_conf: Optional[Dict] = None):
    def sigterm_handler(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    loop = asyncio.get_event_loop()
    task = loop.create_task(
        _start_test_cluster(address=address, logging_conf=logging_conf)
    )
    loop.run_until_complete(task)


def run_test_cluster_in_subprocess(
    address: str, logging_conf: Optional[Dict] = None
) -> multiprocessing.Process:
    # prevent re-init cuda error.
    multiprocessing.set_start_method(method="spawn", force=True)

    p = multiprocessing.Process(target=run_test_cluster, args=(address, logging_conf))
    p.start()
    return p


@contextlib.contextmanager
def _cluster_and_api(logging_conf):
    """Start the test cluster and RESTful API subprocesses.

    Hardened for slow CI runners: a subprocess that dies on startup (e.g.
    get_next_port() releases the probed port, so another process can grab
    it before the server binds) is respawned on a fresh port instead of
    burning the whole health-check window on a dead process. All spawned
    subprocesses are killed on failure paths too, so a failed fixture does
    not leak processes that disturb subsequent tests.
    """
    from .api.restful_api import run_in_subprocess as run_restful_api
    from .deploy.utils import health_check as cluster_health_check

    logging.config.dictConfig(logging_conf)  # type: ignore

    # This fixture is used by tests that exercise unauthenticated requests;
    # advanced auth defaults to on, so it must be explicitly disabled here,
    # before any subprocess (which inherits this env) is started.
    os.environ["XINFERENCE_AUTH_ADVANCED"] = "false"

    local_cluster_proc = None
    restful_api_proc = None
    try:
        for spawn_attempt in range(2):
            supervisor_addr = f"localhost:{xo.utils.get_next_port()}"
            local_cluster_proc = run_test_cluster_in_subprocess(
                supervisor_addr, logging_conf
            )
            # 20 attempts: the cluster subprocess uses the spawn start method,
            # and the cold re-import of xinference alone can take tens of
            # seconds on a loaded Windows runner.
            if cluster_health_check(supervisor_addr, max_attempts=20, sleep_interval=5):
                break
            if local_cluster_proc.exitcode is not None:
                print(
                    f"Test cluster subprocess exited with code "
                    f"{local_cluster_proc.exitcode} before becoming healthy, "
                    f"respawning (attempt {spawn_attempt + 1}/2)"
                )
                continue
            raise RuntimeError("Cluster is not available after multiple attempts")
        else:
            raise RuntimeError(
                "Test cluster subprocess kept dying on startup, last exit "
                f"code: {local_cluster_proc.exitcode}"
            )

        for spawn_attempt in range(3):
            port = xo.utils.get_next_port()
            restful_api_proc = run_restful_api(
                supervisor_addr,
                host="localhost",
                port=port,
                logging_conf=logging_conf,
            )
            endpoint = f"http://localhost:{port}"
            if api_health_check(endpoint, max_attempts=10, sleep_interval=5):
                break
            if restful_api_proc.exitcode is not None:
                print(
                    f"RESTful API subprocess exited with code "
                    f"{restful_api_proc.exitcode} before becoming healthy, "
                    f"respawning on a new port (attempt {spawn_attempt + 1}/3)"
                )
                continue
            raise RuntimeError("Endpoint is not available after multiple attempts")
        else:
            raise RuntimeError(
                "RESTful API subprocess kept dying on startup, last exit "
                f"code: {restful_api_proc.exitcode}"
            )

        yield endpoint, supervisor_addr
    finally:
        if local_cluster_proc is not None:
            local_cluster_proc.kill()
        if restful_api_proc is not None:
            restful_api_proc.kill()


@pytest.fixture
def setup():
    with _cluster_and_api(TEST_LOGGING_CONF) as (endpoint, supervisor_addr):
        yield endpoint, supervisor_addr


@pytest.fixture
def setup_with_file_logging():
    with _cluster_and_api(TEST_FILE_LOGGING_CONF) as (endpoint, supervisor_addr):
        yield endpoint, supervisor_addr, TEST_LOG_FILE_PATH
