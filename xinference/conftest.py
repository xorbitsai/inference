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
import logging
import multiprocessing
import os
import signal
import socket
import sys
from contextlib import ExitStack
from typing import Dict, Iterator, Optional, Tuple

import pytest
import xoscar as xo

# skip health checking for CI
if os.environ.get("GITHUB_ACTIONS"):
    os.environ["XINFERENCE_DISABLE_HEALTH_CHECK"] = "1"

from .constants import XINFERENCE_LOG_BACKUP_COUNT, XINFERENCE_LOG_MAX_BYTES
from .core.supervisor import SupervisorActor
from .deploy.utils import create_worker_actor_pool, get_log_file, get_timestamp_ms
from .deploy.worker import start_worker_components

# Vendored modules may be named test_*.py without being Xinference tests.
# Do not import optional third-party frameworks during test collection.
collect_ignore = ["thirdparty"]


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
    use_test_pool: bool = True,
):
    logging.config.dictConfig(logging_conf)  # type: ignore
    pool = None
    try:
        pool = await create_worker_actor_pool(
            address=f"test://{address}" if use_test_pool else address,
            logging_conf=logging_conf,
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


def run_test_cluster(
    address: str,
    logging_conf: Optional[Dict] = None,
    use_test_pool: bool = True,
):
    def sigterm_handler(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    asyncio.run(
        _start_test_cluster(
            address=address, logging_conf=logging_conf, use_test_pool=use_test_pool
        )
    )


def run_test_cluster_in_subprocess(
    address: str,
    logging_conf: Optional[Dict] = None,
    use_test_pool: bool = True,
) -> multiprocessing.Process:
    # prevent re-init cuda error.
    multiprocessing.set_start_method(method="spawn", force=True)

    p = multiprocessing.Process(
        target=run_test_cluster, args=(address, logging_conf, use_test_pool)
    )
    p.start()
    return p


def _get_test_port() -> int:
    # Let the OS choose a bindable port, including on Windows where netstat
    # does not list excluded/reserved port ranges. Use the same IPv4 interface
    # for port selection and both test servers.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _stop_test_process(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.kill()
    process.join(timeout=10)


def _setup_cluster(use_test_pool: bool = True) -> Iterator[Tuple[str, str]]:
    from .api.restful_api import run_in_subprocess as run_restful_api
    from .deploy.utils import health_check as cluster_health_check

    logging.config.dictConfig(TEST_LOGGING_CONF)  # type: ignore

    # This fixture is used by tests that exercise unauthenticated requests;
    # advanced auth defaults to on, so it must be explicitly disabled here,
    # before any subprocess (which inherits this env) is started.
    os.environ["XINFERENCE_AUTH_ADVANCED"] = "false"

    with ExitStack() as cleanup:
        supervisor_addr = f"127.0.0.1:{_get_test_port()}"
        local_cluster_proc = run_test_cluster_in_subprocess(
            supervisor_addr, TEST_LOGGING_CONF, use_test_pool
        )
        cleanup.callback(_stop_test_process, local_cluster_proc)
        if not cluster_health_check(supervisor_addr, max_attempts=10, sleep_interval=5):
            raise RuntimeError("Cluster is not available after multiple attempts")

        port = _get_test_port()
        restful_api_proc = run_restful_api(
            supervisor_addr,
            host="127.0.0.1",
            port=port,
            logging_conf=TEST_LOGGING_CONF,
        )
        cleanup.callback(_stop_test_process, restful_api_proc)
        endpoint = f"http://127.0.0.1:{port}"
        if not api_health_check(endpoint, max_attempts=10, sleep_interval=5):
            raise RuntimeError("Endpoint is not available after multiple attempts")

        yield f"http://127.0.0.1:{port}", supervisor_addr


@pytest.fixture
def setup():
    yield from _setup_cluster()


@pytest.fixture
def setup_real_actor_pool() -> Iterator[Tuple[str, str]]:
    # The in-process test pool ignores start_python and cannot exercise
    # models whose virtualenv dependencies differ from the host environment.
    yield from _setup_cluster(use_test_pool=False)


@pytest.fixture
def setup_with_file_logging():
    from .api.restful_api import run_in_subprocess as run_restful_api
    from .deploy.utils import health_check as cluster_health_check

    logging.config.dictConfig(TEST_FILE_LOGGING_CONF)  # type: ignore

    # This fixture is used by tests that exercise unauthenticated requests;
    # advanced auth defaults to on, so it must be explicitly disabled here,
    # before any subprocess (which inherits this env) is started.
    os.environ["XINFERENCE_AUTH_ADVANCED"] = "false"

    with ExitStack() as cleanup:
        supervisor_addr = f"127.0.0.1:{_get_test_port()}"
        local_cluster_proc = run_test_cluster_in_subprocess(
            supervisor_addr, TEST_FILE_LOGGING_CONF
        )
        cleanup.callback(_stop_test_process, local_cluster_proc)
        if not cluster_health_check(supervisor_addr, max_attempts=10, sleep_interval=5):
            raise RuntimeError("Cluster is not available after multiple attempts")

        port = _get_test_port()
        restful_api_proc = run_restful_api(
            supervisor_addr,
            host="127.0.0.1",
            port=port,
            logging_conf=TEST_FILE_LOGGING_CONF,
        )
        cleanup.callback(_stop_test_process, restful_api_proc)
        endpoint = f"http://127.0.0.1:{port}"
        if not api_health_check(endpoint, max_attempts=10, sleep_interval=5):
            raise RuntimeError("Endpoint is not available after multiple attempts")

        yield f"http://127.0.0.1:{port}", supervisor_addr, TEST_LOG_FILE_PATH
