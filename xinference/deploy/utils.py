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

import json
import logging
import os
import time
import typing
from typing import TYPE_CHECKING, Any, Optional

import xoscar as xo

from ..constants import XINFERENCE_DEFAULT_LOG_FILE_NAME, XINFERENCE_LOG_DIR

if TYPE_CHECKING:
    from xoscar.backends.pool import MainActorPoolType

logger = logging.getLogger(__name__)

# mainly for k8s
XINFERENCE_POD_NAME_ENV_KEY = "XINFERENCE_POD_NAME"


class LoggerNameFilter(logging.Filter):
    def filter(self, record):
        return record.name.startswith("xinference") or (
            record.name.startswith("uvicorn.error")
            and record.getMessage().startswith("Uvicorn running on")
        )


def get_log_file(sub_dir: str):
    """
    sub_dir should contain a timestamp.
    """
    pod_name = os.environ.get(XINFERENCE_POD_NAME_ENV_KEY, None)
    if pod_name is not None:
        sub_dir = sub_dir + "_" + pod_name
    log_dir = os.path.join(XINFERENCE_LOG_DIR, sub_dir)
    # Here should be creating a new directory each time, so `exist_ok=False`
    os.makedirs(log_dir, exist_ok=False)
    return os.path.join(log_dir, XINFERENCE_DEFAULT_LOG_FILE_NAME)


def get_config_dict(
    log_level: str, log_file_path: str, log_backup_count: int, log_max_bytes: int
) -> dict:
    # for windows, the path should be a raw string.
    log_file_path = (
        log_file_path.encode("unicode-escape").decode()
        if os.name == "nt"
        else log_file_path
    )
    log_level = log_level.upper()
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "formatter": {
                "format": (
                    "%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s"
                )
            },
        },
        "filters": {
            "logger_name_filter": {
                "()": __name__ + ".LoggerNameFilter",
            },
        },
        "handlers": {
            "stream_handler": {
                "class": "logging.StreamHandler",
                "formatter": "formatter",
                "level": log_level,
                "stream": "ext://sys.stderr",
                "filters": ["logger_name_filter"],
            },
            "console_handler": {
                "class": "logging.StreamHandler",
                "formatter": "formatter",
                "level": log_level,
                "stream": "ext://sys.stderr",
            },
            "file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "formatter",
                "level": log_level,
                "filename": log_file_path,
                "mode": "a",
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "xinference": {
                "handlers": ["stream_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["stream_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["stream_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["stream_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            },
            "transformers": {
                "handlers": ["console_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            },
            "vllm": {
                "handlers": ["console_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            },
        },
    }
    return config_dict


async def create_worker_actor_pool(
    address: str, logging_conf: Optional[dict] = None
) -> "MainActorPoolType":
    return await xo.create_actor_pool(
        address=address,
        n_process=0,
        auto_recover="process",
        logging_conf={"dict": logging_conf},
    )


def health_check(address: str, max_attempts: int, sleep_interval: int = 3) -> bool:
    async def health_check_internal():
        import time

        attempts = 0
        while attempts < max_attempts:
            time.sleep(sleep_interval)
            try:
                from ..core.supervisor import SupervisorActor

                supervisor_ref: xo.ActorRefType[SupervisorActor] = await xo.actor_ref(  # type: ignore
                    address=address, uid=SupervisorActor.default_uid()
                )

                await supervisor_ref.get_status()
                return True
            except Exception as e:
                logger.debug(f"Error while checking cluster: {e}")

            attempts += 1
            if attempts < max_attempts:
                logger.debug(
                    f"Cluster not available, will try {max_attempts - attempts} more times"
                )

        return False

    import asyncio

    from ..isolation import Isolation

    isolation = Isolation(asyncio.new_event_loop(), threaded=True)
    isolation.start()
    available = isolation.call(health_check_internal())
    isolation.stop()
    return available


def get_timestamp_ms():
    t = time.time()
    return int(round(t * 1000))


@typing.no_type_check
def handle_click_args_type(arg: str) -> Any:
    """Convert CLI string arguments to appropriate Python types.

    Handles type conversion for Click command arguments with the following priority:
    1. Special string values: "None" → None, "true"/"True" → True, "false"/"False" → False
    2. Numeric values: "42" → int(42), "3.14" → float(3.14)
    3. JSON objects/arrays: '{"key": "value"}' → dict, '[1, 2, 3]' → list
    4. Default: return original string

    JSON parsing is attempted for arguments starting with '{' or '['.
    If JSON parsing fails, the original string is returned unchanged.

    Examples:
        >>> handle_click_args_type("None")
        None
        >>> handle_click_args_type("42")
        42
        >>> handle_click_args_type('{"mode": 3}')
        {'mode': 3}

    Args:
        arg: String argument from CLI command

    Returns:
        Converted value (None, bool, int, float, dict, list, or str)
    """
    if arg == "None":
        return None
    if arg in ("True", "true"):
        return True
    if arg in ("False", "false"):
        return False
    try:
        result = int(arg)
        return result
    except Exception:
        pass

    try:
        result = float(arg)
        return result
    except Exception:
        pass

    # Try to parse JSON objects and arrays.
    # This allows CLI users to pass structured parameters like:
    #   --compilation_config '{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}'
    #   --kv_transfer_config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
    # See: https://github.com/xorbitsai/inference/issues/4760
    if arg.startswith(("{", "[")):
        try:
            result = json.loads(arg)
            return result
        except (json.JSONDecodeError, ValueError):
            pass

    return arg


def set_envs(key: str, value: str):
    """
    Environment variables are set by the parent process and inherited by child processes
    """
    os.environ[key] = value
