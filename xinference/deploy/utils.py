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
import os
from typing import TYPE_CHECKING, Optional

import xoscar as xo

if TYPE_CHECKING:
    from xoscar.backends.pool import MainActorPoolType


class LoggerNameFilter(logging.Filter):
    def filter(self, record):
        return record.name.startswith("xinference") or (
            record.name.startswith("uvicorn.error")
            and record.getMessage().startswith("Uvicorn running on")
        )


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
                "format": "%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s"
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
        "root": {
            "level": log_level,
            "handlers": ["stream_handler", "file_handler"],
        },
    }
    return config_dict


async def create_worker_actor_pool(
    address: str, logging_conf: Optional[dict] = None
) -> "MainActorPoolType":
    subprocess_start_method = "forkserver" if os.name != "nt" else "spawn"

    return await xo.create_actor_pool(
        address=address,
        n_process=0,
        subprocess_start_method=subprocess_start_method,
        logging_conf={"dict": logging_conf},
    )
