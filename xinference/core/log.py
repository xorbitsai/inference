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
"""Unified log handler factory for application and audit logs."""

import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional


def create_rotating_handler(
    filename: str,
    retention_days: int = 30,
    rotation: str = "daily",
    max_bytes: int = 100 * 1024 * 1024,
    formatter: Optional[logging.Formatter] = None,
    encoding: str = "utf8",
) -> logging.Handler:
    """Create a log handler with unified rotation strategy.

    Parameters
    ----------
    filename : str
        Absolute path to the log file.
    retention_days : int
        Number of backup files to keep (days for daily, count for size).
    rotation : str
        "daily" for time-based rotation, "size" for size-based rotation.
    max_bytes : int
        Max bytes per file (only used when rotation="size").
    formatter : logging.Formatter, optional
        Formatter to attach to the handler.
    encoding : str
        File encoding for log handlers.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if rotation == "daily":
        handler: logging.Handler = TimedRotatingFileHandler(
            filename=filename,
            when="midnight",
            backupCount=retention_days,
            encoding=encoding,
        )
    else:
        handler = RotatingFileHandler(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=retention_days,
            encoding=encoding,
        )

    if formatter:
        handler.setFormatter(formatter)
    return handler
