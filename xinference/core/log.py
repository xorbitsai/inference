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
"""Unified log handler factory for application and audit logs."""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from typing import Optional


def create_rotating_handler(
    filename: str,
    retention_days: int = 30,
    rotation: str = "daily",
    max_bytes: int = 100 * 1024 * 1024,
    formatter: Optional[logging.Formatter] = None,
    encoding: str = "utf8",
    backup_count: int = 0,
) -> logging.Handler:
    """Create a log handler with unified rotation strategy.

    Parameters
    ----------
    filename : str
        Absolute path to the log file.
    retention_days : int
        For "daily": number of backup files (days). For "daily+size":
        date-based retention days. For "size": unused (use backup_count).
    rotation : str
        "daily", "daily+size", or "size".
    max_bytes : int
        Max bytes per file (used by "size" and "daily+size").
    formatter : logging.Formatter, optional
        Formatter to attach to the handler.
    encoding : str
        File encoding for log handlers.
    backup_count : int
        File-count cap for "daily+size" mode (0 = no cap).
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if rotation == "daily":
        handler: logging.Handler = TimedRotatingFileHandler(
            filename=filename,
            when="midnight",
            backupCount=retention_days,
            encoding=encoding,
        )
    elif rotation == "daily+size":
        from ..deploy.utils import SafeTimedAndSizeRotatingFileHandler

        handler = SafeTimedAndSizeRotatingFileHandler(
            filename=filename,
            when="midnight",
            backupCount=backup_count,
            maxBytes=max_bytes,
            retention_days=retention_days,
            encoding=encoding,
        )
    else:
        from ..deploy.utils import SafeRotatingFileHandler

        handler = SafeRotatingFileHandler(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=retention_days,
            encoding=encoding,
        )

    if formatter:
        handler.setFormatter(formatter)
    return handler
