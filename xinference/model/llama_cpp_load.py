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
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple

from ..core.replica_load import EngineLoad


class LlamaCppSchedulerLoad:
    """Read xllamacpp scheduler metrics for one owned native call at a time."""

    def __init__(self) -> None:
        self._metrics_verified = False
        self._active = False
        self._call_id = 0
        self._metrics_task: Optional[asyncio.Task[Optional[Tuple[int, int]]]] = None
        self._metrics_call_id = 0
        self._backlog_since: Optional[float] = None

    @contextmanager
    def request(self) -> Iterator[None]:
        self._backlog_since = None
        self._call_id += 1
        self._active = True
        try:
            yield
        finally:
            self._active = False

    async def get_engine_load(self, server: Any) -> Optional[EngineLoad]:
        if server is None:
            return None
        if not self._active:
            self._backlog_since = None
            if self._metrics_verified:
                return EngineLoad()
        if self._metrics_task is None:
            self._metrics_call_id = self._call_id
            self._metrics_task = asyncio.create_task(
                asyncio.to_thread(self._read_scheduler_metrics, server)
            )
        call_id = self._metrics_call_id
        task = self._metrics_task
        try:
            metrics = await asyncio.wait_for(asyncio.shield(task), 0.1)
        except asyncio.TimeoutError:
            return None
        finally:
            if task.done() and self._metrics_task is task:
                self._metrics_task = None
        if metrics is None:
            return None
        self._metrics_verified = True
        waiting, running = metrics
        if call_id != self._call_id or not self._active:
            return EngineLoad(waiting, running=running)
        if not waiting:
            self._backlog_since = None
            return EngineLoad(running=running)
        now = time.monotonic()
        if self._backlog_since is None:
            self._backlog_since = now
        return EngineLoad(waiting, now - self._backlog_since, running)

    @staticmethod
    def _read_scheduler_metrics(server: Any) -> Optional[Tuple[int, int]]:
        metrics = server.handle_metrics()
        if isinstance(metrics, bytes):
            metrics = metrics.decode()
        if not isinstance(metrics, str):
            return None
        waiting = running = None
        for line in metrics.splitlines():
            name, _, value = line.partition(" ")
            if name == "llamacpp:requests_deferred":
                waiting = int(float(value))
            elif name == "llamacpp:requests_processing":
                running = int(float(value))
        if waiting is None or running is None:
            return None
        return waiting, running
