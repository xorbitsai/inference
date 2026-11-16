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

"""Read vLLM scheduler load through its stat logger interface."""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

from ..core.replica_load import EngineLoad

logger = logging.getLogger(__name__)


class VLLMSchedulerLoad:
    """One vLLM scheduler's latest queue count and continuous backlog duration."""

    def __init__(self):
        self.waiting = 0
        self.running = 0
        self.waiting_since = 0.0
        self.updated_at = 0.0

    def record(self, scheduler_stats, iteration_stats, **kwargs):
        waiting = getattr(scheduler_stats, "num_waiting_reqs", None)
        if not isinstance(waiting, int):
            return
        self.update(
            waiting + getattr(scheduler_stats, "num_skipped_waiting_reqs", 0),
            scheduler_stats.num_running_reqs,
        )

    def update(self, waiting: int, running: int):
        self.updated_at = time.monotonic()
        if waiting and not self.waiting:
            self.waiting_since = self.updated_at
        self.waiting = waiting
        self.running = running

    def backlog_age(self, now: float):
        return max(0.0, now - self.waiting_since) if self.waiting else 0.0

    def log(self):
        pass

    def log_engine_initialized(self):
        pass

    def record_sleep_state(self, *args, **kwargs):
        pass


class _PoolingProgress:
    """Count unfinished inputs through an offline pooling call's progress callback."""

    def __init__(self, count: int, capacity: int):
        self.count = count
        self.capacity = capacity
        self.n = 0
        self.load = VLLMSchedulerLoad()

    def __call__(self, iterable=None, **kwargs):
        # Some vLLM versions pass the input iterator to tqdm during submission.
        return self if iterable is None else iterable

    def update(self, count: int) -> None:
        self.n += count

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        pass


class VLLMPoolingLoad:
    """Scheduler telemetry for AsyncLLM and synchronous vLLM pooling adapters."""

    def __init__(self) -> None:
        self._schedulers: Dict[int, VLLMSchedulerLoad] = {}
        self._native_requests = 0
        self._sync_capacity: Optional[int] = None
        self._sync_metrics_verified = False
        self._sync_progress: Optional[_PoolingProgress] = None

    def stat_logger(self, vllm_config: Any, engine_index: int = 0) -> VLLMSchedulerLoad:
        load = VLLMSchedulerLoad()
        self._schedulers[engine_index] = load
        return load

    def configure_sync(self, model: Any) -> None:
        config = model.llm_engine.vllm_config
        self._sync_capacity = config.scheduler_config.max_num_seqs

    @contextmanager
    def native_request(self) -> Iterator[None]:
        if not self._native_requests:
            for load in self._schedulers.values():
                load.waiting = load.running = 0
        self._native_requests += 1
        try:
            yield
        finally:
            self._native_requests -= 1

    @contextmanager
    def sync_request(
        self, count: int, sequence_limit: Optional[int] = None
    ) -> Iterator[_PoolingProgress]:
        # The adapter owns one synchronous call at a time.
        assert self._sync_capacity is not None
        capacity = self._sync_capacity
        if sequence_limit is not None:
            capacity = min(capacity, sequence_limit)
        self._sync_progress = progress = _PoolingProgress(count, capacity)
        try:
            yield progress
        finally:
            self._sync_progress = None

    @staticmethod
    def _read_sync_metrics(model: Any) -> Optional[Tuple[int, int]]:
        counts = {"vllm:num_requests_waiting": 0, "vllm:num_requests_running": 0}
        seen = set()
        try:
            for metric in model.get_metrics():
                if metric.name in counts:
                    seen.add(metric.name)
                    counts[metric.name] += int(metric.value)
        except Exception:
            logger.debug("Could not read vLLM scheduler metrics", exc_info=True)
            return None
        if seen != counts.keys():
            return None
        return counts["vllm:num_requests_waiting"], counts["vllm:num_requests_running"]

    def get_engine_load(self, model: Any) -> Optional[EngineLoad]:
        if self._sync_capacity is not None:
            progress = self._sync_progress
            if progress is None:
                if not self._sync_metrics_verified:
                    self._sync_metrics_verified = (
                        self._read_sync_metrics(model) is not None
                    )
                return EngineLoad() if self._sync_metrics_verified else None
            metrics = self._read_sync_metrics(model)
            if metrics is None:
                return None
            waiting, running = metrics
            if progress is not self._sync_progress:
                return EngineLoad()
            # Offline pooling can retain unfinished inputs outside EngineCore.
            waiting = max(waiting, progress.count - progress.n - progress.capacity)
            progress.load.update(waiting, running)
            return EngineLoad(
                waiting, progress.load.backlog_age(time.monotonic()), running
            )

        if not self._schedulers:
            return None
        if not self._native_requests:
            return EngineLoad()
        now = time.monotonic()
        loads = self._schedulers.values()
        return EngineLoad(
            sum(load.waiting for load in loads),
            max(load.backlog_age(now) for load in loads),
            sum(load.running for load in loads),
        )
