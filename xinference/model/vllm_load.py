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

import time


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
