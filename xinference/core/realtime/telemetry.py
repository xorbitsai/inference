# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded local traces, not a media log or a high-cardinality metrics registry."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict

logger = logging.getLogger(__name__)


class RealtimeTrace:
    def __init__(self, session_id: str):
        self.trace_id = hashlib.sha256(session_id.encode()).hexdigest()[:16]
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._events: deque = deque(maxlen=128)
        self._durations: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "total_ms": 0, "max_ms": 0}
        )

    def event(self, stage: str, **fields: Any) -> None:
        # Callers pass only stage names, numeric ranges/timings and classifications.
        event = {"stage": stage, "at_ms": (time.monotonic() - self.started) * 1000}
        event.update(fields)
        with self._lock:
            self._events.append(event)
        logger.debug("Realtime trace=%s event=%s", self.trace_id, event)

    def duration(self, stage: str, elapsed: float) -> None:
        with self._lock:
            item = self._durations[stage]
            item["count"] += 1
            item["total_ms"] += elapsed * 1000
            item["max_ms"] = max(item["max_ms"], elapsed * 1000)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "trace_id": self.trace_id,
                "elapsed_ms": (time.monotonic() - self.started) * 1000,
                "durations": {
                    key: dict(value) for key, value in self._durations.items()
                },
                "events": list(self._events),
            }
