from __future__ import annotations

import asyncio
from collections import Counter


class RouterMetrics:
    def __init__(self) -> None:
        self._counters: Counter[tuple[str, ...]] = Counter()
        self._tokenization_outcomes: Counter[str] = Counter()
        self._tokenization_rejected: Counter[str] = Counter()
        self._tokenization_duration_count = 0
        self._tokenization_duration_sum = 0.0
        self._tokenization_duration_max = 0.0
        self._tokenization_input_bytes_count = 0
        self._tokenization_input_bytes_sum = 0
        self._tokenization_input_bytes_max = 0
        self._tokenization_active = 0
        self._tokenization_waiting = 0
        self._lock = asyncio.Lock()

    async def increment(self, *labels: str) -> None:
        async with self._lock:
            self._counters[tuple(labels)] += 1

    async def increment_tokenization_rejected(self, reason: str) -> None:
        async with self._lock:
            self._tokenization_rejected[reason] += 1

    async def observe_tokenization(
        self, *, duration_seconds: float, input_bytes: int, outcome: str
    ) -> None:
        async with self._lock:
            self._tokenization_outcomes[outcome] += 1
            self._tokenization_duration_count += 1
            self._tokenization_duration_sum += duration_seconds
            self._tokenization_duration_max = max(
                self._tokenization_duration_max, duration_seconds
            )
            self._tokenization_input_bytes_count += 1
            self._tokenization_input_bytes_sum += input_bytes
            self._tokenization_input_bytes_max = max(
                self._tokenization_input_bytes_max, input_bytes
            )

    async def set_tokenization_capacity(self, *, active: int, waiting: int) -> None:
        async with self._lock:
            self._tokenization_active = active
            self._tokenization_waiting = waiting

    async def summary(self) -> dict[str, object]:
        async with self._lock:
            return {
                "requests": {
                    "/".join(labels): value for labels, value in self._counters.items()
                },
                "tokenization_outcomes": dict(self._tokenization_outcomes),
                "tokenization_rejected": dict(self._tokenization_rejected),
                "tokenization_duration": {
                    "count": self._tokenization_duration_count,
                    "sum_seconds": self._tokenization_duration_sum,
                    "max_seconds": self._tokenization_duration_max,
                },
                "tokenization_input_bytes": {
                    "count": self._tokenization_input_bytes_count,
                    "sum": self._tokenization_input_bytes_sum,
                    "max": self._tokenization_input_bytes_max,
                },
                "tokenization_active": self._tokenization_active,
                "tokenization_waiting": self._tokenization_waiting,
            }

    async def render(self) -> str:
        async with self._lock:
            counters = dict(self._counters)
            outcomes = dict(self._tokenization_outcomes)
            rejected = dict(self._tokenization_rejected)
            duration_count = self._tokenization_duration_count
            duration_sum = self._tokenization_duration_sum
            duration_max = self._tokenization_duration_max
            input_bytes_count = self._tokenization_input_bytes_count
            input_bytes_sum = self._tokenization_input_bytes_sum
            input_bytes_max = self._tokenization_input_bytes_max
            active = self._tokenization_active
            waiting = self._tokenization_waiting

        lines = [
            "# HELP xinference_token_router_requests_total Router request outcomes.",
            "# TYPE xinference_token_router_requests_total counter",
        ]
        for labels, value in sorted(counters.items()):
            event = labels[0] if labels else "unknown"
            pool = labels[1] if len(labels) > 1 else "none"
            lines.append(
                "xinference_token_router_requests_total"
                f'{{event="{event}",pool="{pool}"}} {value}'
            )

        lines.extend(
            [
                "# HELP xinference_token_router_tokenization_active "
                "Tokenization tasks currently admitted.",
                "# TYPE xinference_token_router_tokenization_active gauge",
                f"xinference_token_router_tokenization_active {active}",
                "# HELP xinference_token_router_tokenization_waiting "
                "Tokenization tasks waiting for admission.",
                "# TYPE xinference_token_router_tokenization_waiting gauge",
                f"xinference_token_router_tokenization_waiting {waiting}",
                "# HELP xinference_token_router_tokenization_duration_seconds "
                "Tokenization execution duration.",
                "# TYPE xinference_token_router_tokenization_duration_seconds summary",
                "xinference_token_router_tokenization_duration_seconds_count "
                f"{duration_count}",
                "xinference_token_router_tokenization_duration_seconds_sum "
                f"{duration_sum:.9f}",
                "# HELP xinference_token_router_tokenization_duration_seconds_max "
                "Maximum observed tokenization duration.",
                "# TYPE xinference_token_router_tokenization_duration_seconds_max gauge",
                "xinference_token_router_tokenization_duration_seconds_max "
                f"{duration_max:.9f}",
                "# HELP xinference_token_router_tokenization_input_bytes "
                "Request body bytes processed by tokenization.",
                "# TYPE xinference_token_router_tokenization_input_bytes summary",
                f"xinference_token_router_tokenization_input_bytes_count {input_bytes_count}",
                f"xinference_token_router_tokenization_input_bytes_sum {input_bytes_sum}",
                "# HELP xinference_token_router_tokenization_input_bytes_max "
                "Maximum observed tokenization request body size.",
                "# TYPE xinference_token_router_tokenization_input_bytes_max gauge",
                f"xinference_token_router_tokenization_input_bytes_max {input_bytes_max}",
                "# HELP xinference_token_router_tokenization_outcomes_total "
                "Tokenization task outcomes.",
                "# TYPE xinference_token_router_tokenization_outcomes_total counter",
            ]
        )
        for outcome, value in sorted(outcomes.items()):
            lines.append(
                "xinference_token_router_tokenization_outcomes_total"
                f'{{outcome="{outcome}"}} {value}'
            )
        lines.extend(
            [
                "# HELP xinference_token_router_tokenization_rejected_total "
                "Tokenization admission rejections.",
                "# TYPE xinference_token_router_tokenization_rejected_total counter",
            ]
        )
        for reason, value in sorted(rejected.items()):
            lines.append(
                "xinference_token_router_tokenization_rejected_total"
                f'{{reason="{reason}"}} {value}'
            )
        return "\n".join(lines) + "\n"
