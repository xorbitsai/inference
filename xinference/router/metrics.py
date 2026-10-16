from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping, Tuple

_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
_WAIT_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5)


def _escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(values: Mapping[str, object]) -> str:
    return (
        "{"
        + ",".join(f'{key}="{_escape(value)}"' for key, value in values.items())
        + "}"
    )


class _Histogram:
    def __init__(self, buckets: Iterable[float]) -> None:
        self.buckets = tuple(float(value) for value in buckets)
        self.counts: Dict[Tuple[str, ...], list[int]] = defaultdict(
            lambda: [0] * len(self.buckets)
        )
        self.sums: Dict[Tuple[str, ...], float] = defaultdict(float)
        self.totals: Counter[Tuple[str, ...]] = Counter()

    def observe(self, key: Tuple[str, ...], value: float) -> None:
        value = max(0.0, float(value))
        for index, boundary in enumerate(self.buckets):
            if value <= boundary:
                self.counts[key][index] += 1
        self.sums[key] += value
        self.totals[key] += 1

    def snapshot(self) -> "_Histogram":
        copied = _Histogram(self.buckets)
        copied.counts.update({key: list(value) for key, value in self.counts.items()})
        copied.sums.update(self.sums)
        copied.totals.update(self.totals)
        return copied


class RouterMetrics:
    """Concurrency-safe in-process metrics for a single Router Runtime.

    The original ``increment`` and tokenization Summary metrics are retained for
    compatibility. Monitoring V2.1 metrics are emitted alongside them.
    """

    def __init__(self, router_uid: str = "") -> None:
        self._router_uid = router_uid
        self._counters: Counter[tuple[str, ...]] = Counter()
        self._route_requests: Counter[tuple[str, str, str, str]] = Counter()
        self._requests_in_flight: Counter[tuple[str, str]] = Counter()
        self._rule_matches: Counter[tuple[str, str, str]] = Counter()
        self._backend_selections: Counter[tuple[str, str, str, str]] = Counter()
        self._backend_requests: Counter[tuple[str, str, str, str]] = Counter()
        self._pool_rejected: Counter[tuple[str, str, str]] = Counter()
        self._request_duration = _Histogram(_DURATION_BUCKETS)
        self._backend_duration = _Histogram(_DURATION_BUCKETS)
        self._pool_wait_duration = _Histogram(_WAIT_BUCKETS)
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

    def set_router_uid(self, router_uid: str) -> None:
        self._router_uid = str(router_uid)

    async def increment(self, *labels: str) -> None:
        """Increment the legacy request outcome counter."""
        async with self._lock:
            self._counters[tuple(labels)] += 1

    async def request_started(self, router_uid: str = "", pool: str = "none") -> None:
        uid = router_uid or self._router_uid
        async with self._lock:
            self._requests_in_flight[(uid, pool)] += 1

    async def assign_request_pool(
        self, pool: str, *, previous_pool: str = "none", router_uid: str = ""
    ) -> None:
        """Move one in-flight request from its pre-route bucket to a pool."""
        uid = router_uid or self._router_uid
        if pool == previous_pool:
            return
        async with self._lock:
            previous_key = (uid, previous_pool)
            self._requests_in_flight[previous_key] = max(
                0, self._requests_in_flight[previous_key] - 1
            )
            self._requests_in_flight[(uid, pool)] += 1

    async def finish_request(
        self,
        result: str,
        pool: str,
        *,
        duration_seconds: float,
        route_mode: str = "non_stream",
        router_uid: str = "",
    ) -> None:
        uid = router_uid or self._router_uid
        key = (uid, result, route_mode, pool)
        async with self._lock:
            self._counters[(result, pool)] += 1
            self._route_requests[key] += 1
            self._request_duration.observe(key, duration_seconds)
            in_flight_key = (uid, pool)
            self._requests_in_flight[in_flight_key] = max(
                0, self._requests_in_flight[in_flight_key] - 1
            )

    async def record_rule_match(
        self, rule_id: str, outcome: str = "selected", *, router_uid: str = ""
    ) -> None:
        async with self._lock:
            self._rule_matches[(router_uid or self._router_uid, rule_id, outcome)] += 1

    async def record_backend_selection(
        self,
        rule_id: str,
        backend_model_uid: str,
        pool: str = "none",
        *,
        router_uid: str = "",
    ) -> None:
        async with self._lock:
            self._backend_selections[
                (router_uid or self._router_uid, backend_model_uid, rule_id, pool)
            ] += 1

    async def observe_backend(
        self,
        backend_model_uid: str,
        result: str,
        duration_seconds: float,
        *,
        pool: str = "none",
        router_uid: str = "",
    ) -> None:
        key = (router_uid or self._router_uid, backend_model_uid, result, pool)
        async with self._lock:
            self._backend_requests[key] += 1
            self._backend_duration.observe(key, duration_seconds)

    async def observe_pool_wait(
        self, pool: str, duration_seconds: float, *, router_uid: str = ""
    ) -> None:
        async with self._lock:
            self._pool_wait_duration.observe(
                (router_uid or self._router_uid, pool), duration_seconds
            )

    async def record_pool_rejected(
        self, pool: str, reason: str, *, router_uid: str = ""
    ) -> None:
        async with self._lock:
            self._pool_rejected[(router_uid or self._router_uid, pool, reason)] += 1

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

    @staticmethod
    def _render_histogram(
        lines: list[str],
        name: str,
        help_text: str,
        histogram: _Histogram,
        label_names: tuple[str, ...],
    ) -> None:
        lines.extend([f"# HELP {name} {help_text}", f"# TYPE {name} histogram"])
        for key in sorted(histogram.totals):
            base = dict(zip(label_names, key))
            for boundary, count in zip(histogram.buckets, histogram.counts[key]):
                lines.append(
                    f"{name}_bucket{_labels({**base, 'le': boundary})} {count}"
                )
            lines.append(
                f"{name}_bucket{_labels({**base, 'le': '+Inf'})} {histogram.totals[key]}"
            )
            lines.append(f"{name}_sum{_labels(base)} {histogram.sums[key]:.9f}")
            lines.append(f"{name}_count{_labels(base)} {histogram.totals[key]}")

    async def render(
        self,
        *,
        runtime_summary: Mapping[str, Any] | None = None,
        process: Mapping[str, Any] | None = None,
        runtime_metadata: Mapping[str, Any] | None = None,
    ) -> str:
        async with self._lock:
            counters = dict(self._counters)
            route_requests = dict(self._route_requests)
            in_flight = dict(self._requests_in_flight)
            rule_matches = dict(self._rule_matches)
            backend_selections = dict(self._backend_selections)
            backend_requests = dict(self._backend_requests)
            pool_rejected = dict(self._pool_rejected)
            request_duration = self._request_duration.snapshot()
            backend_duration = self._backend_duration.snapshot()
            pool_wait_duration = self._pool_wait_duration.snapshot()
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
                f'{_labels({"event": event, "pool": pool})} {value}'
            )

        lines.extend(
            [
                "# HELP xinference_token_router_route_requests_total Final Router request results.",
                "# TYPE xinference_token_router_route_requests_total counter",
            ]
        )
        for (uid, result, route_mode, pool), value in sorted(route_requests.items()):
            lines.append(
                "xinference_token_router_route_requests_total"
                f'{_labels({"router_uid": uid, "result": result, "route_mode": route_mode, "pool": pool})} {value}'
            )
        self._render_histogram(
            lines,
            "xinference_token_router_request_duration_seconds",
            "End-to-end Router request duration.",
            request_duration,
            ("router_uid", "result", "route_mode", "pool"),
        )
        lines.extend(
            [
                "# HELP xinference_token_router_requests_in_flight Router requests currently in flight.",
                "# TYPE xinference_token_router_requests_in_flight gauge",
            ]
        )
        for (uid, pool), value in sorted(in_flight.items()):
            lines.append(
                "xinference_token_router_requests_in_flight"
                f'{_labels({"router_uid": uid, "pool": pool})} {value}'
            )

        for name, help_text, values, label_names in (
            (
                "xinference_token_router_rule_matches_total",
                "Routing rule matches.",
                rule_matches,
                ("router_uid", "rule_id", "outcome"),
            ),
            (
                "xinference_token_router_backend_selections_total",
                "Backend selections by rule.",
                backend_selections,
                ("router_uid", "backend_model_uid", "rule_id", "pool"),
            ),
            (
                "xinference_token_router_backend_requests_total",
                "Backend request final results.",
                backend_requests,
                ("router_uid", "backend_model_uid", "result", "pool"),
            ),
            (
                "xinference_token_router_pool_rejected_total",
                "Concurrency pool rejections.",
                pool_rejected,
                ("router_uid", "pool", "reason"),
            ),
        ):
            lines.extend([f"# HELP {name} {help_text}", f"# TYPE {name} counter"])
            for key, value in sorted(values.items()):
                lines.append(f"{name}{_labels(dict(zip(label_names, key)))} {value}")
        self._render_histogram(
            lines,
            "xinference_token_router_backend_request_duration_seconds",
            "Backend request duration.",
            backend_duration,
            ("router_uid", "backend_model_uid", "result", "pool"),
        )
        self._render_histogram(
            lines,
            "xinference_token_router_pool_wait_duration_seconds",
            "Concurrency pool wait duration.",
            pool_wait_duration,
            ("router_uid", "pool"),
        )

        summary = runtime_summary or {}
        uid = str((runtime_metadata or {}).get("router_uid") or self._router_uid)
        pools = summary.get("pools", {}) if isinstance(summary, Mapping) else {}
        for metric_name, field, help_text in (
            (
                "xinference_token_router_pool_limit",
                "max_active",
                "Concurrency pool active limit.",
            ),
            (
                "xinference_token_router_pool_in_use",
                "active",
                "Concurrency pool slots in use.",
            ),
            (
                "xinference_token_router_pool_waiting",
                "waiting",
                "Requests waiting for a pool slot.",
            ),
        ):
            lines.extend(
                [f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} gauge"]
            )
            if isinstance(pools, Mapping):
                for pool, pool_state in sorted(
                    pools.items(), key=lambda item: str(item[0])
                ):
                    if isinstance(pool_state, Mapping):
                        lines.append(
                            f"{metric_name}{_labels({'router_uid': uid, 'pool': pool})} {pool_state.get(field, 0)}"
                        )

        lines.extend(
            [
                "# HELP xinference_token_router_tokenization_active Tokenization tasks currently admitted.",
                "# TYPE xinference_token_router_tokenization_active gauge",
                f"xinference_token_router_tokenization_active {active}",
                "# HELP xinference_token_router_tokenization_waiting Tokenization tasks waiting for admission.",
                "# TYPE xinference_token_router_tokenization_waiting gauge",
                f"xinference_token_router_tokenization_waiting {waiting}",
                "# HELP xinference_token_router_tokenization_duration_seconds Tokenization execution duration.",
                "# TYPE xinference_token_router_tokenization_duration_seconds summary",
                f"xinference_token_router_tokenization_duration_seconds_count {duration_count}",
                f"xinference_token_router_tokenization_duration_seconds_sum {duration_sum:.9f}",
                "# HELP xinference_token_router_tokenization_duration_seconds_max Maximum observed tokenization duration.",
                "# TYPE xinference_token_router_tokenization_duration_seconds_max gauge",
                f"xinference_token_router_tokenization_duration_seconds_max {duration_max:.9f}",
                "# HELP xinference_token_router_tokenization_input_bytes Request body bytes processed by tokenization.",
                "# TYPE xinference_token_router_tokenization_input_bytes summary",
                f"xinference_token_router_tokenization_input_bytes_count {input_bytes_count}",
                f"xinference_token_router_tokenization_input_bytes_sum {input_bytes_sum}",
                "# HELP xinference_token_router_tokenization_input_bytes_max Maximum observed tokenization request body size.",
                "# TYPE xinference_token_router_tokenization_input_bytes_max gauge",
                f"xinference_token_router_tokenization_input_bytes_max {input_bytes_max}",
                "# HELP xinference_token_router_tokenization_outcomes_total Tokenization task outcomes.",
                "# TYPE xinference_token_router_tokenization_outcomes_total counter",
            ]
        )
        for outcome, value in sorted(outcomes.items()):
            lines.append(
                "xinference_token_router_tokenization_outcomes_total"
                f'{_labels({"outcome": outcome})} {value}'
            )
        lines.extend(
            [
                "# HELP xinference_token_router_tokenization_rejected_total Tokenization admission rejections.",
                "# TYPE xinference_token_router_tokenization_rejected_total counter",
            ]
        )
        for reason, value in sorted(rejected.items()):
            lines.append(
                "xinference_token_router_tokenization_rejected_total"
                f'{_labels({"reason": reason})} {value}'
            )

        metadata = dict(runtime_metadata or {})
        try:
            from xinference import __version__
        except Exception:  # pragma: no cover
            __version__ = "unknown"
        lines.extend(
            [
                "# HELP xinference_token_router_build_info Token Router Runtime build information (value=1).",
                "# TYPE xinference_token_router_build_info gauge",
                "xinference_token_router_build_info"
                f'{_labels({"version": metadata.get("software_version", __version__), "commit": metadata.get("software_revision", "")})} 1',
                "# HELP xinference_token_router_config_revision Active Router configuration revision.",
                "# TYPE xinference_token_router_config_revision gauge",
                f"xinference_token_router_config_revision{_labels({'router_uid': uid})} {summary.get('revision', 0)}",
                "# HELP xinference_token_router_assignment_generation Active Runtime Assignment generation.",
                "# TYPE xinference_token_router_assignment_generation gauge",
                "xinference_token_router_assignment_generation"
                f'{_labels({"router_uid": uid, "assignment_id": metadata.get("assignment_id", "")})} {metadata.get("assignment_generation") or 0}',
            ]
        )

        resources = process or {}
        process_metrics = (
            (
                "xinference_token_router_process_cpu_seconds_total",
                "counter",
                "cpu_seconds_total",
            ),
            ("xinference_token_router_process_cpu_utilization", "gauge", "cpu_cores"),
            (
                "xinference_token_router_process_resident_memory_bytes",
                "gauge",
                "rss_bytes",
            ),
            (
                "xinference_token_router_process_virtual_memory_bytes",
                "gauge",
                "virtual_memory_bytes",
            ),
            ("xinference_token_router_process_threads", "gauge", "thread_count"),
            (
                "xinference_token_router_process_children",
                "gauge",
                "child_process_count",
            ),
            (
                "xinference_token_router_process_start_time_seconds",
                "gauge",
                "started_at",
            ),
            (
                "xinference_token_router_process_uptime_seconds",
                "gauge",
                "uptime_seconds",
            ),
        )
        for name, metric_type, field in process_metrics:
            lines.extend(
                [
                    f"# HELP {name} Token Router Runtime process-tree {field}.",
                    f"# TYPE {name} {metric_type}",
                    f"{name} {resources.get(field, 0)}",
                ]
            )
        return "\n".join(lines) + "\n"
