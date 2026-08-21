# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Process-tree resource metrics for an independent Token Router runtime."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Iterable, Optional

import psutil

_IGNORED_PROCESS_ERRORS = (psutil.Error, OSError, AttributeError)


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as file:
            return file.read().strip()
    except (OSError, UnicodeError):
        return None


def _cgroup_cpu_quota() -> Optional[float]:
    """Return a cgroup v1/v2 CPU quota in cores when one is configured."""
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        quota, _, period = cpu_max.partition(" ")
        if quota != "max":
            try:
                value = float(quota) / float(period)
                if value > 0:
                    return value
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    quota_text = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_text = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_text and period_text:
        try:
            value = float(quota_text) / float(period_text)
            if value > 0:
                return value
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return None


def _cgroup_memory_limit() -> Optional[int]:
    """Return a finite cgroup v1/v2 memory limit when one is configured."""
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        raw = _read_text(path)
        if not raw or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        # Common cgroup-v1 "unlimited" sentinels are close to LONG_MAX.
        if 0 < value < (1 << 60):
            return value
    return None


class ProcessMetricsCollector:
    """Collect non-blocking CPU and memory metrics for the Router process tree."""

    def __init__(
        self,
        pid: Optional[int] = None,
        *,
        clock: Callable[[], float] = time.time,
        process: Optional[psutil.Process] = None,
    ) -> None:
        self._clock = clock
        if process is not None:
            self._process: Optional[psutil.Process] = process
        else:
            try:
                self._process = psutil.Process(pid or os.getpid())
            except _IGNORED_PROCESS_ERRORS:
                self._process = None
        self.started_at: Optional[float] = (
            self._safe_call(self._process.create_time, None)
            if self._process is not None
            else None
        )
        self._prime_cpu_baselines()

    @staticmethod
    def _safe_call(call: Callable[[], Any], default: Any) -> Any:
        try:
            return call()
        except _IGNORED_PROCESS_ERRORS:
            return default

    def _processes(self) -> Iterable[psutil.Process]:
        if self._process is None:
            return
        process = self._process
        yield process
        children = self._safe_call(lambda: process.children(recursive=True), [])
        yield from children

    def _prime_cpu_baselines(self) -> None:
        for process in self._processes():
            self._safe_call(lambda: process.cpu_percent(interval=None), 0.0)

    def _cpu_capacity(self) -> float:
        process = self._process
        if process is not None:
            affinity = self._safe_call(
                lambda: process.cpu_affinity(),  # type: ignore[attr-defined]
                [],
            )
            if affinity:
                return float(len(affinity))

        quota = _cgroup_cpu_quota()
        if quota is not None:
            return quota

        cpu_count = psutil.cpu_count()
        return float(cpu_count) if cpu_count is not None else 0.0

    @staticmethod
    def _memory_capacity() -> int:
        cgroup_limit = _cgroup_memory_limit()
        if cgroup_limit is not None:
            return cgroup_limit
        try:
            return int(psutil.virtual_memory().total)
        except _IGNORED_PROCESS_ERRORS:
            return 0

    def collect(self) -> Dict[str, Any]:
        """Return a best-effort process-tree snapshot without blocking for sampling."""
        sampled_at = self._clock()
        processes = list(self._processes())
        if not processes:
            return {}

        cpu_percent = 0.0
        main_rss = 0
        child_rss = 0
        child_count = 0
        thread_count = 0

        for index, process in enumerate(processes):
            current_cpu = self._safe_call(
                lambda: process.cpu_percent(interval=None), None
            )
            if current_cpu is not None:
                cpu_percent += float(current_cpu)

            memory_info = self._safe_call(process.memory_info, None)
            if memory_info is not None:
                rss = int(memory_info.rss)
                if index == 0:
                    main_rss = rss
                else:
                    child_rss += rss

            current_threads = self._safe_call(process.num_threads, None)
            if current_threads is not None:
                thread_count += int(current_threads)

            if index > 0:
                child_count += 1

        started_at = self.started_at
        result: Dict[str, Any] = {
            "cpu_percent": cpu_percent,
            "cpu_cores": cpu_percent / 100.0,
            "cpu_count": self._cpu_capacity(),
            "rss_bytes": main_rss + child_rss,
            "memory_total_bytes": self._memory_capacity(),
            "main_process_rss_bytes": main_rss,
            "child_process_rss_bytes": child_rss,
            "child_process_count": child_count,
            "thread_count": thread_count,
            "sampled_at": sampled_at,
        }
        if started_at is not None:
            result["started_at"] = started_at
            result["uptime_seconds"] = max(0.0, sampled_at - started_at)
        return result
