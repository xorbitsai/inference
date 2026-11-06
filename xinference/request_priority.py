# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import heapq
import itertools
from typing import Any, Dict, List, Optional, Tuple

INVALID_PRIORITY = "priority must be an integer"
MIN_REQUEST_PRIORITY = -10000
MAX_REQUEST_PRIORITY = 10000
PRIORITY_OUT_OF_RANGE = "priority must be between -10000 and 10000"
VLLM_PRIORITY_VERSION = "Request priority requires vLLM >= 0.6.3"
VLLM_PRIORITY_DISABLED = (
    "Nonzero priority requires launching vLLM with scheduling_policy=priority"
)
SGLANG_PRIORITY_DISABLED = (
    "Nonzero priority requires launching SGLang with enable_priority_scheduling=true"
)
TRT_PRIORITY_DISABLED = "Nonzero priority requires launching TRT-LLM with scheduler_config.waiting_queue_policy=priority"
TRT_PRIORITY_VERSION = "Request priority requires a TRT-LLM version with native priority support (1.3.0rc24 or later); omit priority on older versions"
PRIORITY_ERROR_MESSAGES = (
    INVALID_PRIORITY,
    PRIORITY_OUT_OF_RANGE,
    VLLM_PRIORITY_VERSION,
    VLLM_PRIORITY_DISABLED,
    SGLANG_PRIORITY_DISABLED,
    TRT_PRIORITY_DISABLED,
    TRT_PRIORITY_VERSION,
)


def validate_request_priority(value: Any) -> int:
    """Validate a scheduling priority without coercing strings or booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(INVALID_PRIORITY)
    if not MIN_REQUEST_PRIORITY <= value <= MAX_REQUEST_PRIORITY:
        raise ValueError(PRIORITY_OUT_OF_RANGE)
    return value


def to_trt_request_priority(priority: int) -> float:
    """TRT uses [0, 1], higher first; keep every supported integer distinct."""
    return (MAX_REQUEST_PRIORITY - priority) / (
        MAX_REQUEST_PRIORITY - MIN_REQUEST_PRIORITY
    )


class PriorityScheduler:
    """Bound active work and admit eligible waiters by priority, then arrival."""

    def __init__(self, capacity: int):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Priority scheduler capacity must be a positive integer")
        self._available = capacity
        self._capacity = capacity
        self._active: Dict[int, int] = {}
        self._sequence = itertools.count()
        self._waiting: List[Tuple[int, int, asyncio.Future, int]] = []
        self._closed = False

    @property
    def idle(self) -> bool:
        return self._available == self._capacity

    @property
    def closed(self) -> bool:
        return self._closed

    async def acquire(self, priority: int, capacity: Optional[int] = None) -> None:
        capacity = self._capacity if capacity is None else capacity
        if type(capacity) is not int or not 1 <= capacity <= self._capacity:
            raise ValueError("Request capacity must be within the scheduler capacity")
        if self._closed:
            raise RuntimeError("Model is stopping")
        future = asyncio.get_running_loop().create_future()
        entry = (priority, next(self._sequence), future, capacity)
        heapq.heappush(self._waiting, entry)
        self._wake()
        try:
            await future
        except BaseException:
            if future.done() and not future.cancelled() and future.exception() is None:
                # A slot was reserved before this waiter observed cancellation.
                self.release(capacity)
            else:
                future.cancel()
                # Cancellation must reclaim queue storage even while every
                # active slot remains occupied.
                if entry in self._waiting:
                    self._waiting.remove(entry)
                    heapq.heapify(self._waiting)
                self._wake()
            raise

    def release(self, capacity: Optional[int] = None) -> None:
        capacity = self._capacity if capacity is None else capacity
        self._active[capacity] -= 1
        self._available += 1
        self._wake()

    def _wake(self) -> None:
        blocked = []
        while self._available and self._waiting and not self._closed:
            entry = heapq.heappop(self._waiting)
            _, _, future, capacity = entry
            if not future.done():
                if self._active.get(capacity, 0) >= capacity:
                    # MLX's serial fallback and batch paths share the total
                    # capacity, but only one fallback may enter native work.
                    blocked.append(entry)
                    continue
                self._active[capacity] = self._active.get(capacity, 0) + 1
                self._available -= 1
                future.set_result(None)
        for entry in blocked:
            heapq.heappush(self._waiting, entry)

    def close(self) -> None:
        self._closed = True
        while self._waiting:
            _, _, future, _ = heapq.heappop(self._waiting)
            if not future.done():
                future.set_exception(RuntimeError("Model is stopping"))
