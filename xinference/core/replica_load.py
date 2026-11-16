# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

"""The load contract between engine adapters, model actors, and replica policy."""

import asyncio
import math
import time
from collections import OrderedDict
from contextvars import ContextVar
from typing import Awaitable, Dict, NamedTuple, Optional, Protocol, Tuple, Union

from .exceptions import BatchRequestRedirect, RequestQueueFullError

batch_queue_budget: ContextVar[Optional[float]] = ContextVar(
    "batch_queue_budget", default=None
)


class ReplicaRequestQueue(asyncio.Queue):
    # asyncio.Queue supplies the wakeup for the batch processor. OrderedDict
    # lets cancellation and work redistribution remove entries by identity.
    _queue: OrderedDict
    waiting: Dict[asyncio.Future, Tuple[float, asyncio.TimerHandle, bool]]

    def _init(self, maxsize):
        self._queue = OrderedDict()

    def _put(self, item):
        payload, future = item
        self._queue[future] = payload

    def _get(self):
        future, payload = self._queue.popitem(last=False)
        self.waiting.pop(future)[1].cancel()
        self.task_done()
        if self.empty():
            self._has_waiters.clear()
        return payload, future

    def __init__(self, capacity: int, timeout: float, *, label: str = "batch"):
        super().__init__(maxsize=capacity)
        self.timeout = timeout
        self.label = label
        self.waiting = {}
        self._has_waiters = asyncio.Event()

    async def wait_for_item(self):
        while self.empty():
            await self._has_waiters.wait()

    def peek(self):
        return next(iter(self._queue.values()))

    def add(self, payload, future, can_redirect, remaining):
        if self.full():
            raise RequestQueueFullError(f"Replica {self.label} queue is full")
        budget = self.timeout if remaining is None else min(self.timeout, remaining)
        if budget <= 0:
            raise RequestQueueFullError(f"Timed out waiting for a {self.label}")
        timer = asyncio.get_running_loop().call_later(budget, self.expire, future)
        self.waiting[future] = time.monotonic(), timer, can_redirect
        self.put_nowait((payload, future))
        self._has_waiters.set()

    def discard(self, future):
        entry = self.waiting.pop(future, None)
        if entry is None:
            return False
        entry[1].cancel()
        self._queue.pop(future)
        self.task_done()
        if self.empty():
            self._has_waiters.clear()
        return True

    def expire(self, future):
        if self.discard(future) and not future.done():
            future.set_exception(
                RequestQueueFullError(f"Timed out waiting for a {self.label}")
            )

    def oldest_wait(self):
        oldest = next(iter(self.waiting.values()), None)
        return time.monotonic() - oldest[0] if oldest else 0.0

    def redirect(self, target, limit, min_wait):
        now = time.monotonic()
        moved = 0
        for future, (queued_at, timer, can_redirect) in list(self.waiting.items()):
            if now - queued_at < min_wait or moved >= limit:
                break
            if can_redirect and not future.done() and self.discard(future):
                future.set_exception(BatchRequestRedirect(target, timer.when()))
                moved += 1
        return moved


class SerializedReplicaAdmission:
    """One execution slot with the same bounded, transferable waiters as batching.

    A concrete adapter may declare `replica_autoscaling = "serialized"` if all
    inference completes within the call, with no independent engine scheduler.
    """

    def __init__(self, capacity: int, timeout: float):
        self.queue = ReplicaRequestQueue(capacity, timeout, label="generation")
        self.running = False

    async def __aenter__(self) -> None:
        if not self.running:
            self.running = True
            return
        future = asyncio.get_running_loop().create_future()
        remaining = batch_queue_budget.get()
        self.queue.add(None, future, remaining is None, remaining)
        try:
            await future
        except asyncio.CancelledError:
            if not self.queue.discard(future) and not future.cancelled():
                # Release a slot granted immediately before cancellation.
                if future.exception() is None:
                    self.release()
            raise

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def release(self) -> None:
        while not self.queue.empty():
            _, future = self.queue.get_nowait()
            if not future.done():
                future.set_result(None)
                return
        self.running = False


class EngineLoad(NamedTuple):
    """Backend work counts in engine units, such as sequences or decode jobs."""

    waiting: int = 0
    backlog_seconds: float = 0.0
    running: int = 0


class EngineLoadReporter(Protocol):
    """Implement on a concrete adapter with `replica_autoscaling = "native"`."""

    def enable_engine_load_reporting(self) -> None:
        """Enable instrumentation before the engine loads."""
        ...

    def get_engine_load(
        self,
    ) -> Union[Optional[EngineLoad], Awaitable[Optional[EngineLoad]]]:
        """Report native work, including waiting inputs hidden from REST admission.

        Keep sampling cheap and non-blocking. Return None when unavailable.
        Provide a sample at startup once initialized, including when idle.
        Report work until the engine releases it, including after cancellation.
        """
        ...


def validate_engine_load(load: EngineLoad) -> None:
    if (
        not isinstance(load, EngineLoad)
        or type(load.waiting) is not int
        or type(load.running) is not int
        or min(load.waiting, load.running) < 0
        or not math.isfinite(load.backlog_seconds)
        or load.backlog_seconds < 0
        or (not load.waiting and load.backlog_seconds != 0)
    ):
        raise ValueError("Invalid autoscaling engine load snapshot")


class ReplicaLoad(NamedTuple):
    queued: int
    oldest_wait: float
    spare: int
    engine_waiting: int = 0
    engine_backlog_seconds: float = 0.0
    engine_running: int = 0
