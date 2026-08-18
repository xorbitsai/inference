from __future__ import annotations

import asyncio
from dataclasses import dataclass


class AdmissionRejected(RuntimeError):
    def __init__(self, pool: str, reason: str, retry_after_seconds: int) -> None:
        super().__init__(f"{pool} pool rejected request: {reason}")
        self.pool = pool
        self.reason = reason
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class GateSnapshot:
    active: int
    waiting: int
    max_active: int
    max_queue: int


class CapacityGate:
    def __init__(
        self,
        pool: str,
        *,
        max_active: int,
        max_queue: int,
        queue_timeout_seconds: float,
        retry_after_seconds: int,
    ) -> None:
        self.pool = pool
        self.max_active = max_active
        self.max_queue = max_queue
        self.queue_timeout_seconds = queue_timeout_seconds
        self.retry_after_seconds = retry_after_seconds
        self._active = 0
        self._waiting = 0
        self._condition = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._condition:
            if self._active < self.max_active:
                self._active += 1
                return
            if self._waiting >= self.max_queue:
                raise AdmissionRejected(
                    self.pool, "queue_full", self.retry_after_seconds
                )
            self._waiting += 1
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: self._active < self.max_active),
                    timeout=self.queue_timeout_seconds,
                )
                self._active += 1
            except TimeoutError as exc:
                raise AdmissionRejected(
                    self.pool, "queue_timeout", self.retry_after_seconds
                ) from exc
            finally:
                self._waiting -= 1

    async def release(self) -> None:
        async with self._condition:
            if self._active <= 0:
                raise RuntimeError(f"CapacityGate {self.pool} released without acquire")
            self._active -= 1
            self._condition.notify(1)

    async def snapshot(self) -> GateSnapshot:
        async with self._condition:
            return GateSnapshot(
                active=self._active,
                waiting=self._waiting,
                max_active=self.max_active,
                max_queue=self.max_queue,
            )
