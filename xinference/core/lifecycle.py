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

"""Worker lifecycle and admission control, independent of actor transport."""

import asyncio
import contextlib
import functools
import inspect
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Iterator,
    Optional,
    Set,
    TypeVar,
)

_T = TypeVar("_T")


class WorkerPhase(str, Enum):
    PREPARED = "prepared"
    ACTIVE = "active"
    DRAINING = "draining"
    TRANSFERRED = "transferred"
    STOPPED = "stopped"


class WorkerLifecycle:
    def __init__(self, active: bool = False) -> None:
        self.phase = WorkerPhase.ACTIVE if active else WorkerPhase.PREPARED
        self._tasks: Dict[Any, int] = {}
        self._background: Set[asyncio.Task] = set()
        self._idle = asyncio.Event()
        self._idle.set()

    def start_background(
        self, coroutine: Coroutine[Any, Any, _T]
    ) -> Optional[asyncio.Task[_T]]:
        # Reserve admission before the task's first run so drain also waits for
        # scheduled work, including tasks cancelled before their first run.
        if self.phase != WorkerPhase.ACTIVE and not (
            self.phase == WorkerPhase.DRAINING and asyncio.current_task() in self._tasks
        ):
            coroutine.close()
            return None
        task = asyncio.create_task(coroutine)
        self._tasks[task] = self._tasks.get(task, 0) + 1
        self._background.add(task)
        self._idle.clear()

        def finished(done):
            self._tasks.pop(done, None)
            self._background.discard(done)
            if not self._tasks:
                self._idle.set()

        task.add_done_callback(finished)
        return task

    def stop_background(self) -> None:
        for task in tuple(self._background):
            task.cancel()

    @contextlib.contextmanager
    def sync_operation(self) -> Iterator[None]:
        task = asyncio.current_task()
        # Nested operations retain admission granted to their outer operation.
        if task not in self._tasks and self.phase != WorkerPhase.ACTIVE:
            raise RuntimeError(f"Worker management is unavailable: {self.phase.value}")
        self._tasks[task] = self._tasks.get(task, 0) + 1
        self._idle.clear()
        try:
            yield
        finally:
            self._tasks[task] -= 1
            if not self._tasks[task]:
                del self._tasks[task]
            if not self._tasks:
                self._idle.set()

    @contextlib.asynccontextmanager
    async def operation(self) -> AsyncIterator[None]:
        with self.sync_operation():
            yield

    async def drain(self, timeout: float) -> None:
        if self.phase != WorkerPhase.ACTIVE:
            raise RuntimeError(f"Cannot drain worker in {self.phase.value}")
        if asyncio.current_task() in self._tasks:
            raise RuntimeError("A management operation cannot drain itself")
        self.phase = WorkerPhase.DRAINING
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
        except BaseException:
            self.phase = WorkerPhase.ACTIVE
            raise


def spawn_management_task(owner: Any, coroutine: Coroutine[Any, Any, _T]):
    lifecycle = getattr(owner, "_lifecycle", None)
    return (
        lifecycle.start_background(coroutine)
        if lifecycle
        else asyncio.create_task(coroutine)
    )


def management_operation(method: Callable) -> Callable:
    if not inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        def sync_admitted(self, *args, **kwargs):
            lifecycle = getattr(self, "_lifecycle", None)
            if lifecycle is None:
                return method(self, *args, **kwargs)
            with lifecycle.sync_operation():
                return method(self, *args, **kwargs)

        return sync_admitted

    @functools.wraps(method)
    async def admitted(self, *args, **kwargs):
        lifecycle = getattr(self, "_lifecycle", None)
        if lifecycle is None:  # Lightweight actor test doubles have no runtime.
            return await method(self, *args, **kwargs)
        async with lifecycle.operation():
            return await method(self, *args, **kwargs)

    return admitted


async def finish_before_cancelling(awaitable: Awaitable[_T]) -> _T:
    """Caller cancellation cannot interrupt ownership transfer or its rollback."""
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise
