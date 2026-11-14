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

import asyncio
import inspect
import logging
import time
import types
from collections import OrderedDict
from contextvars import ContextVar
from typing import Dict, Optional, Tuple

from xoscar.batch import _ExtensibleWrapper

from ..constants import XINFERENCE_BATCH_INTERVAL, XINFERENCE_BATCH_SIZE
from ..core.exceptions import BatchRequestRedirect, RequestQueueFullError

logger = logging.getLogger(__name__)
batch_queue_budget: ContextVar[Optional[float]] = ContextVar(
    "batch_queue_budget", default=None
)


def _get_batch_queue_budget() -> Optional[float]:
    return batch_queue_budget.get()


class _RedistributableBatchQueue(asyncio.Queue):
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

    def __init__(self, capacity, timeout):
        super().__init__(maxsize=capacity)
        self.timeout = timeout
        self.waiting = {}
        self._has_waiters = asyncio.Event()

    async def wait_for_item(self):
        while self.empty():
            await self._has_waiters.wait()

    def peek(self):
        return next(iter(self._queue.values()))

    def add(self, args, kwargs, future, can_redirect, remaining):
        if self.full():
            raise RequestQueueFullError("Replica batch queue is full")
        budget = self.timeout if remaining is None else min(self.timeout, remaining)
        if budget <= 0:
            raise RequestQueueFullError("Timed out waiting for a batch")
        timer = asyncio.get_running_loop().call_later(budget, self.expire, future)
        self.waiting[future] = time.monotonic(), timer, can_redirect
        self.put_nowait(((args, kwargs), future))
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
            future.set_exception(RequestQueueFullError("Timed out waiting for a batch"))

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


class BatchMixin:
    allow_batch = True
    batch_size = XINFERENCE_BATCH_SIZE
    batch_interval = XINFERENCE_BATCH_INTERVAL

    def __init__(self, func: _ExtensibleWrapper, **kwargs):
        self._queue: Optional[asyncio.Queue] = None
        self._func = func
        self._func_name = func.func.__name__
        wrapper = self._wrap_method()
        self._batch_wrapper = wrapper
        setattr(self, self._func_name, types.MethodType(wrapper, self))

        self._process_batch_task = None
        self._pending_batch_futures: set[asyncio.Future] = set()

        if "batch_size" in kwargs:
            self.batch_size = int(kwargs.pop("batch_size") or XINFERENCE_BATCH_SIZE)
        if "batch_interval" in kwargs:
            self.batch_interval = float(
                kwargs.pop("batch_interval") or XINFERENCE_BATCH_INTERVAL
            )

    @property
    def queue(self):
        if self._queue is None:
            self._queue: asyncio.Queue = asyncio.Queue()
        return self._queue

    def enable_batch_redistribution(self, queue_capacity: int, queue_timeout: float):
        # Some engines replace this wrapper with their own native async queue.
        self._queue = _RedistributableBatchQueue(queue_capacity, queue_timeout)

    def supports_batch_redistribution(self) -> bool:
        wrapper = getattr(self, self._func_name)
        return (
            isinstance(self._queue, _RedistributableBatchQueue)
            and getattr(wrapper, "__func__", None) is self._batch_wrapper
        )

    def batch_queue_load(self):
        queue = self._queue
        if isinstance(queue, _RedistributableBatchQueue):
            return queue.qsize(), queue.oldest_wait()
        return 0, 0.0

    def redirect_batch_waiters(self, target, limit):
        queue = self._queue
        if isinstance(queue, _RedistributableBatchQueue):
            return queue.redirect(target, limit, max(self.batch_interval, 0.01))
        return 0

    def _cleanup_finished_process_batch_task(self):
        task = self._process_batch_task
        if task is not None and task.done():
            # Run cleanup synchronously before a new request is tracked. The
            # scheduled done callback becomes a no-op once a replacement task is
            # installed, avoiding both a hang and failure of the new request.
            self._on_process_batch_done(task)

    def _ensure_process_batch_running(self):
        self._cleanup_finished_process_batch_task()
        if self._process_batch_task is not None:
            return

        # Keep the task reference so a failed processor can be restarted by the
        # next request instead of leaving queued futures waiting forever.
        task = asyncio.create_task(self._process_batch())
        self._process_batch_task = task
        task.add_done_callback(self._on_process_batch_done)

    def _on_process_batch_done(self, task: asyncio.Task):
        if self._process_batch_task is not task:
            return

        self._process_batch_task = None

        if task.cancelled():
            self._fail_pending_requests(asyncio.CancelledError())
            return

        exception = task.exception()
        if exception is None:
            exception = RuntimeError(
                f"Batch processor {self._func_name} stopped unexpectedly"
            )
        logger.error(
            "Batch processor %s stopped unexpectedly",
            self._func_name,
            exc_info=(type(exception), exception, exception.__traceback__),
        )
        self._fail_pending_requests(exception)

    def _fail_pending_requests(self, exception: BaseException):
        # Futures are tracked from enqueue until completion, so this also covers
        # requests already dequeued by a processor that exits unexpectedly.
        pending_futures = list(self._pending_batch_futures)
        for future in pending_futures:
            if future.done():
                continue
            if isinstance(exception, asyncio.CancelledError):
                future.cancel()
            else:
                try:
                    future.set_exception(type(exception)(*exception.args))
                except Exception:
                    future.set_exception(exception)

        # Discard queued entries after their futures have been completed. This
        # prevents a replacement processor from doing work for failed callers.
        if self._queue is None:
            return
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _get_batch_size(self, *args, **kwargs) -> int:
        raise NotImplementedError

    async def _process_batch(self):
        pending = None
        while True:
            # Wait until at least one item is available.
            if pending is None:
                (first_args, first_kwargs), first_future = await self.queue.get()
            else:
                (first_args, first_kwargs), first_future = pending
                pending = None

            if first_future.done():
                continue

            delays = [self._func.delay(*first_args, **first_kwargs)]
            size = self._get_batch_size(*first_args, **first_kwargs)
            futures = [first_future]

            # Try to gather more items into the same batch within a short timeout
            # window. This allows batching multiple requests that arrive close
            # together.
            while size < self.batch_size:
                try:
                    queue = self.queue
                    if isinstance(queue, _RedistributableBatchQueue):
                        if queue.empty():
                            if not self.batch_interval:
                                break
                            await asyncio.wait_for(
                                queue.wait_for_item(), timeout=self.batch_interval
                            )
                            if queue.empty():
                                # The item expired or moved while we were waking up.
                                break
                        args, kwargs = queue.peek()
                        next_size = self._get_batch_size(*args, **kwargs)
                        if size + next_size > self.batch_size:
                            # Leave it visible, expirable and eligible for transfer
                            # while the current batch is running.
                            break
                        (args, kwargs), future = queue.get_nowait()
                    else:
                        if self.batch_interval == 0:
                            (args, kwargs), future = queue.get_nowait()
                        else:
                            (args, kwargs), future = await asyncio.wait_for(
                                queue.get(), timeout=self.batch_interval
                            )
                        if future.done():
                            continue
                        next_size = self._get_batch_size(*args, **kwargs)
                    if size + next_size > self.batch_size:
                        # Preserve FIFO order without putting the request back
                        # behind newer queue entries.
                        pending = ((args, kwargs), future)
                        break
                    size += next_size
                    delays.append(self._func.delay(*args, **kwargs))
                    futures.append(future)
                except (asyncio.QueueEmpty, asyncio.TimeoutError):
                    break

            # A caller awaiting its future may be cancelled while its item is
            # queued. Do not execute work solely for cancelled callers.
            active = [
                (delay, future)
                for delay, future in zip(delays, futures)
                if not future.done()
            ]
            if not active:
                continue
            active_delays, active_futures = zip(*active)

            logger.debug("Calling batch %s with %d size", self._func_name, size)
            try:
                if (
                    isinstance(self._queue, _RedistributableBatchQueue)
                    and not self._func.is_async
                ):
                    # Leave the actor loop responsive to admission and work
                    # redistribution while a synchronous engine is encoding.
                    results = await asyncio.to_thread(self._func.batch, *active_delays)
                else:
                    results = self._func.batch(*active_delays)
                if inspect.isawaitable(results):
                    results = await results
                if len(results) != len(active_futures):
                    raise RuntimeError(
                        "#results should be equal to #futures, "
                        f"got {len(results)} and {len(active_futures)}"
                    )
            except asyncio.CancelledError:
                for future in futures:
                    if not future.done():
                        future.cancel()
                raise
            except Exception as error:  # Handle errors for the entire batch.
                for future in futures:
                    if not future.done():
                        future.set_exception(error)
            else:
                # A future can also be cancelled while the batch function is
                # running. Skipping completed futures prevents InvalidStateError
                # from terminating the processor and stalling later requests.
                for future, result in zip(active_futures, results):
                    if not future.done():
                        future.set_result(result)

    def _wrap_method(self):

        async def _replaced_async_method(model, *args, **kwargs):
            # Finish cleanup for a processor that exited between event-loop turns
            # before associating the new request with the replacement processor.
            self._cleanup_finished_process_batch_task()
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            queue = self.queue
            if isinstance(queue, _RedistributableBatchQueue):
                remaining = kwargs.pop("_batch_queue_budget", _get_batch_queue_budget())
                queue.add(args, kwargs, fut, remaining is None, remaining)
            else:
                queue.put_nowait(((args, kwargs), fut))
            self._pending_batch_futures.add(fut)
            fut.add_done_callback(self._pending_batch_futures.discard)
            self._ensure_process_batch_running()
            if not isinstance(queue, _RedistributableBatchQueue):
                return await fut
            try:
                return await asyncio.shield(fut)
            except asyncio.CancelledError:
                if queue.discard(fut):
                    fut.cancel()
                else:
                    # A batch already owns the request. Let native work finish.
                    while not fut.done():
                        try:
                            await asyncio.shield(fut)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    if not fut.cancelled():
                        fut.exception()
                raise

        return _replaced_async_method
