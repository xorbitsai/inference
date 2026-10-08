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
import types

from xoscar.batch import _ExtensibleWrapper

from ..constants import XINFERENCE_BATCH_INTERVAL, XINFERENCE_BATCH_SIZE

logger = logging.getLogger(__name__)


class BatchMixin:
    allow_batch = True
    batch_size = XINFERENCE_BATCH_SIZE
    batch_interval = XINFERENCE_BATCH_INTERVAL

    def __init__(self, func: _ExtensibleWrapper, **kwargs):
        self._queue = None
        self._func = func
        self._func_name = func.func.__name__
        setattr(self, self._func_name, types.MethodType(self._wrap_method(), self))

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
                    (args, kwargs), future = await asyncio.wait_for(
                        self.queue.get(), timeout=self.batch_interval
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
                except asyncio.TimeoutError:
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
            self._pending_batch_futures.add(fut)
            fut.add_done_callback(self._pending_batch_futures.discard)
            self.queue.put_nowait(((args, kwargs), fut))
            self._ensure_process_batch_running()
            return await fut

        return _replaced_async_method
