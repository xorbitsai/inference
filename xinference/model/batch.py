# Copyright 2022-2026 XProbe Inc.
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

        self._is_process_batch_running = False

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

    def _ensure_process_batch_running(self):
        if self._is_process_batch_running:
            return

        # create asyncio task to process batch
        asyncio.create_task(self._process_batch())
        self._is_process_batch_running = True

    def _get_batch_size(self, *args, **kwargs) -> int:
        raise NotImplementedError

    async def _process_batch(self):
        while True:
            # Wait until at least one item is available
            (first_args, first_kwargs), first_future = await self._queue.get()

            delays = [self._func.delay(*first_args, **first_kwargs)]
            size = self._get_batch_size(*first_args, **first_kwargs)
            futures = [first_future]

            # Try to gather more items into the same batch within a short timeout window
            while size <= self.batch_size:
                try:
                    # Wait for a new request for a short time window (e.g. 3ms)
                    # This allows batching multiple requests that arrive close in time.
                    (args, kwargs), future = await asyncio.wait_for(
                        self._queue.get(), timeout=self.batch_interval
                    )
                    size += self._get_batch_size(*args, **kwargs)
                    delays.append(self._func.delay(*args, **kwargs))
                    futures.append(future)
                except asyncio.TimeoutError:
                    # No new items arrived within the timeout window,
                    # stop collecting and start processing the current batch.
                    break

            logger.debug("Calling batch %s with %d size", self._func_name, size)

            try:
                results = self._func.batch(*delays)
                if inspect.isawaitable(results):
                    results = await results
            except Exception as e:  # Handle errors for the entire batch
                for fut in futures:
                    fut.set_exception(e)
            else:
                # Ensure the number of results matches the number of input futures
                assert len(results) == len(
                    futures
                ), f"#results should be equal to #futures, got {len(results)} and {len(futures)}"
                # Deliver the results to the corresponding waiting callers
                for fut, result in zip(futures, results):
                    fut.set_result(result)

    def _wrap_method(self):

        async def _replaced_async_method(model, *args, **kwargs):
            self._ensure_process_batch_running()
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            await self.queue.put(((args, kwargs), fut))
            return await fut

        return _replaced_async_method
