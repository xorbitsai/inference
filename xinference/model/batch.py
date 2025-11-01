# Copyright 2022-2025 XProbe Inc.
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
import types

from xoscar.batch import _ExtensibleWrapper


class BatchMixin:
    allow_batch = True
    batch_size = 128

    def __init__(self, func: _ExtensibleWrapper):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._func = func
        setattr(self, func.func.__name__, types.MethodType(self._wrap_method(), self))

        self._is_process_batch_running = False

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
            # wait until there is object to process
            (first_args, first_kwargs), first_future = await self._queue.get()

            delays = [self._func.delay(*first_args, **first_kwargs)]
            size = self._get_batch_size(*first_args, **first_kwargs)
            futures = [first_future]

            while not self._queue.empty() and size <= self.batch_size:
                (args, kwargs), future = await self._queue.get()
                size += self._get_batch_size(*args, **kwargs)
                delays.append(self._func.delay(*args, **kwargs))
                futures.append(future)

            try:
                results = self._func.batch(*delays)
                if inspect.isawaitable(results):
                    results = await results
            except Exception as e:  # noqa: E722
                # Failed, set exception to all futures
                for fut in futures:
                    fut.set_exception(e)
            else:
                assert len(results) == len(
                    futures
                ), f"#results should be equal to #futures, got {len(results)} and {len(futures)}"
                for fut, result in zip(futures, results):
                    fut.set_result(result)

    def _wrap_method(self):

        async def _replaced_async_method(model, *args, **kwargs):
            self._ensure_process_batch_running()
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            await self._queue.put(((args, kwargs), fut))
            return await fut

        return _replaced_async_method
