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
from concurrent.futures import Future

from xoscar.batch import _ExtensibleWrapper


class BatchMixin:
    allow_batch = True
    batch_size = 128

    def __init__(self, func: _ExtensibleWrapper):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._func = func
        setattr(self, func.__name__, types.MethodType(self._wrap_method(), self))

        # create asyncio task to process batch
        asyncio.create_task(self._process_batch())

    def _get_batch_size(self, *args, **kwargs) -> int:
        raise NotImplementedError

    async def _process_batch(self):
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

        results = self._func.batch(*delays)
        if inspect.isawaitable(results):
            results = await results

        assert len(results) == len(futures)
        for fut, result in zip(futures, results):
            fut.set_result(result)

    def _wrap_method(self):
        def _replaced_sync_method(self, *args, **kwargs):
            fut = Future()
            self._queue.put(((args, kwargs), fut))
            return fut.result()

        async def _replaced_async_method(self, *args, **kwargs):
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            self._queue.put(((args, kwargs), fut))
            return await fut

        if inspect.iscoroutinefunction(self._func):
            return _replaced_async_method
        else:
            return _replaced_sync_method
