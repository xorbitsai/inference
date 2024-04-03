# Copyright 2022-2023 XProbe Inc.
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
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor

import torch
from typing_extensions import Callable, Optional, TypeVar


def cuda_count():
    # even if install torch cpu, this interface would return 0.
    return torch.cuda.device_count()


R = TypeVar("R")  # Return type


class AsyncRunner(object):
    def __init__(self):
        self.__thread_pool: Optional[ThreadPoolExecutor] = None

    @property
    def _thread_pool(self):
        if self.__thread_pool is None:
            self.__thread_pool = ThreadPoolExecutor(max_workers=1)
        return self.__thread_pool

    def run_as_future(self, fn: Callable[..., R], *args, **kwargs) -> Future[R]:
        return self._thread_pool.submit(fn, *args, **kwargs)

    async def async_run(self, fn: Callable[..., R], *args, **kwargs) -> R:
        return await asyncio.wrap_future(self.run_as_future(fn, *args, **kwargs))
