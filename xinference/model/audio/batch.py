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
from collections import defaultdict
from typing import Any, Callable, Hashable, List

from xoscar import extensible

from ..batch import BatchMixin


class AudioRequestBatcher(BatchMixin):
    """Collect requests while running one native audio batch at a time."""

    def __init__(
        self,
        run_batch: Callable[[List[dict]], List[Any]],
        group_key: Callable[[dict], Hashable],
        request_batch_size: int = 4,
        batch_interval: float = 0.01,
    ):
        if request_batch_size < 1:
            raise ValueError("request_batch_size must be positive")
        if batch_interval < 0:
            raise ValueError("batch_interval must be nonnegative")
        self._run_batch = run_batch
        self._group_key = group_key
        super().__init__(
            self._enqueue,
            batch_size=request_batch_size,
            batch_interval=batch_interval,
        )
        # BatchMixin treats zero as the global default; audio allows no wait.
        self.batch_interval = batch_interval

    def _get_batch_size(self, *args, **kwargs) -> int:
        return 1

    async def submit(self, request: dict) -> Any:
        result = await self._enqueue(request)
        if isinstance(result, Exception):
            raise result
        return result

    @extensible
    def _enqueue(self, request: dict) -> Any:
        return self._run_batch([request])[0]

    @_enqueue.batch  # type: ignore
    async def _enqueue(self, args_list, kwargs_list):
        return await asyncio.to_thread(
            self._run_groups, [args[0] for args in args_list]
        )

    def _run_groups(self, requests: List[dict]) -> List[Any]:
        groups = defaultdict(list)
        results: List[Any] = [None] * len(requests)
        for index, request in enumerate(requests):
            try:
                groups[self._group_key(request)].append((index, request))
            except Exception as error:
                results[index] = error

        for group in groups.values():
            indices, items = zip(*group)
            try:
                outputs = self._run_batch(list(items))
                if len(outputs) != len(items):
                    raise RuntimeError(
                        "Audio batch returned an unexpected result count"
                    )
            except Exception as error:
                outputs = [error] * len(items)
            for index, output in zip(indices, outputs):
                results[index] = output
        return results
