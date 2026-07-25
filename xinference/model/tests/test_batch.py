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
from contextlib import suppress

import pytest
from xoscar import extensible

from ..batch import BatchMixin


class _BatchProbe(BatchMixin):
    def __init__(self, batch_size):
        self.calls = []
        BatchMixin.__init__(self, self.run, batch_size=batch_size, batch_interval=0.001)

    @extensible
    def run(self, values):
        return values

    @run.batch  # type: ignore
    async def run(self, args_list, kwargs_list):
        assert not any(kwargs_list)
        values = [args[0] for args in args_list]
        self.calls.append(values)
        return values

    def _get_batch_size(self, values):
        return len(values)


async def _shutdown_batch_processor(probe):
    task = probe._process_batch_task
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_sizes", "expected_batch_sizes"),
    [
        ([3, 2], [[3], [2]]),
        ([2, 2, 1], [[2, 2], [1]]),
        ([3, 2, 2], [[3], [2, 2]]),
        ([5, 1], [[5], [1]]),
    ],
)
async def test_batch_mixin_respects_batch_size_and_order(
    request_sizes, expected_batch_sizes
):
    probe = _BatchProbe(batch_size=4)
    requests = [
        [f"request-{request_index}-item-{item_index}" for item_index in range(size)]
        for request_index, size in enumerate(request_sizes)
    ]

    try:
        results = await asyncio.gather(*(probe.run(request) for request in requests))

        assert results == requests
        assert [
            [len(request) for request in batch] for batch in probe.calls
        ] == expected_batch_sizes
        assert [
            request[0].split("-item-")[0] for batch in probe.calls for request in batch
        ] == [f"request-{i}" for i in range(len(requests))]
    finally:
        await _shutdown_batch_processor(probe)
