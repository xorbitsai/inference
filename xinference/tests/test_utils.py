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
from concurrent.futures import Future

import pytest
from typing_extensions import Coroutine

from ..utils import AsyncRunner


@pytest.fixture
def async_runner():
    return AsyncRunner()


def test__thread_pool(async_runner):
    assert async_runner._thread_pool is not None


def test_run_as_future(async_runner):
    future = async_runner.run_as_future(lambda: 1)
    assert isinstance(future, Future)
    assert future.result() == 1


def test_async_run(async_runner):
    assert isinstance(async_runner.async_run(lambda: 1), Coroutine)


@pytest.mark.asyncio
async def test_async_run_a(async_runner):
    assert await async_runner.async_run(lambda: 1) == 1
