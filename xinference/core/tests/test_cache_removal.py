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

# CI: linux

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ..supervisor import SupervisorActor


@pytest.mark.asyncio
async def test_remove_cache_reaches_every_worker_after_a_failure():
    failing = SimpleNamespace(confirm_and_remove_model=AsyncMock(return_value=False))
    healthy = SimpleNamespace(confirm_and_remove_model=AsyncMock(return_value=True))
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._worker_address_to_worker = {"w1:1": failing, "w2:2": healthy}

    assert not await SupervisorActor.confirm_and_remove_model(supervisor, "demo")

    for worker in (failing, healthy):
        worker.confirm_and_remove_model.assert_awaited_once_with(model_version="demo")
