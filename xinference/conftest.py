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


import pytest_asyncio
import xoscar as xo

from .core.service import SupervisorActor


@pytest_asyncio.fixture
async def setup():
    from .deploy.utils import create_worker_actor_pool
    from .deploy.worker import start_worker_components

    pool = await create_worker_actor_pool("127.0.0.1:9998")
    await xo.create_actor(
        SupervisorActor, address=pool.external_address, uid=SupervisorActor.uid()
    )
    await start_worker_components(
        address=pool.external_address, supervisor_address=pool.external_address
    )

    async with pool:
        yield pool
