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

import asyncio
from typing import Type

import xoscar as xo

from .actor.model import ModelActor
from .model.llm.core import Model


class API:
    def __init__(self, endpoint: str):
        self._endpoint = endpoint

    async def create_model(self, model_cls: Type[Model], *args, **kwargs):
        model = model_cls(*args, **kwargs)
        return await xo.create_actor(
            ModelActor, model, address=self._endpoint, uid=ModelActor.gen_uid(model)
        )


# Sync API
class SyncAPI:
    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()

    def create_model(self, model_cls: Type[Model], *args, **kwargs):
        model = model_cls(*args, **kwargs)
        coro = xo.create_actor(
            ModelActor, model, address=self._endpoint, uid=ModelActor.gen_uid(model)
        )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()
