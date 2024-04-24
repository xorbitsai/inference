# Copyright 2022-2024 XProbe Inc.
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

import xoscar as xo


class SchedulerActor(xo.Actor):
    @classmethod
    def gen_uid(cls, model_uid: str, replica_id: str):
        return f"{model_uid}-{replica_id}-scheduler-actor"

    def __init__(self):
        super().__init__()

    async def step(self):
        print("===========This is a step!!!!")

    async def run(self):
        while True:
            await self.step()
            await asyncio.sleep(1)
