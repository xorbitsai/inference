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
from collections import deque

import xoscar as xo


class InferenceRequest:
    def __init__(self, prompt, is_prefill, *args, **kwargs):
        self._prompt = prompt
        self._is_prefill = is_prefill
        self._new_tokens = []
        self._inference_args = args
        self._inference_kwargs = kwargs


class SchedulerActor(xo.Actor):
    @classmethod
    def gen_uid(cls, model_uid: str, replica_id: str):
        return f"{model_uid}-{replica_id}-scheduler-actor"

    def __init__(self):
        super().__init__()
        self._waiting_queue = deque()
        self._running_queue = deque()
        self._model = None

    def _handle_request(self):
        res = []
        while len(self._waiting_queue) > 0:
            res.append(self._waiting_queue.popleft())
        while len(self._running_queue) > 0:
            res.append(self._running_queue.popleft())
        return res

    async def step(self):
        req_list = self._handle_request()
        if not req_list:
            return
        self._model.batch_inference(req_list)

    async def add_request(self, prompt: str, *args, **kwargs):
        req = InferenceRequest(prompt, True, *args, **kwargs)
        self._waiting_queue.append(req)

    async def run(self):
        while True:
            await self.step()
            await asyncio.sleep(1)
