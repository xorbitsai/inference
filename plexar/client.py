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
import uuid
from typing import Optional

import xoscar as xo

from .actor.model import ModelActor
from .actor.service import ControllerActor


class Client:
    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()

        self._controller_ref: Optional[
            xo.ActorRefType["ControllerActor"]
        ] = asyncio.run_coroutine_threadsafe(
            xo.actor_ref(address=self._endpoint, uid=ControllerActor.uid), self._loop
        ).result()

    @classmethod
    def gen_model_uid(cls) -> str:
        # generate a time-based uuid.
        return str(uuid.uuid1())

    def create_model(
        self,
        model_name: str,
        n_parameters_in_billions: Optional[int],
        fmt: Optional[str],
        quantization: Optional[str],
    ) -> xo.ActorRefType["ModelActor"]:
        model_uid = self.gen_model_uid()

        from plexar.model import MODEL_SPECS

        for model_spec in MODEL_SPECS:
            if model_spec.match(
                model_name, n_parameters_in_billions, fmt, quantization
            ):
                pass

        coro = self._controller_ref.launch_builtin_model(
            model_uid=model_uid,
            model_name=model_name,
        )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def terminate_model(self):
        pass

    def list_models(self):
        pass

    def get_model(self):
        pass

    def generate(self):
        # llm
        pass

    def transcribe(self):
        # speech.
        pass
