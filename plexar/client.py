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
import threading
import uuid
from typing import Any, Coroutine, List, Optional

import xoscar as xo

from .core.model import ModelActor
from .core.service import SupervisorActor
from .model import ModelSpec


class Isolation:
    # TODO: better move isolation to xoscar.
    def __init__(self, loop: asyncio.AbstractEventLoop, threaded: bool = True):
        self._loop = loop
        self._threaded = threaded

        self._stopped = None
        self._thread = None
        self._thread_ident = None

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._stopped = asyncio.Event()
        self._loop.run_until_complete(self._stopped.wait())

    def start(self):
        if self._threaded:
            self._thread = thread = threading.Thread(target=self._run)
            thread.daemon = True
            thread.start()
            self._thread_ident = thread.ident

    def call(self, coro: Coroutine) -> Any:
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    @property
    def thread_ident(self):
        return self._thread_ident

    async def _stop(self):
        self._stopped.set()

    def stop(self):
        if self._threaded:
            asyncio.run_coroutine_threadsafe(self._stop(), self._loop).result()
            self._thread.join()


class Client:
    def __init__(self, supervisor_address: str):
        self._supervisor_address = supervisor_address
        self._isolation = Isolation(asyncio.new_event_loop(), threaded=True)
        self._isolation.start()
        self._supervisor_ref: xo.ActorRefType["SupervisorActor"] = self._isolation.call(
            xo.actor_ref(address=self._supervisor_address, uid=SupervisorActor.uid())
        )

    @classmethod
    def gen_model_uid(cls) -> str:
        # generate a time-based uuid.
        return str(uuid.uuid1())

    def launch_model(
        self,
        model_name: str,
        model_size_in_billions: Optional[int] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs
    ) -> str:
        model_uid = self.gen_model_uid()

        coro = self._supervisor_ref.launch_builtin_model(
            model_uid=model_uid,
            model_name=model_name,
            model_size_in_billions=model_size_in_billions,
            model_format=model_format,
            quantization=quantization,
            **kwargs
        )
        self._isolation.call(coro)

        return model_uid

    def terminate_model(self, model_uid: str):
        coro = self._supervisor_ref.terminate_model(model_uid)
        return self._isolation.call(coro)

    def list_models(self) -> List[tuple[str, ModelSpec]]:
        coro = self._supervisor_ref.list_models()
        return self._isolation.call(coro)

    def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        coro = self._supervisor_ref.get_model(model_uid)
        return self._isolation.call(coro)
