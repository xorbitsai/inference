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
from typing import Any, Coroutine


class Isolation:
    # TODO: better move isolation to xoscar.
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        threaded: bool = True,
        daemon: bool = True,
    ):
        self._loop = loop
        self._threaded = threaded

        self._stopped = None
        self._thread = None
        self._thread_ident = None
        self._daemon = daemon

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._stopped = asyncio.Event()
        self._loop.run_until_complete(self._stopped.wait())

    def start(self):
        if self._threaded:
            self._thread = thread = threading.Thread(target=self._run)
            if self._daemon:
                thread.daemon = True
            thread.start()
            self._thread_ident = thread.ident

    def call(self, coro: Coroutine) -> Any:
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    @property
    def thread_ident(self):
        return self._thread_ident

    @property
    def loop(self):
        return self._loop

    async def _stop(self):
        self._stopped.set()

    def stop(self):
        if self._threaded:
            asyncio.run_coroutine_threadsafe(self._stop(), self._loop).result()
            self._thread.join()
