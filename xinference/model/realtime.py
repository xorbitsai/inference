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


"""Model-local realtime interfaces. No engine or media-specific dependencies."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict


class RealtimeSession(ABC):
    """An adapter is owned by one runtime session, never passed through RPC."""

    def __init__(self, output_queue_size: int = 8):
        self._outputs: asyncio.Queue = asyncio.Queue(maxsize=output_queue_size)
        self._closed = False

    async def emit(self, data: Dict[str, Any]) -> None:
        if not self._closed:
            await self._outputs.put(data)

    async def end_outputs(self) -> None:
        await self._outputs.put(None)

    async def iter_outputs(self) -> AsyncIterator[Dict[str, Any]]:
        while True:
            data = await self._outputs.get()
            if data is None:
                return
            yield data

    def notify_input(self, size: int) -> None:
        """Observe received bytes, including a frame awaiting bounded admission.

        Called on the event loop. Adapters sharing hints with native work must
        synchronize them; this hook must not run inference or block the reader.
        """

    def request_input_end(self) -> None:
        """Input end was observed, not necessarily processed. Never finalize here."""

    def request_cancel(self) -> None:
        """Fence output immediately; close() runs only after native work settles."""
        self._closed = True

    def observations(self) -> Dict[str, Any]:
        """Return bounded, media/text-free diagnostics after processing settles."""
        return {}

    @abstractmethod
    async def receive_input(self, frame: bytes) -> None:
        """Consume one frame locally; the runtime supplies bounded buffering."""

    @abstractmethod
    async def finish_input(self) -> None:
        """Finish accepted media, emit final output, then end_outputs()."""

    async def close(self, reason: str) -> None:
        self._closed = True
