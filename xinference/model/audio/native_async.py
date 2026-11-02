# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Ownership of asynchronous audio engines and their individual requests."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Optional


async def settle_request(awaitable) -> Any:
    """Retain request capacity through repeated cancellation until abort settles."""
    task = asyncio.ensure_future(awaitable)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


class NativeAsyncAudioModel:
    allow_batch = True
    realtime_native_async = True

    def __init__(self, model_uid: str, model_path: str, model_spec, **kwargs):
        self.model_family = self._model_spec = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._kwargs = dict(kwargs)
        self._kwargs.pop("device", None)
        self._engine: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._requests: set[str] = set()
        self._aborts: dict[str, asyncio.Task] = {}
        self._stopping = False

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def load(self) -> None:
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError(
                "Native audio engines require the model actor event loop"
            )
        asyncio.run_coroutine_threadsafe(self._load(), self._loop).result()

    async def _load(self) -> None:
        raise NotImplementedError

    async def _abort(self, request_id: str, engine=None) -> None:
        # stop() and a cancelling client may own the same abort concurrently.
        # Share its acknowledgement before shutdown, even across cancellation.
        if request_id not in self._aborts:
            engine = self._engine if engine is None else engine

            async def abort():
                result = engine.abort(request_id)
                if inspect.isawaitable(result):
                    await result

            self._aborts[request_id] = asyncio.create_task(abort())
        await settle_request(self._aborts[request_id])

    async def _generate(self, prompt, **kwargs):
        import uuid

        if self._stopping or self._engine is None:
            raise RuntimeError("Audio engine is not ready")
        request_id = uuid.uuid4().hex
        engine = self._engine
        self._requests.add(request_id)
        generator = None
        completed = False
        try:
            generator = engine.generate(prompt=prompt, request_id=request_id, **kwargs)
            async for output in generator:
                if self._stopping:
                    raise RuntimeError("Audio generation interrupted during shutdown")
                yield output
            if self._stopping:
                raise RuntimeError("Audio generation interrupted during shutdown")
            completed = True
        finally:
            try:
                if not completed:
                    await self._abort(request_id, engine)
                if generator is not None:
                    await settle_request(generator.aclose())
            except BaseException:
                self._stopping = True
                raise
            finally:
                self._requests.discard(request_id)
                self._aborts.pop(request_id, None)

    async def _stop(self) -> None:
        self._stopping = True
        if self._engine is None:
            return
        try:
            for request_id in tuple(self._requests):
                await self._abort(request_id)
        finally:
            try:
                result = self._engine.shutdown()
                if inspect.isawaitable(result):
                    await result
            finally:
                self._engine = None

    def stop(self) -> None:
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._stop(), self._loop).result()
