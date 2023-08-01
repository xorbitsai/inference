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
from typing import Any, Dict, Optional

import xoscar as xo

from ..isolation import Isolation
from . import ModelActor
from .supervisor import SupervisorActor


class AsyncSupervisorAPI:
    def __init__(self, supervisor_address: str):
        self._supervisor_address = supervisor_address
        self._supervisor_ref = None

    async def _get_supervisor_ref(self) -> xo.ActorRefType["SupervisorActor"]:
        if self._supervisor_ref is None:
            self._supervisor_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=SupervisorActor.uid()
            )
        return self._supervisor_ref

    async def launch_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[int] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> str:
        supervisor_ref = await self._get_supervisor_ref()
        await supervisor_ref.launch_builtin_model(
            model_uid=model_uid,
            model_name=model_name,
            model_size_in_billions=model_size_in_billions,
            model_format=model_format,
            quantization=quantization,
            **kwargs,
        )
        return model_uid

    async def terminate_model(self, model_uid: str):
        supervisor_ref = await self._get_supervisor_ref()
        await supervisor_ref.terminate_model(model_uid)

    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        supervisor_ref = await self._get_supervisor_ref()
        return await supervisor_ref.list_models()

    async def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        supervisor_ref = await self._get_supervisor_ref()
        return await supervisor_ref.get_model(model_uid)

    async def is_local_deployment(self) -> bool:
        # TODO: temporary.
        supervisor_ref = await self._get_supervisor_ref()
        return await supervisor_ref.is_local_deployment()


class SyncSupervisorAPI:
    def __init__(self, supervisor_address: str):
        self._supervisor_address = supervisor_address
        self._supervisor_ref = None
        self._isolation = Isolation(asyncio.new_event_loop(), threaded=True)
        self._isolation.start()

    async def _get_supervisor_ref(self) -> xo.ActorRefType["SupervisorActor"]:
        if self._supervisor_ref is None:
            self._supervisor_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=SupervisorActor.uid()
            )
        return self._supervisor_ref

    def launch_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[int] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> str:
        async def _launch_model():
            supervisor_ref = await self._get_supervisor_ref()
            await supervisor_ref.launch_builtin_model(
                model_uid=model_uid,
                model_name=model_name,
                model_size_in_billions=model_size_in_billions,
                model_format=model_format,
                quantization=quantization,
                **kwargs,
            )
            return model_uid

        return self._isolation.call(_launch_model())

    def terminate_model(self, model_uid: str):
        async def _terminate_model():
            supervisor_ref = await self._get_supervisor_ref()
            await supervisor_ref.terminate_model(model_uid)

        return self._isolation.call(_terminate_model())

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        async def _list_models():
            supervisor_ref = await self._get_supervisor_ref()
            return await supervisor_ref.list_models()

        return self._isolation.call(_list_models())

    def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        async def _get_model():
            supervisor_ref = await self._get_supervisor_ref()
            return await supervisor_ref.get_model(model_uid)

        return self._isolation.call(_get_model())

    def is_local_deployment(self) -> bool:
        # TODO: temporary.
        async def _is_local_deployment():
            supervisor_ref = await self._get_supervisor_ref()
            return await supervisor_ref.is_local_deployment()

        return self._isolation.call(_is_local_deployment())
