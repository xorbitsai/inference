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
import time
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import xoscar as xo

from ..core import ModelActor
from .resource import ResourceStatus
from .utils import log_async, log_sync

if TYPE_CHECKING:
    from .worker import WorkerActor

logger = getLogger(__name__)


DEFAULT_NODE_TIMEOUT = 30


@dataclass
class WorkerStatus:
    update_time: float
    status: Dict[str, ResourceStatus]


class SupervisorActor(xo.Actor):
    def __init__(self):
        super().__init__()
        self._worker_address_to_worker: Dict[str, xo.ActorRefType["WorkerActor"]] = {}
        self._model_uid_to_worker: Dict[str, xo.ActorRefType["WorkerActor"]] = {}
        self._worker_status: Dict[str, WorkerStatus] = {}

    @classmethod
    def uid(cls) -> str:
        return "supervisor"

    async def __post_create__(self):
        self._check_dead_nodes_task = asyncio.create_task(self._check_dead_nodes())

    async def __pre_destroy__(self):
        self._check_dead_nodes_task.cancel()

    async def _choose_worker(self) -> xo.ActorRefType["WorkerActor"]:
        # TODO: better allocation strategy.
        min_running_model_count = None
        target_worker = None
        for worker in self._worker_address_to_worker.values():
            running_model_count = await worker.get_model_count()
            if (
                min_running_model_count is None
                or running_model_count < min_running_model_count
            ):
                min_running_model_count = running_model_count
                target_worker = worker

        if target_worker:
            return target_worker

        raise RuntimeError("No available worker found")

    @log_sync(logger=logger)
    def list_model_registrations(self, model_type: str) -> List[Dict[str, Any]]:
        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families

            ret = [
                {"model_name": f.model_name, "is_builtin": True}
                for f in BUILTIN_LLM_FAMILIES
            ]
            user_defined_llm_families = get_user_defined_llm_families()
            ret.extend(
                [
                    {"model_name": f.model_name, "is_builtin": False}
                    for f in user_defined_llm_families
                ]
            )

            return ret
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_sync(logger=logger)
    def get_model_registration(
        self, model_type: str, model_name: str
    ) -> Dict[str, Any]:
        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families

            for f in BUILTIN_LLM_FAMILIES + get_user_defined_llm_families():
                if f.model_name == model_name:
                    return f

            raise ValueError(f"Model {model_name} not found")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def register_model(self, model_type: str, model: str, persist: bool):
        if model_type == "LLM":
            from ..model.llm import LLMFamilyV1, register_llm

            llm_family = LLMFamilyV1.parse_raw(model)
            register_llm(llm_family, persist)

            if not self.is_local_deployment:
                for worker in self._worker_address_to_worker.values():
                    await worker.register_model(model_type, model, persist)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def unregister_model(self, model_type: str, model_name: str):
        if model_type == "LLM":
            from ..model.llm import unregister_llm

            unregister_llm(model_name)

            if not self.is_local_deployment:
                for worker in self._worker_address_to_worker.values():
                    await worker.unregister_model(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    async def launch_builtin_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[int],
        model_format: Optional[str],
        quantization: Optional[str],
        **kwargs,
    ) -> xo.ActorRefType["ModelActor"]:
        logger.debug(
            (
                f"Enter launch_builtin_model, model_uid: %s, model_name: %s, model_size: %s, "
                f"model_format: %s, quantization: %s"
            ),
            model_uid,
            model_name,
            str(model_size_in_billions) if model_size_in_billions else "",
            model_format,
            quantization,
        )

        if model_uid in self._model_uid_to_worker:
            raise ValueError(f"Model is already in the model list, uid: {model_uid}")

        worker_ref = await self._choose_worker()
        model_ref = yield worker_ref.launch_builtin_model(
            model_uid=model_uid,
            model_name=model_name,
            model_size_in_billions=model_size_in_billions,
            model_format=model_format,
            quantization=quantization,
            **kwargs,
        )
        # TODO: not protected.
        self._model_uid_to_worker[model_uid] = worker_ref

        raise xo.Return(model_ref)

    async def _check_dead_nodes(self):
        while True:
            dead_nodes = []
            for address, status in self._worker_status.items():
                if time.time() - status.update_time > DEFAULT_NODE_TIMEOUT:
                    dead_models = []
                    for model_uid in self._model_uid_to_worker:
                        if self._model_uid_to_worker[model_uid].address == address:
                            dead_models.append(model_uid)
                    logger.error(
                        "Worker timeout. address: %s, influenced models: %s",
                        address,
                        dead_models,
                    )
                    dead_nodes.append(address)

            for address in dead_nodes:
                self._worker_status.pop(address)
                self._worker_address_to_worker.pop(address)
            await asyncio.sleep(5)

    @log_async(logger=logger)
    async def terminate_model(self, model_uid: str):
        if model_uid not in self._model_uid_to_worker:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        worker_ref = self._model_uid_to_worker[model_uid]
        await worker_ref.terminate_model(model_uid=model_uid)
        del self._model_uid_to_worker[model_uid]

    @log_async(logger=logger)
    async def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        if model_uid not in self._model_uid_to_worker:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        worker_ref = self._model_uid_to_worker[model_uid]
        return await worker_ref.get_model(model_uid=model_uid)

    @log_async(logger=logger)
    async def describe_model(self, model_uid: str) -> Dict[str, Any]:
        if model_uid not in self._model_uid_to_worker:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        worker_ref = self._model_uid_to_worker[model_uid]
        return await worker_ref.describe_model(model_uid=model_uid)

    @log_async(logger=logger)
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        ret = {}
        for worker in self._worker_address_to_worker.values():
            ret.update(await worker.list_models())
        return ret

    @log_sync(logger=logger)
    def is_local_deployment(self) -> bool:
        # TODO: temporary.
        return (
            len(self._worker_address_to_worker) == 1
            and list(self._worker_address_to_worker)[0] == self.address
        )

    @log_async(logger=logger)
    async def add_worker(self, worker_address: str):
        from .worker import WorkerActor

        assert worker_address not in self._worker_address_to_worker

        worker_ref = await xo.actor_ref(address=worker_address, uid=WorkerActor.uid())
        self._worker_address_to_worker[worker_address] = worker_ref
        logger.info("Worker %s has been added successfully", worker_address)

    async def report_worker_status(
        self, worker_address: str, status: Dict[str, ResourceStatus]
    ):
        self._worker_status[worker_address] = WorkerStatus(
            update_time=time.time(), status=status
        )
