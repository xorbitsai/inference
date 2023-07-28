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
import platform
import time
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import xoscar as xo

from ..core import ModelActor
from ..model.llm import LLMFamilyV1, LLMSpecV1
from .resource import ResourceStatus, gather_node_info

logger = getLogger(__name__)


DEFAULT_NODE_DEAD_TIMEOUT = 30
DEFAULT_NODE_CHECK_INTERVAL = 1


def log(func: Callable):
    # TODO: support non-async function
    import time
    from functools import wraps

    @wraps(func)
    async def wrapped(*args, **kwargs):
        logger.debug(f"Enter {func.__name__}, args: {args}, kwargs: {kwargs}")
        start = time.time()
        ret = await func(*args, **kwargs)
        logger.debug(
            f"Leave {func.__name__}, elapsed time: {int(time.time() - start)} ms"
        )
        return ret

    return wrapped


@dataclass
class WorkerStatus:
    update_time: float
    status: Dict[str, ResourceStatus]


class SupervisorActor(xo.Actor):
    def __init__(self):
        super().__init__()
        self._worker_address_to_worker: Dict[str, xo.ActorRefType[WorkerActor]] = {}
        self._model_uid_to_worker: Dict[str, xo.ActorRefType[WorkerActor]] = {}
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
                if time.time() - status.update_time > DEFAULT_NODE_DEAD_TIMEOUT:
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

    @log
    async def terminate_model(self, model_uid: str):
        if model_uid not in self._model_uid_to_worker:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        worker_ref = self._model_uid_to_worker[model_uid]
        await worker_ref.terminate_model(model_uid=model_uid)
        del self._model_uid_to_worker[model_uid]

    @log
    async def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        if model_uid not in self._model_uid_to_worker:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        worker_ref = self._model_uid_to_worker[model_uid]
        return await worker_ref.get_model(model_uid=model_uid)

    @log
    async def describe_model(self, model_uid: str) -> Dict[str, Any]:
        if model_uid not in self._model_uid_to_worker:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        worker_ref = self._model_uid_to_worker[model_uid]
        return await worker_ref.describe_model(model_uid=model_uid)

    @log
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        ret = {}
        for worker in self._worker_address_to_worker.values():
            ret.update(await worker.list_models())
        return ret

    def is_local_deployment(self) -> bool:
        # TODO: temporary.
        return (
            len(self._worker_address_to_worker) == 1
            and list(self._worker_address_to_worker)[0] == self.address
        )

    @log
    async def add_worker(self, worker_address: str):
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


class WorkerActor(xo.Actor):
    def __init__(self, supervisor_address: str, subpool_addresses: List[str]):
        super().__init__()
        self._supervisor_address = supervisor_address
        self._supervisor_ref = None
        self._model_uid_to_model: Dict[str, xo.ActorRefType["ModelActor"]] = {}
        self._model_uid_to_model_spec: Dict[
            str, Tuple[LLMFamilyV1, LLMSpecV1, str]
        ] = {}
        self._subpool_address_to_model_uids: Dict[str, Set[str]] = dict(
            [(subpool_address, set()) for subpool_address in subpool_addresses]
        )
        logger.debug(f"Worker actor initialized with subpools: {subpool_addresses}")

    @classmethod
    def uid(cls) -> str:
        return "worker"

    async def __post_create__(self):
        self._supervisor_ref: xo.ActorRefType["SupervisorActor"] = await xo.actor_ref(
            address=self._supervisor_address, uid=SupervisorActor.uid()
        )
        await self._supervisor_ref.add_worker(self.address)
        self._upload_task = asyncio.create_task(self._periodical_report_status())

    async def __pre_destroy__(self):
        self._upload_task.cancel()

    async def get_model_count(self) -> int:
        return len(self._model_uid_to_model)

    def _choose_subpool(self) -> str:
        min_running_model_count = None
        target_subpool_address = None
        for subpool_address in self._subpool_address_to_model_uids:
            running_model_count = len(
                self._subpool_address_to_model_uids[subpool_address]
            )
            if (
                min_running_model_count is None
                or running_model_count < min_running_model_count
            ):
                min_running_model_count = running_model_count
                target_subpool_address = subpool_address

        if target_subpool_address:
            logger.debug(
                "Subpool selected: %s, running model count: %d",
                target_subpool_address,
                min_running_model_count,
            )
            return target_subpool_address

        raise RuntimeError("No available slot found")

    def _check_model_is_valid(self, model_name):
        # baichuan-base and baichuan-chat depend on `cpm_kernels` module,
        # but `cpm_kernels` cannot run on Darwin system.
        if platform.system() == "Darwin":
            if model_name in ["baichuan-base", "baichuan-chat"]:
                raise ValueError(f"{model_name} model can't run on Darwin system.")

    @staticmethod
    def _to_llm_description(
        llm_family: LLMFamilyV1, llm_spec: LLMSpecV1, quantization: str
    ) -> Dict[str, Any]:
        return {
            "model_type": "LLM",
            "model_name": llm_family.model_name,
            "model_lang": llm_family.model_lang,
            "model_ability": llm_family.model_ability,
            "model_description": llm_family.model_description,
            "model_format": llm_spec.model_format,
            "model_size_in_billions": llm_spec.model_size_in_billions,
            "quantization": quantization,
        }

    @log
    async def launch_builtin_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[int],
        model_format: Optional[str],
        quantization: Optional[str],
        **kwargs,
    ) -> xo.ActorRefType["ModelActor"]:
        assert model_uid not in self._model_uid_to_model
        self._check_model_is_valid(model_name)

        from ..model.llm import match_llm, match_llm_cls

        assert self._supervisor_ref is not None
        match_result = match_llm(
            model_name,
            model_format,
            model_size_in_billions,
            quantization,
            await self._supervisor_ref.is_local_deployment(),
        )
        if not match_result:
            raise ValueError(
                f"Model not found, name: {model_name}, format: {model_format},"
                f" size: {model_size_in_billions}, quantization: {quantization}"
            )
        llm_family, llm_spec, quantization = match_result
        assert quantization is not None

        from ..model.llm.llm_family import cache

        save_path = await asyncio.to_thread(cache, llm_family, llm_spec, quantization)

        llm_cls = match_llm_cls(llm_family, llm_spec)
        if not llm_cls:
            raise ValueError(
                f"Model not supported, name: {model_name}, format: {model_format},"
                f" size: {model_size_in_billions}, quantization: {quantization}"
            )

        model = llm_cls(
            model_uid, llm_family, llm_spec, quantization, save_path, kwargs
        )
        subpool_address = self._choose_subpool()
        model_ref = await xo.create_actor(
            ModelActor, address=subpool_address, uid=model_uid, model=model
        )
        await model_ref.load()
        self._model_uid_to_model[model_uid] = model_ref
        self._model_uid_to_model_spec[model_uid] = (llm_family, llm_spec, quantization)
        self._subpool_address_to_model_uids[subpool_address].add(model_uid)
        return model_ref

    @log
    async def terminate_model(self, model_uid: str):
        if model_uid not in self._model_uid_to_model:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        model_ref = self._model_uid_to_model[model_uid]

        await xo.destroy_actor(model_ref)
        del self._model_uid_to_model[model_uid]
        del self._model_uid_to_model_spec[model_uid]
        for subpool_address in self._subpool_address_to_model_uids:
            if model_uid in self._subpool_address_to_model_uids[subpool_address]:
                self._subpool_address_to_model_uids[subpool_address].remove(model_uid)

    @log
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        ret = {}
        for k, v in self._model_uid_to_model_spec.items():
            ret[k] = self._to_llm_description(v[0], v[1], v[2])
        return ret

    @log
    async def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        if model_uid not in self._model_uid_to_model:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        return self._model_uid_to_model[model_uid]

    @log
    async def describe_model(self, model_uid: str) -> Dict[str, Any]:
        if model_uid not in self._model_uid_to_model:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        llm_family, llm_spec, quantization = self._model_uid_to_model_spec[model_uid]
        return self._to_llm_description(llm_family, llm_spec, quantization)

    async def report_status(self):
        status = await asyncio.to_thread(gather_node_info)
        await self._supervisor_ref.report_worker_status(self.address, status)

    async def _periodical_report_status(self):
        while True:
            try:
                await self.report_status()
            except asyncio.CancelledError:  # pragma: no cover
                break
            except RuntimeError as ex:  # pragma: no cover
                if "cannot schedule new futures" not in str(ex):
                    # when atexit is triggered, the default pool might be shutdown
                    # and to_thread will fail
                    break
            except (
                Exception
            ) as ex:  # pragma: no cover  # noqa: E722  # nosec  # pylint: disable=bare-except
                logger.error(f"Failed to upload node info: {ex}")
            try:
                await asyncio.sleep(DEFAULT_NODE_CHECK_INTERVAL)
            except asyncio.CancelledError:  # pragma: no cover
                break
