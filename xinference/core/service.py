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
from typing import Callable, Dict, List, Optional, Tuple

import xoscar as xo

from ..core import ModelActor
from ..model import ModelSpec
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

            return target_worker

        raise RuntimeError("TODO")

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
        assert model_uid not in self._model_uid_to_worker

        worker_ref = await self._choose_worker()
        model_ref = await worker_ref.launch_builtin_model(
            model_uid=model_uid,
            model_name=model_name,
            model_size_in_billions=model_size_in_billions,
            model_format=model_format,
            quantization=quantization,
            **kwargs,
        )
        self._model_uid_to_worker[model_uid] = worker_ref

        return model_ref

    async def _check_dead_nodes(self):
        while True:
            for address, status in self._worker_status.items():
                if time.time() - status.update_time > DEFAULT_NODE_DEAD_TIMEOUT:
                    self._worker_status.pop(address)
                    self._worker_address_to_worker.pop(address)
            await asyncio.sleep(5)

    @log
    async def terminate_model(self, model_uid: str):
        assert model_uid in self._model_uid_to_worker

        worker_ref = self._model_uid_to_worker[model_uid]
        await worker_ref.terminate_model(model_uid=model_uid)
        del self._model_uid_to_worker[model_uid]

    @log
    async def get_model(self, model_uid: str):
        assert model_uid in self._model_uid_to_worker

        worker_ref = self._model_uid_to_worker[model_uid]
        return await worker_ref.get_model(model_uid=model_uid)

    @log
    async def list_models(self) -> List[Tuple[str, ModelSpec]]:
        ret = []
        for worker in self._worker_address_to_worker.values():
            ret.extend(await worker.list_models())
        return ret

    @log
    async def add_worker(self, worker_address: str):
        assert worker_address not in self._worker_address_to_worker

        worker_ref = await xo.actor_ref(address=worker_address, uid=WorkerActor.uid())
        self._worker_address_to_worker[worker_address] = worker_ref

    @log
    async def report_worker_status(
        self, worker_address: str, status: Dict[str, ResourceStatus]
    ):
        self._worker_status[worker_address] = WorkerStatus(
            update_time=time.time(), status=status
        )


class WorkerActor(xo.Actor):
    def __init__(self, supervisor_address: str):
        super().__init__()
        self._supervisor_address = supervisor_address
        self._supervisor_ref = None
        self._model_uid_to_model: Dict[str, xo.ActorRefType["ModelActor"]] = {}
        self._model_uid_to_model_spec: Dict[str, ModelSpec] = {}

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

        from ..model import MODEL_FAMILIES

        for model_family in MODEL_FAMILIES:
            model_spec = model_family.match(
                model_name=model_name,
                model_format=model_format,
                model_size_in_billions=model_size_in_billions,
                quantization=quantization,
            )

            if model_spec is None:
                continue

            cls = model_family.cls
            save_path = model_family.cache(
                model_spec.model_size_in_billions, model_spec.quantization
            )
            model = cls(model_uid, model_spec, save_path, kwargs)
            model_ref = await xo.create_actor(
                ModelActor, address=self.address, uid=model_uid, model=model
            )
            self._model_uid_to_model[model_uid] = model_ref
            self._model_uid_to_model_spec[model_uid] = model_spec
            return model_ref

        raise ValueError(
            f"Model not found, name: {model_name}, format: {model_format},"
            f" size: {model_size_in_billions}, quantization: {quantization}"
        )

    @log
    async def terminate_model(self, model_uid: str):
        assert model_uid in self._model_uid_to_model

        model_ref = self._model_uid_to_model[model_uid]
        await xo.destroy_actor(model_ref)
        del self._model_uid_to_model[model_uid]
        del self._model_uid_to_model_spec[model_uid]

    @log
    async def list_models(self) -> List[Tuple[str, ModelSpec]]:
        ret = []
        for k, v in self._model_uid_to_model_spec.items():
            ret.append((k, v))
        return ret

    @log
    async def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        assert model_uid in self._model_uid_to_model

        return self._model_uid_to_model[model_uid]

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
