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
import itertools
import time
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import xoscar as xo

from ..core import ModelActor
from .resource import ResourceStatus
from .utils import (
    build_replica_model_uid,
    is_valid_model_uid,
    iter_replica_model_uid,
    log_async,
    log_sync,
    parse_replica_model_uid,
)

if TYPE_CHECKING:
    from ..model.embedding import EmbeddingModelSpec
    from ..model.image import ImageModelFamilyV1
    from ..model.llm import LLMFamilyV1
    from .worker import WorkerActor


logger = getLogger(__name__)


DEFAULT_NODE_TIMEOUT = 60


@dataclass
class WorkerStatus:
    update_time: float
    status: Dict[str, ResourceStatus]


@dataclass
class ReplicaInfo:
    replica: int
    scheduler: Iterator


class SupervisorActor(xo.StatelessActor):
    def __init__(self):
        super().__init__()
        self._worker_address_to_worker: Dict[str, xo.ActorRefType["WorkerActor"]] = {}
        self._worker_status: Dict[str, WorkerStatus] = {}
        self._replica_model_uid_to_worker: Dict[
            str, xo.ActorRefType["WorkerActor"]
        ] = {}
        self._model_uid_to_replica_info: Dict[str, ReplicaInfo] = {}
        self._uptime = None
        self._lock = asyncio.Lock()

    @classmethod
    def uid(cls) -> str:
        return "supervisor"

    async def __post_create__(self):
        self._uptime = time.time()
        # comment this line to avoid worker lost
        # self._check_dead_nodes_task = asyncio.create_task(self._check_dead_nodes())
        logger.info(f"Xinference supervisor {self.address} started")

    async def _choose_worker(self) -> xo.ActorRefType["WorkerActor"]:
        # TODO: better allocation strategy.
        min_running_model_count = None
        target_worker = None

        workers = list(self._worker_address_to_worker.values())
        for worker in workers:
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
    def get_status(self) -> Dict:
        return {
            "uptime": int(time.time() - self._uptime),
            "workers": self._worker_status,
        }

    def _to_llm_reg(
        self, llm_family: "LLMFamilyV1", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.llm import get_cache_status

        if self.is_local_deployment():
            specs = []
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            for spec in llm_family.model_specs:
                cache_status = get_cache_status(llm_family, spec)
                specs.append({**spec.dict(), "cache_status": cache_status})
            return {**llm_family.dict(), "is_builtin": is_builtin, "model_specs": specs}
        else:
            return {**llm_family.dict(), "is_builtin": is_builtin}

    def _to_embedding_model_reg(
        self, model_spec: "EmbeddingModelSpec", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.embedding import get_cache_status

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_spec)
            return {
                **model_spec.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            return {
                **model_spec.dict(),
                "is_builtin": is_builtin,
            }

    def _to_image_model_reg(
        self, model_family: "ImageModelFamilyV1", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.image import get_cache_status

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_family)
            return {
                **model_family.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            return {
                **model_family.dict(),
                "is_builtin": is_builtin,
            }

    @log_sync(logger=logger)
    def list_model_registrations(
        self, model_type: str, detailed: bool = False
    ) -> List[Dict[str, Any]]:
        def sort_helper(item):
            assert isinstance(item["model_name"], str)
            return item.get("model_name").lower()

        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families

            ret = []
            for family in BUILTIN_LLM_FAMILIES:
                if detailed:
                    ret.append(self._to_llm_reg(family, True))
                else:
                    ret.append({"model_name": family.model_name, "is_builtin": True})

            for family in get_user_defined_llm_families():
                if detailed:
                    ret.append(self._to_llm_reg(family, False))
                else:
                    ret.append({"model_name": family.model_name, "is_builtin": False})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "embedding":
            from ..model.embedding import BUILTIN_EMBEDDING_MODELS
            from ..model.embedding.custom import get_user_defined_embeddings

            ret = []
            for model_name, family in BUILTIN_EMBEDDING_MODELS.items():
                if detailed:
                    ret.append(self._to_embedding_model_reg(family, is_builtin=True))
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            for model_spec in get_user_defined_embeddings():
                if detailed:
                    ret.append(
                        self._to_embedding_model_reg(model_spec, is_builtin=False)
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "image":
            from ..model.image import BUILTIN_IMAGE_MODELS

            ret = []
            for model_name, family in BUILTIN_IMAGE_MODELS.items():
                if detailed:
                    ret.append(self._to_image_model_reg(family, is_builtin=True))
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            ret.sort(key=sort_helper)
            return ret
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_sync(logger=logger)
    def get_model_registration(self, model_type: str, model_name: str) -> Any:
        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families

            for f in BUILTIN_LLM_FAMILIES + get_user_defined_llm_families():
                if f.model_name == model_name:
                    return f

            raise ValueError(f"Model {model_name} not found")
        if model_type == "embedding":
            from ..model.embedding import BUILTIN_EMBEDDING_MODELS
            from ..model.embedding.custom import get_user_defined_embeddings

            for f in (
                list(BUILTIN_EMBEDDING_MODELS.values()) + get_user_defined_embeddings()
            ):
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        if model_type == "image":
            from ..model.image import BUILTIN_IMAGE_MODELS

            for f in BUILTIN_IMAGE_MODELS.values():
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def register_model(self, model_type: str, model: str, persist: bool):
        if model_type == "LLM" or model_type == "embedding":
            from ..model.embedding import CustomEmbeddingModelSpec, register_embedding
            from ..model.llm import LLMFamilyV1, register_llm

            if not self.is_local_deployment():
                workers = list(self._worker_address_to_worker.values())
                for worker in workers:
                    await worker.register_model(model_type, model, persist)

            model_spec = (
                CustomEmbeddingModelSpec.parse_raw(model)
                if model_type == "embedding"
                else LLMFamilyV1.parse_raw(model)
            )
            register_fn = (
                register_embedding if model_type == "embedding" else register_llm
            )
            register_fn(model_spec, persist)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def unregister_model(self, model_type: str, model_name: str):
        if model_type == "LLM" or model_type == "embedding":
            from ..model.embedding import unregister_embedding
            from ..model.llm import unregister_llm

            unregister_fn = (
                unregister_llm if model_type == "LLM" else unregister_embedding
            )
            unregister_fn(model_name)

            if not self.is_local_deployment():
                workers = list(self._worker_address_to_worker.values())
                for worker in workers:
                    await worker.unregister_model(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    async def launch_speculative_llm(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[int],
        quantization: Optional[str],
        draft_model_name: str,
        draft_model_size_in_billions: Optional[int],
        draft_quantization: Optional[str],
        n_gpu: Optional[Union[int, str]] = "auto",
    ) -> str:
        logger.debug(
            (
                f"Enter launch_speculative_llm, model_uid: %s, model_name: %s, model_size: %s, "
                f"draft_model_name: %s, draft_model_size: %s"
            ),
            model_uid,
            model_name,
            str(model_size_in_billions) if model_size_in_billions else "",
            draft_model_name,
            draft_model_size_in_billions,
        )

        # TODO: the draft and target model must be on the same worker.
        if not self.is_local_deployment():
            raise ValueError(
                "Speculative model is not supported in distributed deployment yet."
            )

        if model_uid in self._model_uid_to_replica_info:
            raise ValueError(f"Model is already in the model list, uid: {model_uid}")

        worker_ref = await self._choose_worker()
        replica = 1
        self._model_uid_to_replica_info[model_uid] = ReplicaInfo(
            replica=replica, scheduler=itertools.cycle(range(replica))
        )

        try:
            rep_model_uid = f"{model_uid}-{1}-{0}"
            await worker_ref.launch_speculative_model(
                model_uid=rep_model_uid,
                model_name=model_name,
                model_size_in_billions=model_size_in_billions,
                quantization=quantization,
                draft_model_name=draft_model_name,
                draft_model_size_in_billions=draft_model_size_in_billions,
                draft_quantization=draft_quantization,
                n_gpu=n_gpu,
            )
            self._replica_model_uid_to_worker[rep_model_uid] = worker_ref

        except Exception:
            # terminate_model will remove the replica info.
            await self.terminate_model(model_uid, suppress_exception=True)
            raise
        return model_uid

    async def launch_builtin_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[int],
        model_format: Optional[str],
        quantization: Optional[str],
        model_type: Optional[str],
        replica: int = 1,
        n_gpu: Optional[Union[int, str]] = "auto",
        request_limits: Optional[int] = None,
        **kwargs,
    ) -> str:
        logger.debug(
            (
                f"Enter launch_builtin_model, model_uid: %s, model_name: %s, model_size: %s, "
                f"model_format: %s, quantization: %s, replica: %s"
            ),
            model_uid,
            model_name,
            str(model_size_in_billions) if model_size_in_billions else "",
            model_format,
            quantization,
            replica,
        )

        async def _launch_one_model(_replica_model_uid):
            if _replica_model_uid in self._replica_model_uid_to_worker:
                raise ValueError(
                    f"Model is already in the model list, uid: {_replica_model_uid}"
                )

            nonlocal model_type
            worker_ref = await self._choose_worker()
            # LLM as default for compatibility
            model_type = model_type or "LLM"
            await worker_ref.launch_builtin_model(
                model_uid=_replica_model_uid,
                model_name=model_name,
                model_size_in_billions=model_size_in_billions,
                model_format=model_format,
                quantization=quantization,
                model_type=model_type,
                n_gpu=n_gpu,
                request_limits=request_limits,
                **kwargs,
            )
            self._replica_model_uid_to_worker[_replica_model_uid] = worker_ref

        if not is_valid_model_uid(model_uid):
            raise ValueError(
                "The model UID is invalid. Please specify the model UID by a-z or A-Z, 0 < length <= 100."
            )

        if request_limits is not None and request_limits < 0:
            raise ValueError(
                "The `request_limits` parameter must be greater or equal than 0."
            )

        if model_uid in self._model_uid_to_replica_info:
            raise ValueError(f"Model is already in the model list, uid: {model_uid}")
        # Set replica info first for exception handler to terminate model.
        self._model_uid_to_replica_info[model_uid] = ReplicaInfo(
            replica=replica, scheduler=itertools.cycle(range(replica))
        )
        try:
            for rep_model_uid in iter_replica_model_uid(model_uid, replica):
                await _launch_one_model(rep_model_uid)
        except Exception:
            # terminate_model will remove the replica info.
            await self.terminate_model(model_uid, suppress_exception=True)
            raise
        return model_uid

    async def _check_dead_nodes(self):
        while True:
            dead_nodes = []
            for address, status in self._worker_status.items():
                if time.time() - status.update_time > DEFAULT_NODE_TIMEOUT:
                    dead_models = []
                    for model_uid in self._replica_model_uid_to_worker:
                        if (
                            self._replica_model_uid_to_worker[model_uid].address
                            == address
                        ):
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
    async def terminate_model(self, model_uid: str, suppress_exception=False):
        async def _terminate_one_model(_replica_model_uid):
            worker_ref = self._replica_model_uid_to_worker.get(_replica_model_uid, None)

            if worker_ref is None:
                raise ValueError(
                    f"Model not found in the model list, uid: {_replica_model_uid}"
                )
            await worker_ref.terminate_model(model_uid=_replica_model_uid)
            del self._replica_model_uid_to_worker[_replica_model_uid]

        replica_info = self._model_uid_to_replica_info.get(model_uid, None)
        if replica_info is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        for rep_model_uid in iter_replica_model_uid(model_uid, replica_info.replica):
            try:
                await _terminate_one_model(rep_model_uid)
            except Exception:
                if not suppress_exception:
                    raise
        self._model_uid_to_replica_info.pop(model_uid, None)

    @log_async(logger=logger)
    async def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        replica_info = self._model_uid_to_replica_info.get(model_uid, None)
        if replica_info is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        replica_model_uid = build_replica_model_uid(
            model_uid, replica_info.replica, next(replica_info.scheduler)
        )

        worker_ref = self._replica_model_uid_to_worker.get(replica_model_uid, None)
        if worker_ref is None:
            raise ValueError(
                f"Model not found in the model list, uid: {replica_model_uid}"
            )
        return await worker_ref.get_model(model_uid=replica_model_uid)

    @log_async(logger=logger)
    async def describe_model(self, model_uid: str) -> Dict[str, Any]:
        replica_info = self._model_uid_to_replica_info.get(model_uid, None)
        if replica_info is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")
        # Use rep id 0 to instead of next(replica_info.scheduler) to avoid
        # consuming the generator.
        replica_model_uid = build_replica_model_uid(model_uid, replica_info.replica, 0)
        worker_ref = self._replica_model_uid_to_worker.get(replica_model_uid, None)
        if worker_ref is None:
            raise ValueError(
                f"Model not found in the model list, uid: {replica_model_uid}"
            )
        info = await worker_ref.describe_model(model_uid=replica_model_uid)
        info["replica"] = replica_info.replica
        return info

    @log_async(logger=logger)
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        ret = {}

        workers = list(self._worker_address_to_worker.values())
        for worker in workers:
            ret.update(await worker.list_models())
        return {parse_replica_model_uid(k)[0]: v for k, v in ret.items()}

    def is_local_deployment(self) -> bool:
        # TODO: temporary.
        return (
            len(self._worker_address_to_worker) == 1
            and list(self._worker_address_to_worker)[0] == self.address
        )

    @log_async(logger=logger)
    async def add_worker(self, worker_address: str):
        from .worker import WorkerActor

        assert (
            worker_address not in self._worker_address_to_worker
        ), f"Worker {worker_address} exists"

        worker_ref = await xo.actor_ref(address=worker_address, uid=WorkerActor.uid())
        self._worker_address_to_worker[worker_address] = worker_ref
        logger.debug("Worker %s has been added successfully", worker_address)

    async def report_worker_status(
        self, worker_address: str, status: Dict[str, ResourceStatus]
    ):
        if worker_address not in self._worker_status:
            logger.debug("Worker %s resources: %s", worker_address, status)
        self._worker_status[worker_address] = WorkerStatus(
            update_time=time.time(), status=status
        )
