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
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

import xoscar as xo

from ..constants import (
    XINFERENCE_DISABLE_HEALTH_CHECK,
    XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
    XINFERENCE_HEALTH_CHECK_INTERVAL,
    XINFERENCE_HEALTH_CHECK_TIMEOUT,
)
from ..core import ModelActor
from ..core.status_guard import InstanceInfo, LaunchStatus
from .metrics import record_metrics
from .resource import GPUStatus, ResourceStatus
from .utils import (
    build_replica_model_uid,
    gen_random_string,
    is_valid_model_uid,
    iter_replica_model_uid,
    log_async,
    log_sync,
    parse_model_version,
    parse_replica_model_uid,
)

if TYPE_CHECKING:
    from ..model.audio import AudioModelFamilyV1
    from ..model.embedding import EmbeddingModelSpec
    from ..model.image import ImageModelFamilyV1
    from ..model.llm import LLMFamilyV1
    from ..model.rerank import RerankModelSpec
    from .worker import WorkerActor


logger = getLogger(__name__)


ASYNC_LAUNCH_TASKS = {}  # type: ignore


def callback_for_async_launch(model_uid: str):
    ASYNC_LAUNCH_TASKS.pop(model_uid, None)
    logger.debug(f"Model uid: {model_uid} async launch completes.")


@dataclass
class WorkerStatus:
    update_time: float
    failure_remaining_count: int
    status: Dict[str, Union[ResourceStatus, GPUStatus]]


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
        if not XINFERENCE_DISABLE_HEALTH_CHECK:
            # Run _check_dead_nodes() in a dedicated thread.
            from ..isolation import Isolation

            self._isolation = Isolation(asyncio.new_event_loop(), threaded=True)
            self._isolation.start()
            asyncio.run_coroutine_threadsafe(
                self._check_dead_nodes(), loop=self._isolation.loop
            )
        logger.info(f"Xinference supervisor {self.address} started")
        from .cache_tracker import CacheTrackerActor
        from .status_guard import StatusGuardActor

        self._status_guard_ref: xo.ActorRefType[
            "StatusGuardActor"
        ] = await xo.create_actor(
            StatusGuardActor, address=self.address, uid=StatusGuardActor.uid()
        )
        self._cache_tracker_ref: xo.ActorRefType[
            "CacheTrackerActor"
        ] = await xo.create_actor(
            CacheTrackerActor, address=self.address, uid=CacheTrackerActor.uid()
        )

        from .event import EventCollectorActor

        self._event_collector_ref: xo.ActorRefType[
            EventCollectorActor
        ] = await xo.create_actor(
            EventCollectorActor, address=self.address, uid=EventCollectorActor.uid()
        )

        from ..model.embedding import (
            CustomEmbeddingModelSpec,
            generate_embedding_description,
            get_embedding_model_descriptions,
            register_embedding,
            unregister_embedding,
        )
        from ..model.image import get_image_model_descriptions
        from ..model.llm import (
            CustomLLMFamilyV1,
            generate_llm_description,
            get_llm_model_descriptions,
            register_llm,
            unregister_llm,
        )
        from ..model.rerank import (
            CustomRerankModelSpec,
            generate_rerank_description,
            get_rerank_model_descriptions,
            register_rerank,
            unregister_rerank,
        )

        self._custom_register_type_to_cls: Dict[str, Tuple] = {
            "LLM": (
                CustomLLMFamilyV1,
                register_llm,
                unregister_llm,
                generate_llm_description,
            ),
            "embedding": (
                CustomEmbeddingModelSpec,
                register_embedding,
                unregister_embedding,
                generate_embedding_description,
            ),
            "rerank": (
                CustomRerankModelSpec,
                register_rerank,
                unregister_rerank,
                generate_rerank_description,
            ),
        }

        # record model version
        model_version_infos: Dict[str, List[Dict]] = {}
        model_version_infos.update(get_llm_model_descriptions())
        model_version_infos.update(get_embedding_model_descriptions())
        model_version_infos.update(get_rerank_model_descriptions())
        model_version_infos.update(get_image_model_descriptions())
        await self._cache_tracker_ref.record_model_version(
            model_version_infos, self.address
        )

    async def get_cluster_device_info(self) -> List:
        supervisor_device_info = {
            "ip_address": self.address.split(":")[0],
            "gpu_count": 0,
            "gpu_vram_total": 0,
        }
        res = [{"node_type": "Supervisor", **supervisor_device_info}]
        for worker_addr, worker_status in self._worker_status.items():
            vram_total: float = sum(
                [v.mem_total for k, v in worker_status.status.items() if k != "cpu"]  # type: ignore
            )
            total = (
                vram_total if vram_total == 0 else f"{int(vram_total / 1024 / 1024)}MiB"
            )
            res.append(
                {
                    "node_type": "Worker",
                    "ip_address": worker_addr.split(":")[0],
                    "gpu_count": len(worker_status.status) - 1,
                    "gpu_vram_total": total,
                }
            )
        return res

    @staticmethod
    async def get_builtin_prompts() -> Dict[str, Any]:
        from ..model.llm.llm_family import BUILTIN_LLM_PROMPT_STYLE

        data = {}
        for k, v in BUILTIN_LLM_PROMPT_STYLE.items():
            data[k] = v.dict()
        return data

    @staticmethod
    async def get_builtin_families() -> Dict[str, List[str]]:
        from ..model.llm.llm_family import (
            BUILTIN_LLM_MODEL_CHAT_FAMILIES,
            BUILTIN_LLM_MODEL_GENERATE_FAMILIES,
            BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES,
        )

        return {
            "chat": list(BUILTIN_LLM_MODEL_CHAT_FAMILIES),
            "generate": list(BUILTIN_LLM_MODEL_GENERATE_FAMILIES),
            "tool_call": list(BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES),
        }

    async def get_devices_count(self) -> int:
        from ..utils import cuda_count

        if self.is_local_deployment():
            return cuda_count()
        # distributed deployment, choose a worker and return its cuda_count.
        # Assume that each worker has the same count of cards.
        worker_ref = await self._choose_worker()
        return await worker_ref.get_devices_count()

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

    async def _to_llm_reg(
        self, llm_family: "LLMFamilyV1", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.llm import get_cache_status

        instance_cnt = await self.get_instance_count(llm_family.model_name)
        version_cnt = await self.get_model_version_count(llm_family.model_name)

        if self.is_local_deployment():
            specs = []
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            for spec in llm_family.model_specs:
                cache_status = get_cache_status(llm_family, spec)
                specs.append({**spec.dict(), "cache_status": cache_status})
            res = {**llm_family.dict(), "is_builtin": is_builtin, "model_specs": specs}
        else:
            res = {**llm_family.dict(), "is_builtin": is_builtin}
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_embedding_model_reg(
        self, model_spec: "EmbeddingModelSpec", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.embedding import get_cache_status

        instance_cnt = await self.get_instance_count(model_spec.model_name)
        version_cnt = await self.get_model_version_count(model_spec.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_spec)
            res = {
                **model_spec.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_spec.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_rerank_model_reg(
        self, model_spec: "RerankModelSpec", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.rerank import get_cache_status

        instance_cnt = await self.get_instance_count(model_spec.model_name)
        version_cnt = await self.get_model_version_count(model_spec.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_spec)
            res = {
                **model_spec.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_spec.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_image_model_reg(
        self, model_family: "ImageModelFamilyV1", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.image import get_cache_status

        instance_cnt = await self.get_instance_count(model_family.model_name)
        version_cnt = await self.get_model_version_count(model_family.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_family)
            res = {
                **model_family.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_family.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_audio_model_reg(
        self, model_family: "AudioModelFamilyV1", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.audio import get_cache_status

        instance_cnt = await self.get_instance_count(model_family.model_name)
        version_cnt = await self.get_model_version_count(model_family.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_family)
            res = {
                **model_family.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_family.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    @log_async(logger=logger)
    async def list_model_registrations(
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
                    ret.append(await self._to_llm_reg(family, True))
                else:
                    ret.append({"model_name": family.model_name, "is_builtin": True})

            for family in get_user_defined_llm_families():
                if detailed:
                    ret.append(await self._to_llm_reg(family, False))
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
                    ret.append(
                        await self._to_embedding_model_reg(family, is_builtin=True)
                    )
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            for model_spec in get_user_defined_embeddings():
                if detailed:
                    ret.append(
                        await self._to_embedding_model_reg(model_spec, is_builtin=False)
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
                    ret.append(await self._to_image_model_reg(family, is_builtin=True))
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "audio":
            from ..model.audio import BUILTIN_AUDIO_MODELS

            ret = []
            for model_name, family in BUILTIN_AUDIO_MODELS.items():
                if detailed:
                    ret.append(await self._to_audio_model_reg(family, is_builtin=True))
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "rerank":
            from ..model.rerank import BUILTIN_RERANK_MODELS
            from ..model.rerank.custom import get_user_defined_reranks

            ret = []
            for model_name, family in BUILTIN_RERANK_MODELS.items():
                if detailed:
                    ret.append(await self._to_rerank_model_reg(family, is_builtin=True))
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            for model_spec in get_user_defined_reranks():
                if detailed:
                    ret.append(
                        await self._to_rerank_model_reg(model_spec, is_builtin=False)
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

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
        elif model_type == "embedding":
            from ..model.embedding import BUILTIN_EMBEDDING_MODELS
            from ..model.embedding.custom import get_user_defined_embeddings

            for f in (
                list(BUILTIN_EMBEDDING_MODELS.values()) + get_user_defined_embeddings()
            ):
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        elif model_type == "image":
            from ..model.image import BUILTIN_IMAGE_MODELS

            for f in BUILTIN_IMAGE_MODELS.values():
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        elif model_type == "audio":
            from ..model.audio import BUILTIN_AUDIO_MODELS

            for f in BUILTIN_AUDIO_MODELS.values():
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        elif model_type == "rerank":
            from ..model.rerank import BUILTIN_RERANK_MODELS
            from ..model.rerank.custom import get_user_defined_reranks

            for f in list(BUILTIN_RERANK_MODELS.values()) + get_user_defined_reranks():
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def register_model(self, model_type: str, model: str, persist: bool):
        if model_type in self._custom_register_type_to_cls:
            (
                model_spec_cls,
                register_fn,
                unregister_fn,
                generate_fn,
            ) = self._custom_register_type_to_cls[model_type]

            if not self.is_local_deployment():
                workers = list(self._worker_address_to_worker.values())
                for worker in workers:
                    await worker.register_model(model_type, model, persist)

            model_spec = model_spec_cls.parse_raw(model)
            try:
                register_fn(model_spec, persist)
                await self._cache_tracker_ref.record_model_version(
                    generate_fn(model_spec), self.address
                )
            except Exception as e:
                unregister_fn(model_spec.model_name, raise_error=False)
                raise e
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def unregister_model(self, model_type: str, model_name: str):
        if model_type in self._custom_register_type_to_cls:
            _, _, unregister_fn, _ = self._custom_register_type_to_cls[model_type]
            unregister_fn(model_name)
            await self._cache_tracker_ref.unregister_model_version(model_name)

            if not self.is_local_deployment():
                workers = list(self._worker_address_to_worker.values())
                for worker in workers:
                    await worker.unregister_model(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _gen_model_uid(self, model_name: str) -> str:
        if model_name not in self._model_uid_to_replica_info:
            return model_name
        logger.debug(
            f"{model_name} exists in xinference. Generate suffix to {model_name} for model_uid."
        )
        return f"{model_name}-{gen_random_string(8)}"

    async def get_model_versions(self, model_type: str, model_name: str) -> List[Dict]:
        return await self._cache_tracker_ref.get_model_versions(model_name)

    async def get_model_version_count(self, model_name: str) -> int:
        return await self._cache_tracker_ref.get_model_version_count(model_name)

    @log_async(logger=logger)
    async def launch_model_by_version(
        self,
        model_uid: Optional[str],
        model_type: str,
        model_version: str,
        replica: int = 1,
        n_gpu: Optional[Union[int, str]] = "auto",
        wait_ready: bool = True,
    ):
        parse_results = parse_model_version(model_version, model_type)

        if model_type == "image" and len(parse_results) == 2:
            kwargs = {"controlnet": parse_results[1]}
        else:
            kwargs = {}

        return await self.launch_builtin_model(
            model_uid=model_uid,
            model_name=parse_results[0],
            model_size_in_billions=parse_results[1] if model_type == "LLM" else None,
            model_format=parse_results[2] if model_type == "LLM" else None,
            quantization=parse_results[3] if model_type == "LLM" else None,
            model_type=model_type,
            replica=replica,
            n_gpu=n_gpu,
            wait_ready=wait_ready,
            model_version=model_version,
            **kwargs,
        )

    async def launch_speculative_llm(
        self,
        model_uid: Optional[str],
        model_name: str,
        model_size_in_billions: Optional[int],
        quantization: Optional[str],
        draft_model_name: str,
        draft_model_size_in_billions: Optional[int],
        draft_quantization: Optional[str],
        n_gpu: Optional[Union[int, str]] = "auto",
    ) -> str:
        if model_uid is None:
            model_uid = self._gen_model_uid(model_name)
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
        model_uid: Optional[str],
        model_name: str,
        model_size_in_billions: Optional[int],
        model_format: Optional[str],
        quantization: Optional[str],
        model_type: Optional[str],
        replica: int = 1,
        n_gpu: Optional[Union[int, str]] = "auto",
        request_limits: Optional[int] = None,
        wait_ready: bool = True,
        model_version: Optional[str] = None,
        **kwargs,
    ) -> str:
        if model_uid is None:
            model_uid = self._gen_model_uid(model_name)

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

        async def _launch_model():
            try:
                for rep_model_uid in iter_replica_model_uid(model_uid, replica):
                    await _launch_one_model(rep_model_uid)
            except Exception:
                # terminate_model will remove the replica info.
                await self.terminate_model(model_uid, suppress_exception=True)
                await self._status_guard_ref.update_instance_info(
                    model_uid, {"status": LaunchStatus.ERROR.name}
                )
                raise

        if not is_valid_model_uid(model_uid):
            raise ValueError(
                "The model UID is invalid. Please specify the model UID by 0 < length <= 100."
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
        instance_info = InstanceInfo(
            model_name=model_name,
            model_uid=model_uid,
            model_version=model_version,
            model_ability=[],
            replica=replica,
            status=LaunchStatus.CREATING.name,
            instance_created_ts=int(time.time()),
        )
        await self._status_guard_ref.set_instance_info(model_uid, instance_info)
        if wait_ready:
            await _launch_model()
        else:
            task = asyncio.create_task(_launch_model())
            ASYNC_LAUNCH_TASKS[model_uid] = task
            task.add_done_callback(lambda _: callback_for_async_launch(model_uid))
        return model_uid

    async def get_instance_info(
        self, model_name: Optional[str], model_uid: Optional[str]
    ) -> List[Dict]:
        infos = await self._status_guard_ref.get_instance_info(
            model_name=model_name, model_uid=model_uid
        )
        return [info.dict() for info in sorted(infos, key=lambda info: info.model_uid)]

    async def get_instance_count(self, model_name: str) -> int:
        return await self._status_guard_ref.get_instance_count(model_name)

    async def _check_dead_nodes(self):
        while True:
            try:
                dead_nodes = []
                for address, status in self._worker_status.items():
                    if (
                        time.time() - status.update_time
                        > XINFERENCE_HEALTH_CHECK_TIMEOUT
                    ):
                        status.failure_remaining_count -= 1
                    else:
                        status.failure_remaining_count = (
                            XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD
                        )

                    if status.failure_remaining_count <= 0:
                        dead_models = []
                        for model_uid in self._replica_model_uid_to_worker:
                            if (
                                self._replica_model_uid_to_worker[model_uid].address
                                == address
                            ):
                                dead_models.append(model_uid)
                        logger.error(
                            "Worker dead. address: %s, influenced models: %s",
                            address,
                            dead_models,
                        )
                        dead_nodes.append(address)
                    elif (
                        status.failure_remaining_count
                        != XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD
                    ):
                        logger.error(
                            "Worker timeout. address: %s, check count remaining %s...",
                            address,
                            status.failure_remaining_count,
                        )

                for address in dead_nodes:
                    self._worker_status.pop(address, None)
                    self._worker_address_to_worker.pop(address, None)
            finally:
                await asyncio.sleep(XINFERENCE_HEALTH_CHECK_INTERVAL)

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

    @log_async(logger=logger)
    async def remove_worker(self, worker_address: str):
        if worker_address in self._worker_address_to_worker:
            del self._worker_address_to_worker[worker_address]
            logger.debug("Worker %s has been removed successfully", worker_address)
        else:
            logger.warning(
                f"Worker {worker_address} cannot be removed since it is not registered to supervisor."
            )

    async def report_worker_status(
        self, worker_address: str, status: Dict[str, Union[ResourceStatus, GPUStatus]]
    ):
        if worker_address not in self._worker_status:
            logger.debug("Worker %s resources: %s", worker_address, status)
            self._worker_status[worker_address] = WorkerStatus(
                update_time=time.time(),
                failure_remaining_count=XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
                status=status,
            )
        else:
            worker_status = self._worker_status[worker_address]
            worker_status.update_time = time.time()
            worker_status.status = status

    @staticmethod
    def record_metrics(name, op, kwargs):
        record_metrics(name, op, kwargs)
