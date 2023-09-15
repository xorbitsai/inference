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
import os
import platform
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import xoscar as xo
from xorbits._mars.resource import cuda_count
from xoscar import MainActorPoolType

from ..core import ModelActor
from ..model.core import ModelDescription, create_model_instance
from .resource import gather_node_info
from .utils import log_async, log_sync

logger = getLogger(__name__)


DEFAULT_NODE_HEARTBEAT_INTERVAL = 1


class WorkerActor(xo.Actor):
    def __init__(
        self,
        supervisor_address: str,
        main_pool: MainActorPoolType,
        cuda_devices: List[int],
    ):
        super().__init__()
        self._total_cuda_devices = cuda_devices
        self._supervisor_address = supervisor_address
        self._supervisor_ref = None
        self._model_uid_to_model: Dict[str, xo.ActorRefType["ModelActor"]] = {}
        self._model_uid_to_model_spec: Dict[str, ModelDescription] = {}

        self._gpu_to_model_uid: Dict[int, str] = {}
        self._model_uid_to_addr: Dict[str, str] = {}
        self._main_pool = main_pool
        logger.debug(
            f"Worker actor initialized with main pool: {self._main_pool.external_address}"
        )

    @classmethod
    def uid(cls) -> str:
        return "worker"

    async def __post_create__(self):
        from .supervisor import SupervisorActor

        self._supervisor_ref: xo.ActorRefType["SupervisorActor"] = await xo.actor_ref(
            address=self._supervisor_address, uid=SupervisorActor.uid()
        )
        await self._supervisor_ref.add_worker(self.address)
        self._upload_task = asyncio.create_task(self._periodical_report_status())

    async def __pre_destroy__(self):
        self._upload_task.cancel()

    @log_sync(logger=logger)
    def get_model_count(self) -> int:
        return len(self._model_uid_to_model)

    def allocate_devices(self, n_gpu: int) -> List[int]:
        """
        Allocate GPUs to the model based on the form-filling method to achieve a balanced GPU load as much as possible.
        """
        if n_gpu > len(self._total_cuda_devices) - len(self._gpu_to_model_uid):
            raise RuntimeError("No available slot found for the model")
        devices: List[int] = [
            dev for dev in self._total_cuda_devices if dev not in self._gpu_to_model_uid
        ][:n_gpu]
        return sorted(devices)

    async def _create_subpool(
        self,
        model_uid: str,
        n_gpu: Optional[Union[int, str]] = "auto",
    ) -> Tuple[str, List[str]]:
        env = {}
        devices = []
        if isinstance(n_gpu, int) or (n_gpu == "auto" and cuda_count() > 0):
            # Currently, n_gpu=auto means using 1 GPU
            gpu_cnt = n_gpu if isinstance(n_gpu, int) else 1
            devices = self.allocate_devices(gpu_cnt)
            env["CUDA_VISIBLE_DEVICES"] = ",".join([str(dev) for dev in devices])
            logger.debug(f"GPU selected: {devices} for model {model_uid}")
        if n_gpu is None:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            logger.debug(f"GPU disabled for model {model_uid}")

        sub_pool_address = await self._main_pool.append_sub_pool(
            env=env, start_method="forkserver" if os.name != "nt" else "spawn"
        )
        return sub_pool_address, [str(dev) for dev in devices]

    def _check_model_is_valid(self, model_name):
        # baichuan-base and baichuan-chat depend on `cpm_kernels` module,
        # but `cpm_kernels` cannot run on Darwin system.
        if platform.system() == "Darwin":
            # TODO: there's no baichuan-base.
            if model_name in ["baichuan-base", "baichuan-chat"]:
                raise ValueError(f"{model_name} model can't run on Darwin system.")

    @log_sync(logger=logger)
    async def register_model(self, model_type: str, model: str, persist: bool):
        # TODO: centralized model registrations
        if model_type == "LLM":
            from ..model.llm import LLMFamilyV1, register_llm

            llm_family = LLMFamilyV1.parse_raw(model)
            register_llm(llm_family, persist)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_sync(logger=logger)
    async def unregister_model(self, model_type: str, model_name: str):
        # TODO: centralized model registrations
        if model_type == "LLM":
            from ..model.llm import unregister_llm

            unregister_llm(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def launch_builtin_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[int],
        model_format: Optional[str],
        quantization: Optional[str],
        model_type: str = "LLM",
        n_gpu: Optional[Union[int, str]] = "auto",
        **kwargs,
    ) -> xo.ActorRefType["ModelActor"]:
        if n_gpu is not None:
            if isinstance(n_gpu, int) and (n_gpu <= 0 or n_gpu > cuda_count()):
                raise ValueError(
                    f"The parameter `n_gpu` must be greater than 0 and "
                    f"not greater than the number of GPUs: {cuda_count()} on the machine."
                )
            if isinstance(n_gpu, str) and n_gpu != "auto":
                raise ValueError("Currently `n_gpu` only supports `auto`.")

        assert model_uid not in self._model_uid_to_model
        self._check_model_is_valid(model_name)
        assert self._supervisor_ref is not None
        is_local_deployment = await self._supervisor_ref.is_local_deployment()

        model, model_description = create_model_instance(
            model_uid,
            model_type,
            model_name,
            model_format,
            model_size_in_billions,
            quantization,
            is_local_deployment,
            **kwargs,
        )

        subpool_address, devices = await self._create_subpool(model_uid, n_gpu=n_gpu)
        model_ref = await xo.create_actor(
            ModelActor, address=subpool_address, uid=model_uid, model=model
        )
        await model_ref.load()
        self._model_uid_to_model[model_uid] = model_ref
        self._model_uid_to_model_spec[model_uid] = model_description
        for dev in devices:
            self._gpu_to_model_uid[int(dev)] = model_uid
        self._model_uid_to_addr[model_uid] = subpool_address
        return model_ref

    @log_async(logger=logger)
    async def terminate_model(self, model_uid: str):
        if model_uid not in self._model_uid_to_model:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        model_ref = self._model_uid_to_model[model_uid]

        await xo.destroy_actor(model_ref)
        del self._model_uid_to_model[model_uid]
        del self._model_uid_to_model_spec[model_uid]

        devs = [dev for dev, uid in self._gpu_to_model_uid.items() if uid == model_uid]
        for dev in devs:
            del self._gpu_to_model_uid[dev]

        sub_pool_addr = self._model_uid_to_addr[model_uid]
        await self._main_pool.remove_sub_pool(sub_pool_addr)
        del self._model_uid_to_addr[model_uid]

    @log_sync(logger=logger)
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        ret = {}
        for k, v in self._model_uid_to_model_spec.items():
            ret[k] = v.to_dict()
        return ret

    @log_sync(logger=logger)
    def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        if model_uid not in self._model_uid_to_model:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        return self._model_uid_to_model[model_uid]

    @log_sync(logger=logger)
    def describe_model(self, model_uid: str) -> Dict[str, Any]:
        if model_uid not in self._model_uid_to_model:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        return self._model_uid_to_model_spec[model_uid].to_dict()

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
                await asyncio.sleep(DEFAULT_NODE_HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:  # pragma: no cover
                break
