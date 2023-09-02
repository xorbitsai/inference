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
from logging import getLogger
from typing import Any, Dict, List, Optional, Set

import xoscar as xo

from ..core import ModelActor
from ..model.core import ModelDescription, create_model_instance
from .resource import gather_node_info
from .utils import log_async, log_sync

logger = getLogger(__name__)


DEFAULT_NODE_HEARTBEAT_INTERVAL = 1


class WorkerActor(xo.Actor):
    def __init__(self, supervisor_address: str, subpool_addresses: List[str]):
        super().__init__()
        self._supervisor_address = supervisor_address
        self._supervisor_ref = None
        self._model_uid_to_model: Dict[str, xo.ActorRefType["ModelActor"]] = {}
        self._model_uid_to_model_spec: Dict[str, ModelDescription] = {}
        self._subpool_address_to_model_uids: Dict[str, Set[str]] = dict(
            [(subpool_address, set()) for subpool_address in subpool_addresses]
        )
        logger.debug(f"Worker actor initialized with subpools: {subpool_addresses}")

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
        **kwargs,
    ) -> xo.ActorRefType["ModelActor"]:
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

        subpool_address = self._choose_subpool()
        model_ref = await xo.create_actor(
            ModelActor, address=subpool_address, uid=model_uid, model=model
        )
        await model_ref.load()
        self._model_uid_to_model[model_uid] = model_ref
        self._model_uid_to_model_spec[model_uid] = model_description
        self._subpool_address_to_model_uids[subpool_address].add(model_uid)
        return model_ref

    @log_async(logger=logger)
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

    @log_sync(logger=logger)
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        ret = {}
        for k, v in self._model_uid_to_model_spec.items():
            ret[k] = v.to_description()
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

        return self._model_uid_to_model_spec[model_uid].to_description()

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
