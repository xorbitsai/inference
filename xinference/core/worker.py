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
import signal
from collections import defaultdict
from logging import getLogger
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import xoscar as xo
from xoscar import MainActorPoolType

from ..constants import XINFERENCE_CACHE_DIR
from ..core import ModelActor
from ..core.status_guard import LaunchStatus
from ..model.core import ModelDescription, create_model_instance
from ..utils import cuda_count
from .resource import gather_node_info
from .utils import log_async, log_sync, parse_replica_model_uid, purge_dir

logger = getLogger(__name__)


DEFAULT_NODE_HEARTBEAT_INTERVAL = 5


class WorkerActor(xo.StatelessActor):
    def __init__(
        self,
        supervisor_address: str,
        main_pool: MainActorPoolType,
        cuda_devices: List[int],
    ):
        super().__init__()
        # static attrs.
        self._total_cuda_devices = cuda_devices
        self._supervisor_address = supervisor_address
        self._supervisor_ref = None
        self._main_pool = main_pool
        self._main_pool.recover_sub_pool = self.recover_sub_pool

        # internal states.
        self._model_uid_to_model: Dict[str, xo.ActorRefType["ModelActor"]] = {}
        self._model_uid_to_model_spec: Dict[str, ModelDescription] = {}
        self._gpu_to_model_uid: Dict[int, str] = {}
        self._gpu_to_embedding_model_uids: Dict[int, Set[str]] = defaultdict(set)
        self._model_uid_to_addr: Dict[str, str] = {}
        self._model_uid_to_launch_args: Dict[str, Dict] = {}

        self._lock = asyncio.Lock()

    async def recover_sub_pool(self, address):
        logger.warning("Process %s is down, create model.", address)
        for model_uid, addr in self._model_uid_to_addr.items():
            if addr == address:
                launch_args = self._model_uid_to_launch_args.get(model_uid)
                try:
                    await self.terminate_model(model_uid)
                except Exception:
                    pass
                await self.launch_builtin_model(**launch_args)
                break

    @classmethod
    def uid(cls) -> str:
        return "worker"

    async def __post_create__(self):
        from .status_guard import StatusGuardActor
        from .supervisor import SupervisorActor

        self._status_guard_ref: xo.ActorRefType[
            "StatusGuardActor"
        ] = await xo.actor_ref(
            address=self._supervisor_address, uid=StatusGuardActor.uid()
        )
        self._supervisor_ref: xo.ActorRefType["SupervisorActor"] = await xo.actor_ref(
            address=self._supervisor_address, uid=SupervisorActor.uid()
        )
        await self._supervisor_ref.add_worker(self.address)
        self._upload_task = asyncio.create_task(self._periodical_report_status())
        logger.info(f"Xinference worker {self.address} started")
        logger.info("Purge cache directory: %s", XINFERENCE_CACHE_DIR)
        purge_dir(XINFERENCE_CACHE_DIR)

        from ..model.embedding import (
            CustomEmbeddingModelSpec,
            register_embedding,
            unregister_embedding,
        )
        from ..model.llm import register_llm, unregister_llm
        from ..model.llm.llm_family import CustomLLMFamilyV1
        from ..model.rerank.custom import (
            CustomRerankModelSpec,
            register_rerank,
            unregister_rerank,
        )

        self._custom_register_type_to_cls: Dict[str, Tuple] = {
            "LLM": (CustomLLMFamilyV1, register_llm, unregister_llm),
            "embedding": (
                CustomEmbeddingModelSpec,
                register_embedding,
                unregister_embedding,
            ),
            "rerank": (CustomRerankModelSpec, register_rerank, unregister_rerank),
        }

        # Windows does not have signal handler
        if os.name != "nt":

            async def signal_handler():
                await self._supervisor_ref.remove_worker(self.address)
                os._exit(0)

            loop = asyncio.get_running_loop()
            loop.add_signal_handler(
                signal.SIGINT, lambda: asyncio.create_task(signal_handler())
            )

    async def __pre_destroy__(self):
        self._upload_task.cancel()

    @staticmethod
    def get_devices_count():
        from ..utils import cuda_count

        return cuda_count()

    @log_sync(logger=logger)
    def get_model_count(self) -> int:
        return len(self._model_uid_to_model)

    async def is_model_vllm_backend(self, model_uid: str) -> bool:
        assert self._supervisor_ref is not None
        _model_uid, _, _ = parse_replica_model_uid(model_uid)
        model_ref = await self._supervisor_ref.get_model(_model_uid)
        return await model_ref.is_vllm_backend()

    async def allocate_devices_for_embedding(self, model_uid: str) -> int:
        """
        we assume that embedding model only takes 1 GPU slot.
        """
        candidates = []
        for _dev in self._total_cuda_devices:
            if _dev not in self._gpu_to_model_uid:
                candidates.append(_dev)
            else:
                existing_model_uid = self._gpu_to_model_uid[_dev]
                is_vllm_model = await self.is_model_vllm_backend(existing_model_uid)
                if not is_vllm_model:
                    candidates.append(_dev)

        if len(candidates) == 0:
            raise RuntimeError(
                "No available slot found for the embedding model. "
                "We recommend to launch the embedding model first, and then launch the LLM models."
            )

        device, min_cnt = -1, -1
        # Pick the device with the fewest existing models among all the candidate devices.
        for _dev in candidates:
            existing_cnt = len(self._gpu_to_embedding_model_uids[_dev])
            if _dev in self._gpu_to_model_uid:
                existing_cnt += 1
            if min_cnt == -1 or existing_cnt < min_cnt:
                device, min_cnt = _dev, existing_cnt

        self._gpu_to_embedding_model_uids[device].add(model_uid)
        return device

    def allocate_devices(self, model_uid: str, n_gpu: int) -> List[int]:
        if n_gpu > len(self._total_cuda_devices) - len(self._gpu_to_model_uid):
            raise RuntimeError("No available slot found for the model")

        devices: List[int] = [
            dev for dev in self._total_cuda_devices if dev not in self._gpu_to_model_uid
        ][:n_gpu]
        for dev in devices:
            self._gpu_to_model_uid[int(dev)] = model_uid

        return sorted(devices)

    def release_devices(self, model_uid: str):
        devices = [
            dev
            for dev in self._gpu_to_model_uid
            if self._gpu_to_model_uid[dev] == model_uid
        ]
        for dev in devices:
            del self._gpu_to_model_uid[dev]

        # check embedding
        for dev in self._gpu_to_embedding_model_uids:
            if model_uid in self._gpu_to_embedding_model_uids[dev]:
                self._gpu_to_embedding_model_uids[dev].remove(model_uid)

    async def _create_subpool(
        self,
        model_uid: str,
        model_type: Optional[str] = None,
        n_gpu: Optional[Union[int, str]] = "auto",
    ) -> Tuple[str, List[str]]:
        env = {}
        devices = []
        if isinstance(n_gpu, int) or (n_gpu == "auto" and cuda_count() > 0):
            # Currently, n_gpu=auto means using 1 GPU
            gpu_cnt = n_gpu if isinstance(n_gpu, int) else 1
            devices = (
                [await self.allocate_devices_for_embedding(model_uid)]
                if model_type in ["embedding", "rerank"]
                else self.allocate_devices(model_uid=model_uid, n_gpu=gpu_cnt)
            )
            env["CUDA_VISIBLE_DEVICES"] = ",".join([str(dev) for dev in devices])
            logger.debug(f"GPU selected: {devices} for model {model_uid}")
        if n_gpu is None:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            logger.debug(f"GPU disabled for model {model_uid}")

        if os.name != "nt" and platform.system() != "Darwin":
            # Linux
            start_method = "forkserver"
        else:
            # Windows and macOS
            start_method = "spawn"
        subpool_address = await self._main_pool.append_sub_pool(
            env=env, start_method=start_method
        )
        return subpool_address, [str(dev) for dev in devices]

    def _check_model_is_valid(self, model_name: str, model_format: Optional[str]):
        # baichuan-base and baichuan-chat depend on `cpm_kernels` module,
        # but `cpm_kernels` cannot run on Darwin system.
        if platform.system() == "Darwin" and model_format == "pytorch":
            if "baichuan" in model_name:
                raise ValueError(f"{model_name} model can't run on Darwin system.")

    @log_sync(logger=logger)
    def register_model(self, model_type: str, model: str, persist: bool):
        # TODO: centralized model registrations
        if model_type in self._custom_register_type_to_cls:
            (
                model_spec_cls,
                register_fn,
                unregister_fn,
            ) = self._custom_register_type_to_cls[model_type]
            model_spec = model_spec_cls.parse_raw(model)
            try:
                register_fn(model_spec, persist)
            except Exception as e:
                unregister_fn(model_spec.model_name, raise_error=False)
                raise e
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_sync(logger=logger)
    def unregister_model(self, model_type: str, model_name: str):
        # TODO: centralized model registrations
        if model_type in self._custom_register_type_to_cls:
            _, _, unregister_fn = self._custom_register_type_to_cls[model_type]
            unregister_fn(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def launch_speculative_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[int],
        quantization: Optional[str],
        draft_model_name: str,
        draft_model_size_in_billions: Optional[int],
        draft_quantization: Optional[str],
        n_gpu: Optional[Union[int, str]] = "auto",
    ):
        if n_gpu is not None:
            if isinstance(n_gpu, int) and (n_gpu <= 0 or n_gpu > cuda_count()):
                raise ValueError(
                    f"The parameter `n_gpu` must be greater than 0 and "
                    f"not greater than the number of GPUs: {cuda_count()} on the machine."
                )
            if isinstance(n_gpu, str) and n_gpu != "auto":
                raise ValueError("Currently `n_gpu` only supports `auto`.")

        from ..model.llm.core import create_speculative_llm_model_instance

        subpool_address, devices = await self._create_subpool(model_uid, n_gpu=n_gpu)

        model, model_description = await asyncio.to_thread(
            create_speculative_llm_model_instance,
            subpool_addr=subpool_address,
            devices=devices,
            model_uid=model_uid,
            model_name=model_name,
            model_size_in_billions=model_size_in_billions,
            quantization=quantization,
            draft_model_name=draft_model_name,
            draft_model_size_in_billions=draft_model_size_in_billions,
            draft_quantization=draft_quantization,
            is_local_deployment=True,
        )

        try:
            model_ref = await xo.create_actor(
                ModelActor, address=subpool_address, uid=model_uid, model=model
            )
            await model_ref.load()
        except:
            logger.error(f"Failed to load model {model_uid}", exc_info=True)
            self.release_devices(model_uid=model_uid)
            await self._main_pool.remove_sub_pool(subpool_address)
            raise

        self._model_uid_to_model[model_uid] = model_ref
        self._model_uid_to_model_spec[model_uid] = model_description
        for dev in devices:
            self._gpu_to_model_uid[int(dev)] = model_uid
        self._model_uid_to_addr[model_uid] = subpool_address

    async def _get_model_ability(self, model: Any, model_type: str) -> List[str]:
        from ..model.llm.core import LLM

        if model_type == "embedding":
            return ["embed"]
        elif model_type == "rerank":
            return ["rerank"]
        elif model_type == "image":
            return ["text_to_image"]
        elif model_type == "multimodal":
            return ["multimodal"]
        else:
            assert model_type == "LLM"
            assert isinstance(model, LLM)
            return model.model_family.model_ability

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
        request_limits: Optional[int] = None,
        **kwargs,
    ):
        launch_args = locals()
        launch_args.pop("self")
        launch_args.pop("kwargs")
        launch_args.update(kwargs)
        if n_gpu is not None:
            if isinstance(n_gpu, int) and (n_gpu <= 0 or n_gpu > cuda_count()):
                raise ValueError(
                    f"The parameter `n_gpu` must be greater than 0 and "
                    f"not greater than the number of GPUs: {cuda_count()} on the machine."
                )
            if isinstance(n_gpu, str) and n_gpu != "auto":
                raise ValueError("Currently `n_gpu` only supports `auto`.")

        assert model_uid not in self._model_uid_to_model
        self._check_model_is_valid(model_name, model_format)
        assert self._supervisor_ref is not None
        is_local_deployment = await self._supervisor_ref.is_local_deployment()

        subpool_address, devices = await self._create_subpool(
            model_uid, model_type, n_gpu=n_gpu
        )

        try:
            origin_uid, _, _ = parse_replica_model_uid(model_uid)
            await self._status_guard_ref.update_instance_info(
                origin_uid, {"status": LaunchStatus.DOWNLOADING.name}
            )
            model, model_description = await asyncio.to_thread(
                create_model_instance,
                subpool_address,
                devices,
                model_uid,
                model_type,
                model_name,
                model_format,
                model_size_in_billions,
                quantization,
                is_local_deployment,
                **kwargs,
            )
            abilities = await self._get_model_ability(model, model_type)
            await self._status_guard_ref.update_instance_info(
                origin_uid,
                {"model_ablity": abilities, "status": LaunchStatus.DOWNLOADED.name},
            )
            model_ref = await xo.create_actor(
                ModelActor,
                address=subpool_address,
                uid=model_uid,
                model=model,
                request_limits=request_limits,
            )
            await model_ref.load()
            await self._status_guard_ref.update_instance_info(
                origin_uid,
                {"model_ablity": abilities, "status": LaunchStatus.READY.name},
            )
        except:
            logger.error(f"Failed to load model {model_uid}", exc_info=True)
            self.release_devices(model_uid=model_uid)
            await self._main_pool.remove_sub_pool(subpool_address)
            raise

        self._model_uid_to_model[model_uid] = model_ref
        self._model_uid_to_model_spec[model_uid] = model_description
        self._model_uid_to_addr[model_uid] = subpool_address
        self._model_uid_to_launch_args[model_uid] = launch_args

    @log_async(logger=logger)
    async def terminate_model(self, model_uid: str):
        origin_uid, _, _ = parse_replica_model_uid(model_uid)
        model_ref = self._model_uid_to_model.get(model_uid, None)
        if model_ref is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        try:
            await xo.destroy_actor(model_ref)
        except Exception as e:
            logger.debug(
                "Destroy model actor failed, model uid: %s, error: %s", model_uid, e
            )
        try:
            subpool_address = self._model_uid_to_addr[model_uid]
            await self._main_pool.remove_sub_pool(subpool_address)
        finally:
            del self._model_uid_to_model[model_uid]
            del self._model_uid_to_model_spec[model_uid]
            self.release_devices(model_uid)
            del self._model_uid_to_addr[model_uid]
            del self._model_uid_to_launch_args[model_uid]
            await self._status_guard_ref.update_instance_info(
                origin_uid, {"status": LaunchStatus.TERMINATED.name}
            )

    @log_async(logger=logger)
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        ret = {}

        items = list(self._model_uid_to_model_spec.items())
        for k, v in items:
            ret[k] = v.to_dict()
        return ret

    @log_sync(logger=logger)
    def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        model_ref = self._model_uid_to_model.get(model_uid, None)
        if model_ref is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")
        return model_ref

    @log_sync(logger=logger)
    def describe_model(self, model_uid: str) -> Dict[str, Any]:
        model_desc = self._model_uid_to_model_spec.get(model_uid, None)
        if model_desc is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")
        return model_desc.to_dict()

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
