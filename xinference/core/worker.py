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
import queue
import shutil
import signal
import threading
import time
from collections import defaultdict
from logging import getLogger
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import xoscar as xo
from async_timeout import timeout
from xoscar import MainActorPoolType

from ..constants import (
    XINFERENCE_CACHE_DIR,
    XINFERENCE_DISABLE_HEALTH_CHECK,
    XINFERENCE_DISABLE_METRICS,
    XINFERENCE_HEALTH_CHECK_INTERVAL,
)
from ..core.model import ModelActor
from ..core.status_guard import LaunchStatus
from ..device_utils import get_available_device_env_name, gpu_count
from ..model.core import ModelDescription, create_model_instance
from ..types import PeftModelConfig
from .event import Event, EventCollectorActor, EventType
from .metrics import launch_metrics_export_server, record_metrics
from .resource import gather_node_info
from .utils import log_async, log_sync, parse_replica_model_uid, purge_dir

logger = getLogger(__name__)


MODEL_ACTOR_AUTO_RECOVER_LIMIT: Optional[int]
_MODEL_ACTOR_AUTO_RECOVER_LIMIT = os.getenv("XINFERENCE_MODEL_ACTOR_AUTO_RECOVER_LIMIT")
if _MODEL_ACTOR_AUTO_RECOVER_LIMIT is not None:
    MODEL_ACTOR_AUTO_RECOVER_LIMIT = int(_MODEL_ACTOR_AUTO_RECOVER_LIMIT)
else:
    MODEL_ACTOR_AUTO_RECOVER_LIMIT = None


class WorkerActor(xo.StatelessActor):
    def __init__(
        self,
        supervisor_address: str,
        main_pool: MainActorPoolType,
        gpu_devices: List[int],
        metrics_exporter_host: Optional[str] = None,
        metrics_exporter_port: Optional[int] = None,
    ):
        super().__init__()
        # static attrs.
        self._total_gpu_devices = gpu_devices
        self._supervisor_address = supervisor_address
        self._supervisor_ref = None
        self._main_pool = main_pool
        self._main_pool.recover_sub_pool = self.recover_sub_pool

        # internal states.
        self._model_uid_to_model: Dict[str, xo.ActorRefType["ModelActor"]] = {}
        self._model_uid_to_model_spec: Dict[str, ModelDescription] = {}
        self._gpu_to_model_uid: Dict[int, str] = {}
        self._gpu_to_embedding_model_uids: Dict[int, Set[str]] = defaultdict(set)
        # Dict structure: gpu_index: {(replica_model_uid, model_type)}
        self._user_specified_gpu_to_model_uids: Dict[
            int, Set[Tuple[str, str]]
        ] = defaultdict(set)
        self._model_uid_to_addr: Dict[str, str] = {}
        self._model_uid_to_recover_count: Dict[str, Optional[int]] = {}
        self._model_uid_to_launch_args: Dict[str, Dict] = {}

        if XINFERENCE_DISABLE_METRICS:
            logger.info(
                "Worker metrics is disabled due to the environment XINFERENCE_DISABLE_METRICS=1"
            )
        elif metrics_exporter_host is not None or metrics_exporter_port is not None:
            # metrics export server.
            logger.info(
                f"Starting metrics export server at {metrics_exporter_host}:{metrics_exporter_port}"
            )
            q: queue.Queue = queue.Queue()
            self._metrics_thread = threading.Thread(
                name="Metrics Export Server",
                target=launch_metrics_export_server,
                args=(q, metrics_exporter_host, metrics_exporter_port),
                daemon=True,
            )
            self._metrics_thread.start()
            logger.info("Checking metrics export server...")
            while self._metrics_thread.is_alive():
                try:
                    host, port = q.get(block=False)[:2]
                    logger.info(f"Metrics server is started at: http://{host}:{port}")
                    break
                except queue.Empty:
                    pass
            else:
                raise Exception("Metrics server thread exit.")

        self._lock = asyncio.Lock()

    async def recover_sub_pool(self, address):
        logger.warning("Process %s is down.", address)
        # Xoscar does not remove the address from sub_processes.
        try:
            await self._main_pool.remove_sub_pool(address)
        except Exception:
            pass
        for model_uid, addr in self._model_uid_to_addr.items():
            if addr == address:
                launch_args = self._model_uid_to_launch_args.get(model_uid)
                if launch_args is None:
                    logger.warning(
                        "Not recreate model because the it is down during launch."
                    )
                else:
                    recover_count = self._model_uid_to_recover_count.get(model_uid)
                    try:
                        await self.terminate_model(model_uid)
                    except Exception:
                        pass
                    if recover_count is not None:
                        if recover_count > 0:
                            logger.warning(
                                "Recreating model actor %s, remain %s times ...",
                                model_uid,
                                recover_count - 1,
                            )
                            event_model_uid, _, __ = parse_replica_model_uid(model_uid)
                            try:
                                await self._event_collector_ref.report_event(
                                    event_model_uid,
                                    Event(
                                        event_type=EventType.WARNING,
                                        event_ts=int(time.time()),
                                        event_content="Recreate model",
                                    ),
                                )
                            except Exception as e:
                                # Report callback error can be log and ignore, should not interrupt the Process
                                logger.error("report_event error: %s" % (e))

                            self._model_uid_to_recover_count[model_uid] = (
                                recover_count - 1
                            )
                            await self.launch_builtin_model(**launch_args)
                        else:
                            logger.warning("Stop recreating model actor.")
                    else:
                        logger.warning("Recreating model actor %s ...", model_uid)
                        await self.launch_builtin_model(**launch_args)
                break

    @classmethod
    def uid(cls) -> str:
        return "worker"

    async def __post_create__(self):
        from ..isolation import Isolation
        from .cache_tracker import CacheTrackerActor
        from .status_guard import StatusGuardActor
        from .supervisor import SupervisorActor

        self._status_guard_ref: xo.ActorRefType[  # type: ignore
            "StatusGuardActor"
        ] = await xo.actor_ref(
            address=self._supervisor_address, uid=StatusGuardActor.uid()
        )
        self._event_collector_ref: xo.ActorRefType[  # type: ignore
            EventCollectorActor
        ] = await xo.actor_ref(
            address=self._supervisor_address, uid=EventCollectorActor.uid()
        )
        self._cache_tracker_ref: xo.ActorRefType[  # type: ignore
            "CacheTrackerActor"
        ] = await xo.actor_ref(
            address=self._supervisor_address, uid=CacheTrackerActor.uid()
        )
        self._supervisor_ref: xo.ActorRefType["SupervisorActor"] = await xo.actor_ref(  # type: ignore
            address=self._supervisor_address, uid=SupervisorActor.uid()
        )
        await self._supervisor_ref.add_worker(self.address)
        if not XINFERENCE_DISABLE_HEALTH_CHECK:
            # Run _periodical_report_status() in a dedicated thread.
            self._isolation = Isolation(asyncio.new_event_loop(), threaded=True)
            self._isolation.start()
            asyncio.run_coroutine_threadsafe(
                self._periodical_report_status(), loop=self._isolation.loop
            )
        logger.info(f"Xinference worker {self.address} started")
        logger.info("Purge cache directory: %s", XINFERENCE_CACHE_DIR)
        purge_dir(XINFERENCE_CACHE_DIR)

        from ..model.audio import (
            CustomAudioModelFamilyV1,
            get_audio_model_descriptions,
            register_audio,
            unregister_audio,
        )
        from ..model.embedding import (
            CustomEmbeddingModelSpec,
            get_embedding_model_descriptions,
            register_embedding,
            unregister_embedding,
        )
        from ..model.image import (
            CustomImageModelFamilyV1,
            get_image_model_descriptions,
            register_image,
            unregister_image,
        )
        from ..model.llm import (
            CustomLLMFamilyV1,
            get_llm_model_descriptions,
            register_llm,
            unregister_llm,
        )
        from ..model.rerank import (
            CustomRerankModelSpec,
            get_rerank_model_descriptions,
            register_rerank,
            unregister_rerank,
        )

        self._custom_register_type_to_cls: Dict[str, Tuple] = {  # type: ignore
            "LLM": (CustomLLMFamilyV1, register_llm, unregister_llm),
            "embedding": (
                CustomEmbeddingModelSpec,
                register_embedding,
                unregister_embedding,
            ),
            "rerank": (CustomRerankModelSpec, register_rerank, unregister_rerank),
            "audio": (CustomAudioModelFamilyV1, register_audio, unregister_audio),
            "image": (
                CustomImageModelFamilyV1,
                register_image,
                unregister_image,
            ),
        }

        # record model version
        model_version_infos: Dict[str, List[Dict]] = {}  # type: ignore
        model_version_infos.update(get_llm_model_descriptions())
        model_version_infos.update(get_embedding_model_descriptions())
        model_version_infos.update(get_rerank_model_descriptions())
        model_version_infos.update(get_image_model_descriptions())
        model_version_infos.update(get_audio_model_descriptions())
        await self._cache_tracker_ref.record_model_version(
            model_version_infos, self.address
        )

        # Windows does not have signal handler
        if os.name != "nt":

            async def signal_handler():
                try:
                    await self._supervisor_ref.remove_worker(self.address)
                except Exception as e:
                    # Ignore the error of rpc, anyway we are exiting
                    logger.exception("remove worker rpc error: %s", e)
                os._exit(0)

            loop = asyncio.get_running_loop()
            loop.add_signal_handler(
                signal.SIGINT, lambda: asyncio.create_task(signal_handler())
            )

    async def __pre_destroy__(self):
        self._isolation.stop()

    @staticmethod
    def get_devices_count():
        from ..device_utils import gpu_count

        return gpu_count()

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
        for _dev in self._total_gpu_devices:
            if (
                _dev not in self._gpu_to_model_uid
                and _dev not in self._user_specified_gpu_to_model_uids
            ):  # no possible vllm model on it, add it to candidates
                candidates.append(_dev)
            else:  # need to judge that whether to have vllm model on this device
                has_vllm_model = False
                if _dev in self._gpu_to_model_uid:
                    existing_model_uid = self._gpu_to_model_uid[_dev]
                    has_vllm_model = await self.is_model_vllm_backend(
                        existing_model_uid
                    )
                if (
                    not has_vllm_model
                    and _dev in self._user_specified_gpu_to_model_uids
                ):
                    for rep_uid, _ in self._user_specified_gpu_to_model_uids[_dev]:
                        has_vllm_model = await self.is_model_vllm_backend(rep_uid)
                        if has_vllm_model:
                            break
                if not has_vllm_model:
                    candidates.append(_dev)

        if len(candidates) == 0:
            raise RuntimeError(
                "No available slot found for the embedding model. "
                "We recommend to launch the embedding model first, and then launch the LLM models."
            )

        device, min_cnt = -1, -1
        # Pick the device with the fewest existing models among all the candidate devices.
        for _dev in candidates:
            existing_cnt = 0
            if _dev in self._gpu_to_embedding_model_uids:
                existing_cnt += len(self._gpu_to_embedding_model_uids[_dev])
            if _dev in self._gpu_to_model_uid:
                existing_cnt += 1
            if _dev in self._user_specified_gpu_to_model_uids:
                existing_cnt += len(self._user_specified_gpu_to_model_uids[_dev])
            if min_cnt == -1 or existing_cnt < min_cnt:
                device, min_cnt = _dev, existing_cnt

        self._gpu_to_embedding_model_uids[device].add(model_uid)
        return device

    def allocate_devices(self, model_uid: str, n_gpu: int) -> List[int]:
        user_specified_allocated_devices: Set[int] = set()
        for dev, model_infos in self._user_specified_gpu_to_model_uids.items():
            allocated_non_embedding_rerank_models = False
            for _, model_type in model_infos:
                allocated_non_embedding_rerank_models = model_type not in [
                    "embedding",
                    "rerank",
                ]
                if allocated_non_embedding_rerank_models:
                    break
            if allocated_non_embedding_rerank_models:
                user_specified_allocated_devices.add(dev)
        allocated_devices = set(self._gpu_to_model_uid.keys()).union(
            user_specified_allocated_devices
        )
        if n_gpu > len(self._total_gpu_devices) - len(allocated_devices):
            raise RuntimeError("No available slot found for the model")

        devices: List[int] = [
            dev
            for dev in self._total_gpu_devices
            if dev not in self._gpu_to_model_uid
            and dev not in user_specified_allocated_devices
        ][:n_gpu]
        for dev in devices:
            self._gpu_to_model_uid[int(dev)] = model_uid

        return sorted(devices)

    async def allocate_devices_with_gpu_idx(
        self, model_uid: str, model_type: str, gpu_idx: List[int]
    ) -> List[int]:
        """
        When user specifies the gpu_idx, allocate models on user-specified GPUs whenever possible
        """
        # must be subset of total devices visible to this worker
        if not set(gpu_idx) <= set(self._total_gpu_devices):
            raise ValueError(
                f"Worker {self.address} cannot use the GPUs with these indexes: {gpu_idx}. "
                f"Worker {self.address} can only see these GPUs: {self._total_gpu_devices}."
            )
        # currently just report a warning log when there are already models on these GPUs
        for idx in gpu_idx:
            existing_model_uids = []
            if idx in self._gpu_to_model_uid:
                rep_uid = self._gpu_to_model_uid[idx]
                is_vllm_model = await self.is_model_vllm_backend(rep_uid)
                if is_vllm_model:
                    raise RuntimeError(
                        f"GPU index {idx} has been occupied with a vLLM model: {rep_uid}, "
                        f"therefore cannot allocate GPU memory for a new model."
                    )
                existing_model_uids.append(rep_uid)
            if idx in self._gpu_to_embedding_model_uids:
                existing_model_uids.extend(self._gpu_to_embedding_model_uids[idx])
            # If user has run the vLLM model on the GPU that was forced to be specified,
            # it is not possible to force this GPU to be allocated again
            if idx in self._user_specified_gpu_to_model_uids:
                for rep_uid, _ in self._user_specified_gpu_to_model_uids[idx]:
                    is_vllm_model = await self.is_model_vllm_backend(rep_uid)
                    if is_vllm_model:
                        raise RuntimeError(
                            f"User specified GPU index {idx} has been occupied with a vLLM model: {rep_uid}, "
                            f"therefore cannot allocate GPU memory for a new model."
                        )

            if existing_model_uids:
                logger.warning(
                    f"WARNING!!! GPU index {idx} has been occupied "
                    f"with these models on it: {existing_model_uids}"
                )

        for idx in gpu_idx:
            self._user_specified_gpu_to_model_uids[idx].add((model_uid, model_type))
        return sorted(gpu_idx)

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

        # check user-specified slots
        for dev in self._user_specified_gpu_to_model_uids:
            model_infos = list(
                filter(
                    lambda x: x[0] == model_uid,
                    self._user_specified_gpu_to_model_uids[dev],
                )
            )
            for model_info in model_infos:
                self._user_specified_gpu_to_model_uids[dev].remove(model_info)

    async def _create_subpool(
        self,
        model_uid: str,
        model_type: Optional[str] = None,
        n_gpu: Optional[Union[int, str]] = "auto",
        gpu_idx: Optional[List[int]] = None,
    ) -> Tuple[str, List[str]]:
        env = {}
        devices = []
        env_name = get_available_device_env_name() or "CUDA_VISIBLE_DEVICES"
        if gpu_idx is None:
            if isinstance(n_gpu, int) or (n_gpu == "auto" and gpu_count() > 0):
                # Currently, n_gpu=auto means using 1 GPU
                gpu_cnt = n_gpu if isinstance(n_gpu, int) else 1
                devices = (
                    [await self.allocate_devices_for_embedding(model_uid)]
                    if model_type in ["embedding", "rerank"]
                    else self.allocate_devices(model_uid=model_uid, n_gpu=gpu_cnt)
                )
                env[env_name] = ",".join([str(dev) for dev in devices])
                logger.debug(f"GPU selected: {devices} for model {model_uid}")
            if n_gpu is None:
                env[env_name] = "-1"
                logger.debug(f"GPU disabled for model {model_uid}")
        else:
            assert isinstance(gpu_idx, list)
            devices = await self.allocate_devices_with_gpu_idx(
                model_uid, model_type, gpu_idx  # type: ignore
            )
            env[env_name] = ",".join([str(dev) for dev in devices])

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

    async def _get_model_ability(self, model: Any, model_type: str) -> List[str]:
        from ..model.llm.core import LLM

        if model_type == "embedding":
            return ["embed"]
        elif model_type == "rerank":
            return ["rerank"]
        elif model_type == "image":
            return ["text_to_image"]
        elif model_type == "audio":
            return ["audio_to_text"]
        else:
            assert model_type == "LLM"
            assert isinstance(model, LLM)
            return model.model_family.model_ability  # type: ignore

    async def update_cache_status(
        self, model_name: str, model_description: ModelDescription
    ):
        version_info = model_description.to_version_info()
        if isinstance(version_info, list):  # image model
            model_path = version_info[0]["model_file_location"]
            await self._cache_tracker_ref.update_cache_status(
                self.address, model_name, None, model_path
            )
        else:
            await self._cache_tracker_ref.update_cache_status(
                self.address,
                model_name,
                version_info["model_version"],
                version_info["model_file_location"],
            )

    @log_async(logger=logger)
    async def launch_builtin_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[Union[int, str]],
        model_format: Optional[str],
        quantization: Optional[str],
        model_engine: Optional[str],
        model_type: str = "LLM",
        n_gpu: Optional[Union[int, str]] = "auto",
        peft_model_config: Optional[PeftModelConfig] = None,
        request_limits: Optional[int] = None,
        gpu_idx: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        # !!! Note that The following code must be placed at the very beginning of this function,
        # or there will be problems with auto-recovery.
        # Because `locals()` will collect all the local parameters of this function and pass to this function again.
        launch_args = locals()
        launch_args.pop("self")
        launch_args.pop("kwargs")
        launch_args.update(kwargs)

        event_model_uid, _, __ = parse_replica_model_uid(model_uid)
        try:
            await self._event_collector_ref.report_event(
                event_model_uid,
                Event(
                    event_type=EventType.INFO,
                    event_ts=int(time.time()),
                    event_content="Launch model",
                ),
            )
        except Exception as e:
            # Report callback error can be log and ignore, should not interrupt the Process
            logger.error("report_event error: %s" % (e))

        if gpu_idx is not None:
            logger.info(
                f"You specify to launch the model: {model_name} on GPU index: {gpu_idx} "
                f"of the worker: {self.address}, "
                f"xinference will automatically ignore the `n_gpu` option."
            )
            if isinstance(gpu_idx, int):
                gpu_idx = [gpu_idx]
            assert isinstance(gpu_idx, list)

        if n_gpu is not None:
            if isinstance(n_gpu, int) and (n_gpu <= 0 or n_gpu > gpu_count()):
                raise ValueError(
                    f"The parameter `n_gpu` must be greater than 0 and "
                    f"not greater than the number of GPUs: {gpu_count()} on the machine."
                )
            if isinstance(n_gpu, str) and n_gpu != "auto":
                raise ValueError("Currently `n_gpu` only supports `auto`.")

        if peft_model_config is not None:
            if model_type in ("embedding", "rerank"):
                raise ValueError(
                    f"PEFT adaptors cannot be applied to embedding or rerank models."
                )
            if model_type == "LLM" and model_format in ("ggufv2", "ggmlv3"):
                raise ValueError(
                    f"PEFT adaptors can only be applied to pytorch-like models"
                )

        assert model_uid not in self._model_uid_to_model
        self._check_model_is_valid(model_name, model_format)

        subpool_address, devices = await self._create_subpool(
            model_uid, model_type, n_gpu=n_gpu, gpu_idx=gpu_idx
        )

        try:
            origin_uid, _, _ = parse_replica_model_uid(model_uid)
            model, model_description = await asyncio.to_thread(
                create_model_instance,
                subpool_address,
                devices,
                model_uid,
                model_type,
                model_name,
                model_engine,
                model_format,
                model_size_in_billions,
                quantization,
                peft_model_config,
                **kwargs,
            )
            await self.update_cache_status(model_name, model_description)
            model_ref = await xo.create_actor(
                ModelActor,
                address=subpool_address,
                uid=model_uid,
                worker_address=self.address,
                model=model,
                model_description=model_description,
                request_limits=request_limits,
            )
            await model_ref.load()
        except:
            logger.error(f"Failed to load model {model_uid}", exc_info=True)
            self.release_devices(model_uid=model_uid)
            await self._main_pool.remove_sub_pool(subpool_address)
            raise

        self._model_uid_to_model[model_uid] = model_ref
        self._model_uid_to_model_spec[model_uid] = model_description
        self._model_uid_to_addr[model_uid] = subpool_address
        self._model_uid_to_recover_count.setdefault(
            model_uid, MODEL_ACTOR_AUTO_RECOVER_LIMIT
        )
        self._model_uid_to_launch_args[model_uid] = launch_args

        # update status to READY
        abilities = await self._get_model_ability(model, model_type)
        await self._status_guard_ref.update_instance_info(
            origin_uid,
            {"model_ability": abilities, "status": LaunchStatus.READY.name},
        )

    @log_async(logger=logger)
    async def terminate_model(self, model_uid: str):
        event_model_uid, _, __ = parse_replica_model_uid(model_uid)
        try:
            await self._event_collector_ref.report_event(
                event_model_uid,
                Event(
                    event_type=EventType.INFO,
                    event_ts=int(time.time()),
                    event_content="Terminate model",
                ),
            )
        except Exception as e:
            # Report callback error can be log and ignore, should not interrupt the Process
            logger.error("report_event error: %s" % (e))

        origin_uid, _, _ = parse_replica_model_uid(model_uid)
        await self._status_guard_ref.update_instance_info(
            origin_uid, {"status": LaunchStatus.TERMINATING.name}
        )
        model_ref = self._model_uid_to_model.get(model_uid, None)
        if model_ref is None:
            logger.debug("Model not found, uid: %s", model_uid)

        try:
            await xo.destroy_actor(model_ref)
        except Exception as e:
            logger.debug(
                "Destroy model actor failed, model uid: %s, error: %s", model_uid, e
            )
        try:
            subpool_address = self._model_uid_to_addr[model_uid]
            await self._main_pool.remove_sub_pool(subpool_address)
        except Exception as e:
            logger.debug(
                "Remove sub pool failed, model uid: %s, error: %s", model_uid, e
            )
        finally:
            self._model_uid_to_model.pop(model_uid, None)
            self._model_uid_to_model_spec.pop(model_uid, None)
            self.release_devices(model_uid)
            self._model_uid_to_addr.pop(model_uid, None)
            self._model_uid_to_recover_count.pop(model_uid, None)
            self._model_uid_to_launch_args.pop(model_uid, None)
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
            raise ValueError(f"Model not found, uid: {model_uid}")
        return model_ref

    @log_sync(logger=logger)
    def describe_model(self, model_uid: str) -> Dict[str, Any]:
        model_desc = self._model_uid_to_model_spec.get(model_uid, None)
        if model_desc is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")
        return model_desc.to_dict()

    async def report_status(self):
        status = dict()
        try:
            # asyncio.timeout is only available in Python >= 3.11
            async with timeout(2):
                status = await asyncio.to_thread(gather_node_info)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Report status got error.")
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
                await asyncio.sleep(XINFERENCE_HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:  # pragma: no cover
                break

    async def list_cached_models(
        self, model_name: Optional[str] = None
    ) -> List[Dict[Any, Any]]:
        lists = await self._cache_tracker_ref.list_cached_models(
            self.address, model_name
        )
        cached_models = []
        for list in lists:
            cached_model = {
                "model_name": list.get("model_name"),
                "model_size_in_billions": list.get("model_size_in_billions"),
                "model_format": list.get("model_format"),
                "quantization": list.get("quantization"),
                "model_version": list.get("model_version"),
            }
            path = list.get("model_file_location")
            cached_model["path"] = path
            # parsing soft links
            if os.path.isdir(path):
                files = os.listdir(path)
                # dir has files
                if files:
                    resolved_file = os.path.realpath(os.path.join(path, files[0]))
                    if resolved_file:
                        cached_model["real_path"] = os.path.dirname(resolved_file)
            else:
                cached_model["real_path"] = os.path.realpath(path)
            cached_model["actor_ip_address"] = self.address
            cached_models.append(cached_model)
        return cached_models

    async def list_deletable_models(self, model_version: str) -> List[str]:
        paths = set()
        path = await self._cache_tracker_ref.list_deletable_models(
            model_version, self.address
        )
        if os.path.isfile(path):
            path = os.path.dirname(path)

        if os.path.isdir(path):
            files = os.listdir(path)
            paths.update([os.path.join(path, file) for file in files])
            # search real path
            if paths:
                paths.update([os.path.realpath(path) for path in paths])

        return list(paths)

    async def confirm_and_remove_model(self, model_version: str) -> bool:
        paths = await self.list_deletable_models(model_version)
        for path in paths:
            try:
                if os.path.islink(path):
                    os.unlink(path)
                elif os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    logger.debug(f"{path} is not a valid path.")
            except Exception as e:
                logger.error(f"Fail to delete {path} with error:{e}.")
                return False
        await self._cache_tracker_ref.confirm_and_remove_model(
            model_version, self.address
        )
        return True

    @staticmethod
    def record_metrics(name, op, kwargs):
        record_metrics(name, op, kwargs)
