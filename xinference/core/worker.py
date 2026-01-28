# Copyright 2022-2026 XProbe Inc.
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
import logging
import os
import pathlib
import platform
import queue
import shutil
import signal
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    no_type_check,
)

import xoscar as xo
from async_timeout import timeout
from packaging.version import Version
from xoscar import MainActorPoolType

from ..constants import (
    XINFERENCE_ALLOW_MULTI_REPLICA_PER_GPU,
    XINFERENCE_CACHE_DIR,
    XINFERENCE_DISABLE_HEALTH_CHECK,
    XINFERENCE_DISABLE_METRICS,
    XINFERENCE_ENABLE_VIRTUAL_ENV,
    XINFERENCE_HEALTH_CHECK_INTERVAL,
    XINFERENCE_VIRTUAL_ENV_DIR,
    XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED,
)
from ..core.model import ModelActor
from ..core.status_guard import LaunchStatus
from ..device_utils import get_available_device_env_name, gpu_count
from ..model.core import VirtualEnvSettings, create_model_instance
from ..model.utils import (
    CancellableDownloader,
    get_engine_params_by_name,
    get_engine_params_by_name_with_virtual_env,
)
from ..types import PeftModelConfig
from ..utils import get_pip_config_args, get_real_path
from .cache_tracker import CacheTrackerActor
from .event import Event, EventCollectorActor, EventType
from .metrics import launch_metrics_export_server, record_metrics
from .resource import gather_node_info
from .status_guard import StatusGuardActor
from .utils import (
    apply_engine_virtualenv_settings,
    build_subpool_envs_for_virtual_env,
    filter_virtualenv_packages_by_markers,
    log_async,
    log_sync,
    merge_virtual_env_packages,
    parse_replica_model_uid,
    purge_dir,
)
from .virtual_env_manager import VirtualEnvManager as XinferenceVirtualEnvManager
from .virtual_env_manager import (
    expand_engine_dependency_placeholders,
    resolve_virtualenv_python_path,
)

try:
    from xoscar.virtualenv import VirtualEnvManager
except ImportError:
    VirtualEnvManager = None

if TYPE_CHECKING:
    from .progress_tracker import Progressor

logger = getLogger(__name__)


MODEL_ACTOR_AUTO_RECOVER_LIMIT: Optional[int]
_MODEL_ACTOR_AUTO_RECOVER_LIMIT = os.getenv("XINFERENCE_MODEL_ACTOR_AUTO_RECOVER_LIMIT")
if _MODEL_ACTOR_AUTO_RECOVER_LIMIT is not None:
    MODEL_ACTOR_AUTO_RECOVER_LIMIT = int(_MODEL_ACTOR_AUTO_RECOVER_LIMIT)
else:
    MODEL_ACTOR_AUTO_RECOVER_LIMIT = None


@dataclass
class ModelStatus:
    last_error: str = ""


@dataclass
class LaunchInfo:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    # virtualenv manager
    virtual_env_manager: Optional["VirtualEnvManager"] = None
    # downloader, report progress or cancel entire download
    downloader: Optional[CancellableDownloader] = None
    # sub pools created for the model
    sub_pools: Optional[List[str]] = None


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
        self._supervisor_ref: Optional[xo.ActorRefType] = None
        self._main_pool = main_pool
        self._main_pool.recover_sub_pool = self.recover_sub_pool
        self._status_guard_ref: xo.ActorRefType[
            "StatusGuardActor"
        ] = None  # type: ignore
        self._event_collector_ref: xo.ActorRefType[  # type: ignore
            EventCollectorActor
        ] = None
        self._cache_tracker_ref: xo.ActorRefType[
            CacheTrackerActor
        ] = None  # type: ignore

        # Virtual environment management
        self._virtual_env_manager: XinferenceVirtualEnvManager = None  # type: ignore

        # internal states.
        # temporary placeholder during model launch process:
        self._model_uid_launching_guard: Dict[str, LaunchInfo] = {}
        # attributes maintained after model launched:
        self._model_uid_to_model: Dict[str, xo.ActorRefType["ModelActor"]] = {}
        self._model_uid_to_model_spec: Dict[str, Dict[str, Any]] = {}
        self._model_uid_to_model_status: Dict[str, ModelStatus] = {}
        self._gpu_to_model_uids: Dict[int, Set[str]] = defaultdict(set)
        # Dict structure: gpu_index: {(replica_model_uid, model_type)}
        self._user_specified_gpu_to_model_uids: Dict[int, Set[Tuple[str, str]]] = (
            defaultdict(set)
        )
        self._allow_multi_replica_per_gpu = XINFERENCE_ALLOW_MULTI_REPLICA_PER_GPU
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
                f"Starting metrics export server at {metrics_exporter_host}:{metrics_exporter_port}"  # noqa: E231
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
                    logger.info(
                        f"Metrics server is started at: http://{host}:{port}"  # noqa: E231
                    )
                    break
                except queue.Empty:
                    pass
            else:
                raise Exception("Metrics server thread exit.")

        # Initialize virtual environment manager
        self._virtual_env_manager = XinferenceVirtualEnvManager(self.address)

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
                        await self.terminate_model(model_uid, is_model_die=True)
                    except Exception:
                        pass
                    if recover_count is not None:
                        if recover_count > 0:
                            logger.warning(
                                "Recreating model actor %s, remain %s times ...",
                                model_uid,
                                recover_count - 1,
                            )
                            event_model_uid, _ = parse_replica_model_uid(model_uid)
                            try:
                                if self._event_collector_ref is not None:
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
                            finally:
                                del event_model_uid

                            self._model_uid_to_recover_count[model_uid] = (
                                recover_count - 1
                            )
                            await self.recover_model(launch_args)
                        else:
                            logger.warning("Stop recreating model actor.")
                    else:
                        logger.warning("Recreating model actor %s ...", model_uid)
                        await self.recover_model(launch_args)
                break

    @classmethod
    def default_uid(cls) -> str:
        return "worker"

    def _get_spec_dicts_with_cache_status(
        self, model_family: Any, cache_manager_cls: Type
    ) -> Tuple[List[dict], List[str]]:
        """
        Build model_specs with cache_status and collect download_hubs.
        """

        specs: List[dict] = []
        download_hubs: List[str] = []
        for spec in model_family.model_specs:
            model_hub = spec.model_hub
            if model_hub not in download_hubs:
                download_hubs.append(model_hub)

            family_copy = model_family.copy()
            family_copy.model_specs = [spec]
            cache_manager = cache_manager_cls(family_copy)
            specs.append(
                {**spec.dict(), "cache_status": cache_manager.get_cache_status()}
            )
        return specs, download_hubs

    def _prefer_model_hub(self, model_family: Any, preferred_hub: str = "huggingface"):
        """
        Return a copy of model_family with a single spec, preferring the given hub.
        """
        specs = getattr(model_family, "model_specs", None)
        if not specs:
            return model_family

        target_spec = next(
            (
                spec
                for spec in specs
                if getattr(spec, "model_hub", None) == preferred_hub
            ),
            specs[0],
        )
        family_copy = model_family.copy()
        family_copy.model_specs = [target_spec]
        return family_copy

    async def __post_create__(self):
        from ..model.audio import (
            CustomAudioModelFamilyV2,
            generate_audio_description,
            register_audio,
            unregister_audio,
        )
        from ..model.embedding import (
            CustomEmbeddingModelFamilyV2,
            generate_embedding_description,
            register_embedding,
            unregister_embedding,
        )
        from ..model.flexible import (
            FlexibleModelSpec,
            generate_flexible_model_description,
            register_flexible_model,
            unregister_flexible_model,
        )
        from ..model.image import (
            CustomImageModelFamilyV2,
            generate_image_description,
            register_image,
            unregister_image,
        )
        from ..model.llm import (
            CustomLLMFamilyV2,
            generate_llm_version_info,
            register_llm,
            unregister_llm,
        )
        from ..model.rerank import (
            CustomRerankModelFamilyV2,
            generate_rerank_description,
            register_rerank,
            unregister_rerank,
        )
        from ..model.video import (
            CustomVideoModelFamilyV2,
            generate_video_description,
            register_video,
            unregister_video,
        )

        self._custom_register_type_to_cls: Dict[str, Tuple] = {  # type: ignore
            "LLM": (
                CustomLLMFamilyV2,
                register_llm,
                unregister_llm,
                generate_llm_version_info,
            ),
            "embedding": (
                CustomEmbeddingModelFamilyV2,
                register_embedding,
                unregister_embedding,
                generate_embedding_description,
            ),
            "rerank": (
                CustomRerankModelFamilyV2,
                register_rerank,
                unregister_rerank,
                generate_rerank_description,
            ),
            "image": (
                CustomImageModelFamilyV2,
                register_image,
                unregister_image,
                generate_image_description,
            ),
            "audio": (
                CustomAudioModelFamilyV2,
                register_audio,
                unregister_audio,
                generate_audio_description,
            ),
            "flexible": (
                FlexibleModelSpec,
                register_flexible_model,
                unregister_flexible_model,
                generate_flexible_model_description,
            ),
            "video": (
                CustomVideoModelFamilyV2,
                register_video,
                unregister_video,
                generate_video_description,
            ),
        }

        logger.info("Purge cache directory: %s", XINFERENCE_CACHE_DIR)
        purge_dir(XINFERENCE_CACHE_DIR)

        try:
            await self.get_supervisor_ref(add_worker=True)
        except Exception:
            # Do not crash the worker if supervisor is down, auto re-connect later
            logger.error(f"cannot connect to supervisor", exc_info=True)

        if not XINFERENCE_DISABLE_HEALTH_CHECK:
            from ..isolation import Isolation

            # Run _periodical_report_status() in a dedicated thread.
            self._isolation = Isolation(asyncio.new_event_loop(), threaded=True)
            self._isolation.start()
            asyncio.run_coroutine_threadsafe(
                self._periodical_report_status(), loop=self._isolation.loop
            )
        logger.info(f"Xinference worker {self.address} started")

        # Windows does not have signal handler
        if os.name != "nt":

            async def signal_handler():
                try:
                    supervisor_ref = await self.get_supervisor_ref(add_worker=False)
                    await supervisor_ref.remove_worker(self.address)
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

    async def trigger_exit(self) -> bool:
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except Exception as e:
            logger.info(f"trigger exit error: {e}")
            return False
        return True

    async def get_supervisor_ref(self, add_worker: bool = True) -> xo.ActorRefType:
        """
        Try connect to supervisor and return ActorRef. Raise exception on error
        Params:
            add_worker: By default will call supervisor.add_worker after first connect
        """
        from .supervisor import SupervisorActor

        if self._supervisor_ref is not None:
            return self._supervisor_ref
        supervisor_ref = await xo.actor_ref(  # type: ignore
            address=self._supervisor_address, uid=SupervisorActor.default_uid()
        )
        # Prevent concurrent operations leads to double initialization, check again.
        if self._supervisor_ref is not None:
            return self._supervisor_ref
        self._supervisor_ref = supervisor_ref
        if add_worker and len(self._model_uid_to_model) == 0:
            # Newly started (or restarted), has no model, notify supervisor
            await self._supervisor_ref.add_worker(self.address)
            logger.info("Connected to supervisor as a fresh worker")

        self._status_guard_ref = await xo.actor_ref(
            address=self._supervisor_address, uid=StatusGuardActor.default_uid()
        )
        self._event_collector_ref = await xo.actor_ref(
            address=self._supervisor_address, uid=EventCollectorActor.default_uid()
        )
        self._cache_tracker_ref = await xo.actor_ref(
            address=self._supervisor_address, uid=CacheTrackerActor.default_uid()
        )
        self._progress_tracker_ref = None
        # cache_tracker is on supervisor
        from ..model.audio import get_audio_model_descriptions
        from ..model.embedding import get_embedding_model_descriptions
        from ..model.flexible import get_flexible_model_descriptions
        from ..model.image import get_image_model_descriptions
        from ..model.llm import get_llm_version_infos
        from ..model.rerank import get_rerank_model_descriptions
        from ..model.video import get_video_model_descriptions

        # record model version
        model_version_infos: Dict[str, List[Dict]] = {}  # type: ignore
        model_version_infos.update(get_llm_version_infos())
        model_version_infos.update(get_embedding_model_descriptions())
        model_version_infos.update(get_rerank_model_descriptions())
        model_version_infos.update(get_image_model_descriptions())
        model_version_infos.update(get_audio_model_descriptions())
        model_version_infos.update(get_video_model_descriptions())
        model_version_infos.update(get_flexible_model_descriptions())
        await self._cache_tracker_ref.record_model_version(
            model_version_infos, self.address
        )
        return self._supervisor_ref

    @staticmethod
    def get_devices_count():
        from ..device_utils import gpu_count

        return gpu_count()

    @log_sync(logger=logger)
    def get_model_count(self) -> int:
        return len(self._model_uid_to_model)

    async def is_model_vllm_backend(self, model_uid: str) -> bool:
        _model_uid, _ = parse_replica_model_uid(model_uid)
        supervisor_ref = await self.get_supervisor_ref()
        model_ref = await supervisor_ref.get_model(_model_uid)
        return await model_ref.is_vllm_backend()

    def allocate_devices(self, model_uid: str, n_gpu: int) -> List[int]:
        if n_gpu > len(self._total_gpu_devices):
            raise RuntimeError("Requested GPUs exceed the number of available devices")

        # If multi-replica-per-GPU is disabled, only pick currently idle GPUs.
        if not self._allow_multi_replica_per_gpu:
            occupied_devices: Set[int] = set()
            for dev, model_uids in self._gpu_to_model_uids.items():
                if model_uids:
                    occupied_devices.add(dev)
            for dev, model_infos in self._user_specified_gpu_to_model_uids.items():
                if model_infos:
                    occupied_devices.add(dev)
            available_devices = [
                dev for dev in self._total_gpu_devices if dev not in occupied_devices
            ]
            if n_gpu > len(available_devices):
                raise RuntimeError("No available slot found for the model")
            selected_devices = available_devices[:n_gpu]
            for dev in selected_devices:
                self._gpu_to_model_uids[int(dev)].add(model_uid)
            return sorted(selected_devices)

        # Default: allow multi-tenant GPUs, pick least-loaded devices.
        gpu_loads: List[Tuple[int, int, int]] = []
        for dev in self._total_gpu_devices:
            running_models = len(self._gpu_to_model_uids.get(dev, set()))
            load = running_models + len(
                self._user_specified_gpu_to_model_uids.get(dev, set())
            )
            # Prefer devices with fewer existing model processes when loads tie
            gpu_loads.append((load, running_models, dev))

        devices: List[int] = []
        for _ in range(n_gpu):
            gpu_loads.sort(key=lambda x: (x[0], x[1], x[2]))
            load, running_models, dev = gpu_loads[0]
            devices.append(dev)
            gpu_loads[0] = (load + 1, running_models + 1, dev)

        for dev in devices:
            self._gpu_to_model_uids[int(dev)].add(model_uid)

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
            if idx in self._gpu_to_model_uids:
                for rep_uid in self._gpu_to_model_uids[idx]:
                    existing_model_uids.append(rep_uid)
            if not self._allow_multi_replica_per_gpu and (
                existing_model_uids
                or len(self._user_specified_gpu_to_model_uids.get(idx, set())) > 0
            ):
                raise RuntimeError(
                    f"GPU index {idx} has been occupied with models: {existing_model_uids}, "
                    f"and multi-replica-per-GPU is disabled."
                )

            if existing_model_uids:
                logger.warning(
                    f"WARNING!!! GPU index {idx} has been occupied "
                    f"with these models on it: {existing_model_uids}"
                )

        for idx in gpu_idx:
            self._user_specified_gpu_to_model_uids[idx].add((model_uid, model_type))
        return sorted(gpu_idx)

    @log_async(logger=logger)
    async def get_gpu_allocation_status(self) -> Dict[str, Any]:
        """Return current device allocation snapshot for scheduling/diagnostics."""
        return {
            "total": list(self._total_gpu_devices),
            "models": {int(k): list(v) for k, v in self._gpu_to_model_uids.items()},
            "user_specified": {
                int(k): [list(t) for t in v]
                for k, v in self._user_specified_gpu_to_model_uids.items()
            },
            "allow_multi_replica_per_gpu": self._allow_multi_replica_per_gpu,
        }

    def release_devices(self, model_uid: str):
        for dev, model_uids in list(self._gpu_to_model_uids.items()):
            if model_uid in model_uids:
                model_uids.remove(model_uid)
                if not model_uids:
                    del self._gpu_to_model_uids[dev]

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
        env: Optional[Dict[str, str]] = None,
        start_python: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        env = {} if env is None else env
        devices = []
        env_name = get_available_device_env_name() or "CUDA_VISIBLE_DEVICES"
        if gpu_idx is None:
            if isinstance(n_gpu, int) or (n_gpu == "auto" and gpu_count() > 0):
                # Currently, n_gpu=auto means using 1 GPU
                gpu_cnt = n_gpu if isinstance(n_gpu, int) else 1
                devices = self.allocate_devices(model_uid=model_uid, n_gpu=gpu_cnt)
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

        subpool_address = await self._main_pool.append_sub_pool(
            env=env, start_python=start_python
        )
        return subpool_address, [str(dev) for dev in devices]

    def _check_model_is_valid(self, model_name: str, model_format: Optional[str]):
        # baichuan-base and baichuan-chat depend on `cpm_kernels` module,
        # but `cpm_kernels` cannot run on Darwin system.
        if platform.system() == "Darwin" and model_format == "pytorch":
            if "baichuan" in model_name:
                raise ValueError(f"{model_name} model can't run on Darwin system.")

    @log_sync(logger=logger)
    async def register_model(self, model_type: str, model: str, persist: bool):
        # TODO: centralized model registrations
        if model_type in self._custom_register_type_to_cls:
            (
                model_spec_cls,
                register_fn,
                unregister_fn,
                generate_fn,
            ) = self._custom_register_type_to_cls[model_type]
            model_spec = model_spec_cls.parse_raw(model)
            try:
                register_fn(model_spec, persist)
                await self._cache_tracker_ref.record_model_version(
                    generate_fn(model_spec), self.address
                )
            except ValueError as e:
                raise e
            except Exception as e:
                unregister_fn(model_spec.model_name, raise_error=False)
                raise e
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_sync(logger=logger)
    async def unregister_model(self, model_type: str, model_name: str):
        # TODO: centralized model registrations
        if model_type in self._custom_register_type_to_cls:
            _, _, unregister_fn, _ = self._custom_register_type_to_cls[model_type]
            unregister_fn(model_name, False)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def update_model_type(self, model_type: str):
        """
        Update model configurations for a specific model type by downloading
        the latest JSON from the remote API and storing it locally.

        Args:
            model_type: Type of model (LLM, embedding, image, etc.)
        """
        import json

        import requests

        supported_types = list(self._custom_register_type_to_cls.keys())

        normalized_for_validation = model_type
        if model_type.lower() == "llm" and "LLM" in supported_types:
            normalized_for_validation = "LLM"
        elif model_type.lower() == "llm" and "llm" in supported_types:
            normalized_for_validation = "llm"

        if normalized_for_validation not in supported_types:
            logger.error(f"Unsupported model type: {normalized_for_validation}")
            raise ValueError(
                f"Unsupported model type '{model_type}'. "
                f"Supported types are: {', '.join(supported_types)}"
            )

        # Construct the URL to download JSON
        url = f"https://model.xinference.io/api/models/download?model_type={model_type.lower()}"

        try:
            # Download JSON from remote API
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse JSON response
            model_data = response.json()

            # Store the JSON data using CacheManager as built-in models
            await self._store_complete_model_configurations(model_type, model_data)

            # Dynamically reload built-in models to make them immediately available
            try:
                if model_type.lower() == "llm":
                    from ..model.llm import register_builtin_model

                    register_builtin_model()
                elif model_type.lower() == "embedding":
                    from ..model.embedding import register_builtin_model

                    register_builtin_model()
                elif model_type.lower() == "audio":
                    from ..model.audio import register_builtin_model

                    register_builtin_model()
                elif model_type.lower() == "image":
                    from ..model.image import register_builtin_model

                    register_builtin_model()
                elif model_type.lower() == "rerank":
                    from ..model.rerank import register_builtin_model

                    register_builtin_model()
                elif model_type.lower() == "video":
                    from ..model.video import register_builtin_model

                    register_builtin_model()
                else:
                    logger.warning(
                        f"No dynamic loading available for model type: {model_type}"
                    )
            except Exception as reload_error:
                logger.error(
                    f"Error reloading built-in models: {reload_error}",
                    exc_info=True,
                )
                # Don't fail the update if reload fails, just log the error

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error downloading model configurations: {e}")
            raise ValueError(f"Failed to download model configurations: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise ValueError(f"Invalid JSON response from remote API: {str(e)}")
        except Exception as e:
            logger.error(
                f"Unexpected error during model update: {e}",
                exc_info=True,
            )
            raise ValueError(f"Failed to update model configurations: {str(e)}")

    async def _store_complete_model_configurations(self, model_type: str, model_data):
        """
        Store complete model configurations as a unified JSON file.
        This is used by update_model_type to preserve the original JSON structure.

        Args:
            model_type: Type of model (as provided by user, e.g., "llm")
            model_data: JSON data containing model configurations (complete array)
        """
        import json

        from ..constants import XINFERENCE_MODEL_DIR

        try:
            model_type_lower = model_type.lower()

            # Use the unified JSON file path (same as original update_model_type logic)
            builtin_dir = os.path.join(
                XINFERENCE_MODEL_DIR, "v2", "builtin", model_type_lower
            )
            json_file_path = os.path.join(
                builtin_dir, f"{model_type_lower}_models.json"
            )

            # Ensure directory exists
            os.makedirs(builtin_dir, exist_ok=True)

            # Store the complete JSON file (preserving original structure)
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(
                f"Error storing complete model configurations: {str(e)}",
                exc_info=True,
            )
            raise ValueError(f"Failed to store complete model configurations: {str(e)}")

    @log_async(logger=logger)
    async def list_model_registrations(
        self, model_type: str, detailed: bool = False
    ) -> List[Dict[str, Any]]:
        def sort_helper(item):
            assert isinstance(item["model_name"], str)
            return item.get("model_name").lower()

        ret = []

        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families
            from ..model.llm.cache_manager import LLMCacheManager

            # Add built-in LLM families
            for family in BUILTIN_LLM_FAMILIES:
                if detailed:
                    specs, download_hubs = self._get_spec_dicts_with_cache_status(
                        family, LLMCacheManager
                    )
                    ret.append(
                        {
                            **family.dict(),
                            "model_specs": specs,
                            "is_builtin": True,
                            "download_hubs": download_hubs,
                        }
                    )
                else:
                    ret.append({"model_name": family.model_name, "is_builtin": True})

            # Add user-defined LLM families
            for family in get_user_defined_llm_families():
                if detailed:
                    specs, download_hubs = self._get_spec_dicts_with_cache_status(
                        family, LLMCacheManager
                    )
                    ret.append(
                        {
                            **family.dict(),
                            "model_specs": specs,
                            "is_builtin": False,
                            "download_hubs": download_hubs,
                        }
                    )
                else:
                    ret.append({"model_name": family.model_name, "is_builtin": False})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "embedding":
            from ..model.embedding import BUILTIN_EMBEDDING_MODELS
            from ..model.embedding.cache_manager import EmbeddingCacheManager
            from ..model.embedding.custom import get_user_defined_embeddings

            # Add built-in embedding models
            for model_name, family_list in BUILTIN_EMBEDDING_MODELS.items():
                for family in family_list:
                    if detailed:
                        specs, download_hubs = self._get_spec_dicts_with_cache_status(
                            family, EmbeddingCacheManager
                        )
                        ret.append(
                            {
                                **family.dict(),
                                "model_specs": specs,
                                "is_builtin": True,
                                "download_hubs": download_hubs,
                            }
                        )
                    else:
                        ret.append({"model_name": model_name, "is_builtin": True})

            # Add user-defined embedding models
            for model_spec in get_user_defined_embeddings():
                if detailed:
                    specs, download_hubs = self._get_spec_dicts_with_cache_status(
                        model_spec, EmbeddingCacheManager
                    )
                    ret.append(
                        {
                            **model_spec.dict(),
                            "model_specs": specs,
                            "is_builtin": False,
                            "download_hubs": download_hubs,
                        }
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "image":
            from ..model.image import BUILTIN_IMAGE_MODELS
            from ..model.image.cache_manager import ImageCacheManager
            from ..model.image.custom import get_user_defined_images

            # Add built-in image models (BUILTIN_IMAGE_MODELS contains model_name -> families list)
            for model_name, families in BUILTIN_IMAGE_MODELS.items():
                download_hubs = []
                for family in families:
                    if family.model_hub not in download_hubs:
                        download_hubs.append(family.model_hub)
                for family in families:
                    if detailed:
                        cache_manager = ImageCacheManager(family)
                        model_specs = [
                            {
                                "model_format": "pytorch",
                                "model_hub": family.model_hub,
                                "model_id": family.model_id,
                                "cache_status": cache_manager.get_cache_status(),
                            }
                        ]
                        ret.append(
                            {
                                **family.dict(),
                                "model_specs": model_specs,
                                "is_builtin": True,
                                "download_hubs": download_hubs,
                            }
                        )
                    else:
                        ret.append({"model_name": model_name, "is_builtin": True})

            # Add user-defined image models
            for model_spec in get_user_defined_images():
                if detailed:
                    cache_manager = ImageCacheManager(model_spec)
                    model_specs = [
                        {
                            "model_format": "pytorch",
                            "model_hub": model_spec.model_hub,
                            "model_id": model_spec.model_id,
                            "cache_status": cache_manager.get_cache_status(),
                        }
                    ]
                    ret.append(
                        {
                            **model_spec.dict(),
                            "model_specs": model_specs,
                            "is_builtin": False,
                            "download_hubs": [model_spec.model_hub],
                        }
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "audio":
            from ..model.audio import BUILTIN_AUDIO_MODELS
            from ..model.audio.custom import get_user_defined_audios
            from ..model.cache_manager import CacheManager

            # Add built-in audio models (BUILTIN_AUDIO_MODELS contains model_name -> families list)
            for model_name, families in BUILTIN_AUDIO_MODELS.items():
                download_hubs = []
                for family in families:
                    if family.model_hub not in download_hubs:
                        download_hubs.append(family.model_hub)
                for family in families:
                    if detailed:
                        audio_cache_manager = CacheManager(family)
                        model_specs = [
                            {
                                "model_format": "pytorch",
                                "model_hub": family.model_hub,
                                "model_id": family.model_id,
                                "cache_status": audio_cache_manager.get_cache_status(),
                            }
                        ]
                        ret.append(
                            {
                                **family.dict(),
                                "model_specs": model_specs,
                                "is_builtin": True,
                                "download_hubs": download_hubs,
                            }
                        )
                    else:
                        ret.append({"model_name": model_name, "is_builtin": True})

            # Add user-defined audio models
            for model_spec in get_user_defined_audios():
                if detailed:
                    audio_cache_manager = CacheManager(model_spec)
                    model_specs = [
                        {
                            "model_format": "pytorch",
                            "model_hub": model_spec.model_hub,
                            "model_id": model_spec.model_id,
                            "cache_status": audio_cache_manager.get_cache_status(),
                        }
                    ]
                    ret.append(
                        {
                            **model_spec.dict(),
                            "model_specs": model_specs,
                            "is_builtin": False,
                            "download_hubs": [model_spec.model_hub],
                        }
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "video":
            from ..model.cache_manager import CacheManager
            from ..model.video import BUILTIN_VIDEO_MODELS

            # Add built-in video models (BUILTIN_VIDEO_MODELS contains model_name -> families list)
            for model_name, families in BUILTIN_VIDEO_MODELS.items():
                download_hubs = []
                for family in families:
                    if family.model_hub not in download_hubs:
                        download_hubs.append(family.model_hub)
                for family in families:
                    if detailed:
                        video_cache_manager = CacheManager(family)
                        model_specs = [
                            {
                                "model_format": "pytorch",
                                "model_hub": family.model_hub,
                                "model_id": family.model_id,
                                "cache_status": video_cache_manager.get_cache_status(),
                                "gguf_model_id": family.gguf_model_id,
                                "gguf_quantizations": family.gguf_quantizations,
                                "gguf_model_file_name_template": (
                                    family.gguf_model_file_name_template
                                ),
                            }
                        ]
                        ret.append(
                            {
                                **family.dict(),
                                "model_specs": model_specs,
                                "is_builtin": True,
                                "download_hubs": download_hubs,
                            }
                        )
                    else:
                        ret.append({"model_name": model_name, "is_builtin": True})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "rerank":
            from ..model.rerank import BUILTIN_RERANK_MODELS
            from ..model.rerank.cache_manager import RerankCacheManager
            from ..model.rerank.custom import get_user_defined_reranks

            # Add built-in rerank models (BUILTIN_RERANK_MODELS contains model_name -> family_list list)
            for model_name, family_list in BUILTIN_RERANK_MODELS.items():
                for family in family_list:
                    if detailed:
                        specs, download_hubs = self._get_spec_dicts_with_cache_status(
                            family, RerankCacheManager
                        )
                        ret.append(
                            {
                                **family.dict(),
                                "model_specs": specs,
                                "is_builtin": True,
                                "download_hubs": download_hubs,
                            }
                        )
                    else:
                        ret.append({"model_name": model_name, "is_builtin": True})

            # Add user-defined rerank models
            for model_spec in get_user_defined_reranks():
                if detailed:
                    specs, download_hubs = self._get_spec_dicts_with_cache_status(
                        model_spec, RerankCacheManager
                    )
                    ret.append(
                        {
                            **model_spec.dict(),
                            "model_specs": specs,
                            "is_builtin": False,
                            "download_hubs": download_hubs,
                        }
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "flexible":
            from ..model.flexible.custom import get_flexible_models

            ret = []

            for model_spec in get_flexible_models():
                if detailed:
                    ret.append(
                        {
                            **model_spec.dict(),
                            "cache_status": True,
                            "is_builtin": False,
                        }
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
    async def get_model_registration(self, model_type: str, model_name: str) -> Any:
        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families

            # Check built-in LLM families
            for f in BUILTIN_LLM_FAMILIES:
                if f.model_name == model_name:
                    return f

            # Check user-defined LLM families
            for f in get_user_defined_llm_families():
                if f.model_name == model_name:
                    return f
        elif model_type == "embedding":
            from ..model.embedding import BUILTIN_EMBEDDING_MODELS
            from ..model.embedding.custom import get_user_defined_embeddings

            # Check built-in embedding models
            for builtin_model_name, family_list in BUILTIN_EMBEDDING_MODELS.items():
                if builtin_model_name != model_name:
                    continue
                for family in family_list:
                    return self._prefer_model_hub(family)

            # Check user-defined embedding models
            for f in get_user_defined_embeddings():
                if f.model_name == model_name:
                    return self._prefer_model_hub(f)
        elif model_type == "image":
            from ..model.image import BUILTIN_IMAGE_MODELS
            from ..model.image.custom import get_user_defined_images

            # Check built-in image models
            if model_name in BUILTIN_IMAGE_MODELS:
                families = BUILTIN_IMAGE_MODELS[model_name]
                for f in families:
                    if f.model_hub == "huggingface":
                        return f

            # Check user-defined image models
            for f in get_user_defined_images():
                if f.model_name == model_name:
                    return f
        elif model_type == "audio":
            from ..model.audio import BUILTIN_AUDIO_MODELS
            from ..model.audio.custom import get_user_defined_audios

            # Check built-in audio models
            if model_name in BUILTIN_AUDIO_MODELS:
                families = BUILTIN_AUDIO_MODELS[model_name]
                for f in families:
                    if f.model_hub == "huggingface":
                        return f

            # Check user-defined audio models
            for f in get_user_defined_audios():
                if f.model_name == model_name:
                    return f
        elif model_type == "video":
            from ..model.video import BUILTIN_VIDEO_MODELS

            # Check built-in video models
            if model_name in BUILTIN_VIDEO_MODELS:
                families = BUILTIN_VIDEO_MODELS[model_name]
                for f in families:
                    if f.model_hub == "huggingface":
                        return f
            return None
        elif model_type == "rerank":
            from ..model.rerank import BUILTIN_RERANK_MODELS
            from ..model.rerank.custom import get_user_defined_reranks

            # Check built-in rerank models
            if model_name in BUILTIN_RERANK_MODELS:
                family_list = BUILTIN_RERANK_MODELS[model_name]
                for f in family_list:
                    return self._prefer_model_hub(f)

            # Check user-defined rerank models
            for f in get_user_defined_reranks():
                if f.model_name == model_name:
                    return self._prefer_model_hub(f)
        return None

    @log_async(logger=logger)
    async def query_engines_by_model_name(
        self,
        model_name: str,
        model_type: Optional[str] = None,
        enable_virtual_env: Optional[bool] = None,
    ):
        if enable_virtual_env is None:
            enable_virtual_env = XINFERENCE_ENABLE_VIRTUAL_ENV
        if enable_virtual_env:
            return get_engine_params_by_name_with_virtual_env(
                model_type, model_name, enable_virtual_env=enable_virtual_env
            )
        return get_engine_params_by_name(
            model_type, model_name, enable_virtual_env=enable_virtual_env
        )

    async def _get_model_ability(self, model: Any, model_type: str) -> List[str]:
        from ..model.llm.core import LLM

        ability_map = {
            "embedding": ["embed"],
            "rerank": ["rerank"],
            "flexible": ["flexible"],
        }
        if model_type in ability_map:
            return ability_map[model_type]
        if model_type in {"image", "audio", "video"}:
            return model.model_ability
        assert model_type == "LLM"
        assert isinstance(model, LLM)
        return model.model_family.model_ability  # type: ignore

    async def update_cache_status(self, model_name: str, version_info: Any):
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

    @classmethod
    def _create_virtual_env_manager(
        cls,
        enable_virtual_env: Optional[bool],
        virtual_env_name: Optional[str],
        env_path: str,
    ) -> Optional[VirtualEnvManager]:
        if enable_virtual_env is None:
            enable_virtual_env = XINFERENCE_ENABLE_VIRTUAL_ENV

        if not enable_virtual_env:
            # skip preparing virtualenv
            return None

        from xoscar.virtualenv import get_virtual_env_manager

        virtual_env_manager: VirtualEnvManager = get_virtual_env_manager(
            virtual_env_name or "uv", env_path
        )
        # create env
        python_path = None
        if not hasattr(sys, "_MEIPASS"):
            # not in pyinstaller
            # we specify python_path explicitly
            # sometimes uv would find other versions.
            python_path = pathlib.Path(sys.executable)
        virtual_env_manager.create_env(python_path=python_path)
        return virtual_env_manager

    @classmethod
    def _prepare_virtual_env(
        cls,
        virtual_env_manager: "VirtualEnvManager",
        settings: Optional[VirtualEnvSettings],
        virtual_env_packages: Optional[List[str]],
        model_engine: Optional[str],
    ):
        engine_defaults: List[str] = []
        if (
            (not settings or not settings.packages)
            and not virtual_env_packages
            and not engine_defaults
        ):
            # no settings or no packages
            return

        if settings is None:
            settings = VirtualEnvSettings(packages=virtual_env_packages or [])

        if settings and model_engine and model_engine.lower() not in ("vllm", "sglang"):
            # Pydantic v1 compatibility: use copy() when model_copy is unavailable.
            if hasattr(settings, "model_copy"):
                settings = settings.model_copy(deep=True)
            else:
                settings = settings.copy(deep=True)
            settings.extra_index_url = None
            settings.index_strategy = None

        if settings.inherit_pip_config:
            # inherit pip config
            pip_config = get_pip_config_args()
            for k, v in pip_config.items():
                if hasattr(settings, k) and not getattr(settings, k):
                    setattr(settings, k, v)

        apply_engine_virtualenv_settings(settings, model_engine)

        base_packages = engine_defaults
        if settings.packages:
            base_packages = base_packages + settings.packages.copy()
        packages = merge_virtual_env_packages(base_packages, virtual_env_packages)
        packages = expand_engine_dependency_placeholders(packages, model_engine)

        try:
            from xoscar.virtualenv.platform import get_cuda_version

            cuda_version = get_cuda_version()
        except Exception:
            cuda_version = None

        if not cuda_version or Version(cuda_version) < Version("13.0"):
            logger.debug(
                f"[DEBUG] CUDA version check: cuda_version={cuda_version}, clearing extra_index_url and index_strategy"
            )
            settings.extra_index_url = None
            settings.index_strategy = None
        else:
            logger.debug(
                f"[DEBUG] CUDA version check passed: cuda_version={cuda_version}, keeping settings.extra_index_url={settings.extra_index_url}"
            )

        packages = filter_virtualenv_packages_by_markers(
            packages, model_engine, cuda_version
        )

        conf = dict(settings)
        conf.pop("packages", None)
        conf.pop("inherit_pip_config", None)
        if XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED:
            conf["skip_installed"] = XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED
        variables = {}
        if model_engine:
            engine_value = model_engine.lower()
            variables["engine"] = engine_value
            variables["model_engine"] = engine_value

        logger.info(
            "Installing packages %s in virtual env %s, with settings(%s)",
            packages,
            virtual_env_manager.env_path,
            ", ".join([f"{k}={v}" for k, v in conf.items() if v]),
        )
        virtual_env_manager.install_packages(packages, **conf, **variables)

    async def _get_progressor(self, request_id: str):
        from .progress_tracker import Progressor, ProgressTrackerActor

        progress_tracker_ref = self._progress_tracker_ref
        if progress_tracker_ref is None:
            progress_tracker_ref = self._progress_tracker_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=ProgressTrackerActor.default_uid()
            )

        progressor = Progressor(
            request_id,
            progress_tracker_ref,
            asyncio.get_running_loop(),
        )
        await progressor.start()
        progressor.set_progress(0.0, "start to launch model")
        return progressor

    @classmethod
    def _upload_download_progress(
        cls, progressor: "Progressor", downloader: CancellableDownloader
    ):
        while not downloader.done:
            progress = downloader.get_progress()
            progressor.set_progress(progress)
            downloader.wait(1)

        progressor.set_progress(1.0, "Start to load model")

    @log_async(logger=logger, level=logging.INFO)
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
        n_worker: Optional[int] = 1,
        shard: Optional[int] = 0,
        driver_info: Optional[dict] = None,
        peft_model_config: Optional[PeftModelConfig] = None,
        request_limits: Optional[int] = None,
        gpu_idx: Optional[Union[int, List[int]]] = None,
        download_hub: Optional[
            Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
        ] = None,
        model_path: Optional[str] = None,
        enable_virtual_env: Optional[bool] = None,
        virtual_env_packages: Optional[List[str]] = None,
        envs: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        # !!! Note that The following code must be placed at the very beginning of this function,
        # or there will be problems with auto-recovery.
        # Because `locals()` will collect all the local parameters of this function and pass to this function again.
        launch_args = locals()
        launch_args.pop("self")
        launch_args.pop("kwargs")
        launch_args.update(kwargs)

        try:
            origin_uid, _ = parse_replica_model_uid(model_uid)
        except Exception as e:
            logger.exception(e)
            raise
        try:
            _ = await self.get_supervisor_ref()
            if self._event_collector_ref is not None:
                await self._event_collector_ref.report_event(
                    origin_uid,
                    Event(
                        event_type=EventType.INFO,
                        event_ts=int(time.time()),
                        event_content="Launch model",
                    ),
                )
        except Exception as e:
            # Report callback error can be log and ignore, should not interrupt the Process
            logger.error("report_event error: %s" % (e), exc_info=True)

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

        device = kwargs.get("device")
        if device and device.lower().startswith("cpu"):
            n_gpu = None

        if peft_model_config is not None:
            if model_type in ("embedding", "rerank"):
                raise ValueError(
                    f"PEFT adaptors cannot be applied to embedding or rerank models."
                )
            if model_type == "LLM" and model_format in ("ggufv2",):
                raise ValueError(
                    f"PEFT adaptors can only be applied to pytorch-like models"
                )
        if model_path is not None:
            if not os.path.exists(model_path):
                raise ValueError(
                    f"Invalid input. `model_path`: {model_path} File or directory does not exist."
                )

        assert model_uid not in self._model_uid_to_model
        self._check_model_is_valid(model_name, model_format)

        if self.get_model_launch_status(model_uid) is not None:
            raise ValueError(f"{model_uid} is running")

        try:
            self._model_uid_launching_guard[model_uid] = launch_info = LaunchInfo()

            # virtualenv
            virtual_env_name = kwargs.pop("virtual_env_name", None)
            # Use v4 structure: .xinference/virtualenv/v4/model_name/model_engine/python_version
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            engine_name = (model_engine or "default").lower()
            virtual_env_path = os.path.join(
                XINFERENCE_VIRTUAL_ENV_DIR,
                "v4",
                model_name,
                engine_name,
                python_version,
            )
            virtual_env_manager = await asyncio.to_thread(
                self._create_virtual_env_manager,
                enable_virtual_env,
                virtual_env_name,
                virtual_env_path,
            )
            subpool_python_path = resolve_virtualenv_python_path(virtual_env_manager)
            subpool_envs = build_subpool_envs_for_virtual_env(
                envs, enable_virtual_env, virtual_env_manager
            )
            subpool_address, devices = await self._create_subpool(
                model_uid,
                model_type,
                n_gpu=n_gpu,
                gpu_idx=gpu_idx,
                start_python=subpool_python_path,
                env=subpool_envs,
            )
            all_subpool_addresses = [subpool_address]
            try:
                xavier_config: Optional[Dict] = kwargs.pop("xavier_config", None)
                if xavier_config is not None:
                    xavier_config["rank_address"] = subpool_address
                model_kwargs = kwargs.copy()
                model_kwargs["enable_virtual_env"] = enable_virtual_env
                if n_worker > 1:  # type: ignore
                    # for model across workers,
                    # add a few kwargs
                    model_kwargs.update(
                        dict(
                            address=subpool_address,
                            n_worker=n_worker,
                            shard=shard,
                            driver_info=driver_info,
                        )
                    )

                with CancellableDownloader(
                    cancelled_event=launch_info.cancel_event
                ) as downloader:
                    launch_info.downloader = downloader
                    progressor = await self._get_progressor("launching-" + model_uid)
                    # split into download and launch
                    progressor.split_stages(2, stage_weight=[0, 0.8, 1.0])
                    with progressor:
                        upload_progress_task = asyncio.create_task(
                            asyncio.to_thread(
                                self._upload_download_progress, progressor, downloader
                            )
                        )
                        model = await asyncio.to_thread(
                            create_model_instance,
                            model_uid,
                            model_type,
                            model_name,
                            model_engine,
                            model_format,
                            model_size_in_billions,
                            quantization,
                            peft_model_config,
                            download_hub,
                            model_path,
                            **model_kwargs,
                        )
                    model.model_family.address = subpool_address
                    model.model_family.accelerators = devices
                    model.model_family.multimodal_projector = model_kwargs.get(
                        "multimodal_projector", None
                    )
                    await self.update_cache_status(
                        model_name, model.model_family.to_version_info()
                    )

                def check_cancel():
                    # check downloader first, sometimes download finished
                    # cancelled already
                    if downloader.cancelled:
                        with progressor:
                            # just report progress
                            pass
                        downloader.raise_error(error_msg="Launch cancelled")

                # check cancel before prepare virtual env
                check_cancel()

                # install packages in virtual env
                if virtual_env_manager:
                    await asyncio.to_thread(
                        self._prepare_virtual_env,
                        virtual_env_manager,
                        model.model_family.virtualenv,
                        virtual_env_packages,
                        model_engine,
                    )
                    launch_info.virtual_env_manager = virtual_env_manager

                # check before creating model actor
                check_cancel()

                model_ref = await xo.create_actor(
                    ModelActor,
                    address=subpool_address,
                    uid=model_uid,
                    supervisor_address=self._supervisor_address,
                    worker_address=self.address,
                    replica_model_uid=model_uid,
                    model=model,
                    request_limits=request_limits,
                    xavier_config=xavier_config,
                    n_worker=n_worker,
                    shard=shard,
                    driver_info=driver_info,
                )
                if await model_ref.need_create_pools() and (
                    len(devices) > 1 or n_worker > 1  # type: ignore
                ):
                    coros = []
                    env_name = get_available_device_env_name() or "CUDA_VISIBLE_DEVICES"
                    env_value = ",".join(devices)
                    for device in devices:
                        coros.append(
                            self._main_pool.append_sub_pool(
                                env={env_name: env_value},
                                start_python=subpool_python_path,
                            )
                        )
                    pool_addresses = await asyncio.gather(*coros)
                    all_subpool_addresses.extend(pool_addresses)
                    await model_ref.set_pool_addresses(pool_addresses)

                # check before loading
                check_cancel()

                # set all subpool addresses
                # when cancelled, all subpool addresses need to be destroyed
                launch_info.sub_pools = all_subpool_addresses

                with progressor:
                    try:
                        await model_ref.load()
                    except xo.ServerClosed:
                        check_cancel()
                        raise
            except:
                logger.error(f"Failed to load model {model_uid}", exc_info=True)
                self.release_devices(model_uid=model_uid)
                for addr in all_subpool_addresses:
                    try:
                        await self._main_pool.remove_sub_pool(addr)
                    except KeyError:
                        continue
                raise
            self._model_uid_to_model[model_uid] = model_ref
            model_spec = model.model_family.to_description()
            self._model_uid_to_model_spec[model_uid] = model_spec
            self._model_uid_to_model_status[model_uid] = ModelStatus()
            self._model_uid_to_addr[model_uid] = subpool_address
            self._model_uid_to_recover_count.setdefault(
                model_uid, MODEL_ACTOR_AUTO_RECOVER_LIMIT
            )
            self._model_uid_to_launch_args[model_uid] = launch_args
        finally:
            del self._model_uid_launching_guard[model_uid]

        # Record virtual environment information if applicable
        if virtual_env_manager is not None and virtual_env_path is not None:
            try:
                # Get package information from virtual environment settings
                packages: List[str] = []
                package_info: Dict[str, Any] = {}
                if (
                    model_spec
                    and hasattr(model_spec, "virtualenv")
                    and model_spec.virtualenv
                ):
                    packages = model_spec.virtualenv.packages or []

                # Virtual environment tracking is no longer needed
                logger.info(
                    f"Virtual environment created for model: {model.model_family.model_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to handle virtual environment info: {e}")

        # update status to READY
        abilities = await self._get_model_ability(model, model_type)
        _ = await self.get_supervisor_ref(add_worker=False)

        if self._status_guard_ref is None:
            _ = await self.get_supervisor_ref()
        assert self._status_guard_ref is not None
        await self._status_guard_ref.update_instance_info(
            origin_uid,
            {"model_ability": abilities, "status": LaunchStatus.READY.name},
        )
        if n_worker > 1 and shard == 0:  # type: ignore
            return subpool_address, await model_ref.get_driver_info()
        else:
            return subpool_address

    @log_async(logger=logger, level=logging.INFO)
    async def wait_for_load(self, model_uid: str):
        model_ref = self._model_uid_to_model[model_uid]
        await model_ref.wait_for_load()

    @log_sync(logger=logger, level=logging.INFO)
    async def cancel_launch_model(self, model_uid: str):
        try:
            launch_info = self._model_uid_launching_guard[model_uid]

            # downloader shared same cancel event
            # sometimes cancel happens very early before downloader
            # even if users cancel at this time,
            # downloader will know and stop everything
            launch_info.cancel_event.set()

            if launch_info.downloader:
                logger.debug("Try to cancel download, %s")
                launch_info.downloader.cancel()
            if launch_info.virtual_env_manager:
                launch_info.virtual_env_manager.cancel_install()
            if launch_info.sub_pools:
                logger.debug("Try to stop sub pools: %s", launch_info.sub_pools)
                coros = []
                for addr in launch_info.sub_pools:
                    coros.append(self._main_pool.remove_sub_pool(addr, force=True))
                await asyncio.gather(*coros)
            if self._status_guard_ref is not None:
                await self._status_guard_ref.update_instance_info(
                    parse_replica_model_uid(model_uid)[0],
                    {"status": LaunchStatus.ERROR.name},
                )
        except KeyError:
            logger.error("Fail to cancel launching", exc_info=True)
            raise RuntimeError(
                "Model is not launching, may have launched or not launched yet"
            )

    @log_async(logger=logger, level=logging.INFO)
    async def terminate_model(self, model_uid: str, is_model_die=False):
        # Terminate model while its launching is not allow
        if model_uid in self._model_uid_launching_guard:
            raise ValueError(f"{model_uid} is launching")
        # In special cases, if the suffix is `-rank0`, this is the Xavier's rank 0 model actor.
        if model_uid.endswith("-rank0"):
            origin_uid = model_uid.removesuffix("-rank0")
        else:
            origin_uid, _ = parse_replica_model_uid(model_uid)
        try:
            _ = await self.get_supervisor_ref()
            if self._event_collector_ref is not None:
                await self._event_collector_ref.report_event(
                    origin_uid,
                    Event(
                        event_type=EventType.INFO,
                        event_ts=int(time.time()),
                        event_content="Terminate model",
                    ),
                )
        except Exception as e:
            # Report callback error can be log and ignore, should not interrupt the Process
            logger.error("report_event error: %s" % (e))

        if self._status_guard_ref is not None:
            await self._status_guard_ref.update_instance_info(
                origin_uid, {"status": LaunchStatus.TERMINATING.name}
            )
        model_ref = self._model_uid_to_model.get(model_uid, None)
        if model_ref is None:
            logger.debug("Model not found, uid: %s", model_uid)

        pool_addresses = None
        if model_ref is not None:
            try:
                # pool addresses if model.need_create_pools()
                pool_addresses = await model_ref.get_pool_addresses()
            except Exception as e:
                # process may disappear, we just ignore it.
                logger.debug("Fail to get pool addresses, error: %s", e)

        try:
            logger.debug("Start to destroy model actor: %s", model_ref)
            coro = xo.destroy_actor(model_ref)
            # see https://github.com/xorbitsai/xoscar/pull/140
            # asyncio.wait_for cannot work for Xoscar actor call,
            # because when time out, the coroutine will be cancelled via raise CancelledEror,
            # inside actor call, the error will be caught and
            # a CancelMessage will be sent to dest actor pool,
            # however the actor pool may be stuck already,
            # thus the timeout will never be raised
            await xo.wait_for(coro, timeout=5)
        except Exception as e:
            logger.debug(
                "Destroy model actor failed, model uid: %s, error: %s", model_uid, e
            )
        try:
            to_remove_addresses = []
            subpool_address = self._model_uid_to_addr[model_uid]
            to_remove_addresses.append(subpool_address)
            if pool_addresses:
                to_remove_addresses.extend(pool_addresses)
            logger.debug("Remove sub pools: %s", to_remove_addresses)
            coros = []
            for to_remove_addr in to_remove_addresses:
                coros.append(
                    self._main_pool.remove_sub_pool(to_remove_addr, force=True)
                )
            await asyncio.gather(*coros)
        except Exception as e:
            logger.debug(
                "Remove sub pool failed, model uid: %s, error: %s", model_uid, e
            )
        finally:
            # Clean up virtual environment tracking
            # Virtual environment tracking is no longer needed

            self._model_uid_to_model.pop(model_uid, None)
            self._model_uid_to_model_spec.pop(model_uid, None)
            self.release_devices(model_uid)
            self._model_uid_to_addr.pop(model_uid, None)
            self._model_uid_to_recover_count.pop(model_uid, None)
            self._model_uid_to_launch_args.pop(model_uid, None)

            if is_model_die:
                status = LaunchStatus.ERROR.name
            else:
                status = LaunchStatus.TERMINATED.name
                self._model_uid_to_model_status.pop(model_uid, None)

            if self._status_guard_ref is None:
                _ = await self.get_supervisor_ref()
            assert self._status_guard_ref is not None
            await self._status_guard_ref.update_instance_info(
                origin_uid, {"status": status}
            )

    # Provide an interface for future version of supervisor to call
    def get_model_launch_status(self, model_uid: str) -> Optional[str]:
        """
        returns:
            CREATING: model is launching
            RREADY: model is running
            None: model is not running (launch error might have happened)
        """

        if model_uid in self._model_uid_launching_guard:
            return LaunchStatus.CREATING.name
        if model_uid in self._model_uid_to_model:
            return LaunchStatus.READY.name
        return None

    @log_async(logger=logger)
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        return {k: v for k, v in self._model_uid_to_model_spec.items()}

    @log_sync(logger=logger)
    def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        model_status = self._model_uid_to_model_status.get(model_uid)
        if model_status and model_status.last_error:
            raise Exception(model_status.last_error)
        model_ref = self._model_uid_to_model.get(model_uid, None)
        if model_ref is None:
            raise ValueError(f"Model not found, uid: {model_uid}")
        return model_ref

    @log_sync(logger=logger)
    def describe_model(self, model_uid: str) -> Dict[str, Any]:
        model_desc = self._model_uid_to_model_spec.get(model_uid, None)
        if model_desc is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")
        return model_desc

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
        supervisor_ref = await self.get_supervisor_ref()
        await supervisor_ref.report_worker_status(self.address, status)

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
            real_path = get_real_path(path)
            if real_path:
                cached_model["real_path"] = real_path
            cached_model["actor_ip_address"] = self.address
            cached_models.append(cached_model)
        return cached_models

    async def list_deletable_models(self, model_version: str) -> List[str]:
        paths = set()
        path = await self._cache_tracker_ref.list_deletable_models(
            model_version, self.address
        )
        if not path:
            return []

        # Always keep the symlink itself so broken links can be unlinked.
        if os.path.islink(path):
            paths.add(path)

        if os.path.isfile(path):
            path = os.path.dirname(path)

        if os.path.isdir(path):
            paths.add(path)
            files = os.listdir(path)
            paths.update([os.path.join(path, file) for file in files])
            # search real path
            if paths:
                paths.update(
                    [
                        real_path
                        for path in paths
                        if os.path.exists((real_path := os.path.realpath(path)))
                    ]
                )

            # get tensorizer path
            from ..model.llm.transformers.tensorizer_utils import get_tensorizer_dir

            tensorizer_path = get_tensorizer_dir(path)
            if os.path.isdir(tensorizer_path):
                files = os.listdir(tensorizer_path)
                paths.update([os.path.join(tensorizer_path, file) for file in files])

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
                logger.error(f"Fail to delete {path} with error:{e}.")  # noqa: E231
                return False

        await self._cache_tracker_ref.confirm_and_remove_model(
            model_version, self.address
        )
        return True

    # Virtual environment management methods
    async def list_virtual_envs(
        self, model_name: Optional[str] = None, model_engine: Optional[str] = None
    ) -> List[Dict[Any, Any]]:
        """List all virtual environments or filter by model name."""
        try:
            result = self._virtual_env_manager.list_virtual_envs(
                model_name, model_engine
            )
            # Add IP address to each virtual environment, same as cache implementation
            virtual_envs = []
            for env in result:
                virtual_env = {
                    "model_name": env.get("model_name"),
                    "model_engine": env.get("model_engine"),
                    "path": env.get("path"),
                    "real_path": env.get("real_path"),
                    "python_version": env.get("python_version"),
                    "actor_ip_address": self.address,
                }
                virtual_envs.append(virtual_env)

            return virtual_envs
        except Exception as e:
            logger.error(f"Error in list_virtual_envs: {e}")
            raise

    async def list_virtual_env_packages(self, model_name: str) -> Dict[str, Any]:
        """List packages installed in a specific virtual environment."""
        return self._virtual_env_manager.list_virtual_env_packages(model_name)

    async def remove_virtual_env(
        self,
        model_name: str,
        model_engine: Optional[str] = None,
        python_version: Optional[str] = None,
    ) -> bool:
        """Remove a virtual environment for a specific model."""
        return self._virtual_env_manager.remove_virtual_env(
            model_name, model_engine, python_version
        )

    async def get_workers_info(self) -> Dict[str, Any]:
        ret = {
            "work-ip": self.address,
            "models": await self.list_models(),
        }
        return ret

    def update_model_status(self, model_uid: str, **kwargs):
        model_status = self._model_uid_to_model_status.get(model_uid)
        if model_status is not None:
            for k, v in kwargs.items():
                setattr(model_status, k, v)

    def get_model_status(self, model_uid: str):
        return self._model_uid_to_model_status.get(model_uid)

    @staticmethod
    def record_metrics(name, op, kwargs):
        record_metrics(name, op, kwargs)

    async def start_transfer_for_vllm(
        self, rep_model_uid: str, rank_addresses: List[str]
    ):
        model_ref = self._model_uid_to_model[rep_model_uid]
        await model_ref.start_transfer_for_vllm(rank_addresses)

    @log_async(logger=logger, level=logging.INFO)
    async def launch_rank0_model(
        self, rep_model_uid: str, xavier_config: Dict[str, Any]
    ) -> Tuple[str, int]:
        from ..model.llm.vllm.xavier.collective_manager import Rank0ModelActor

        subpool_address = await self._main_pool.append_sub_pool()

        store_address = subpool_address.split(":")[0]
        # Note that `store_port` needs to be generated on the worker,
        # as the TCP store is on rank 0, not on the supervisor.
        store_port = xo.utils.get_next_port()
        self._model_uid_launching_guard[rep_model_uid] = LaunchInfo()
        try:
            try:
                xavier_config["rank_address"] = subpool_address
                xavier_config["store_address"] = store_address
                xavier_config["store_port"] = store_port
                model_ref = await xo.create_actor(
                    Rank0ModelActor,
                    address=subpool_address,
                    uid=rep_model_uid,
                    xavier_config=xavier_config,
                )
            except:
                await self._main_pool.remove_sub_pool(subpool_address)
                raise
            self._model_uid_to_model[rep_model_uid] = model_ref
            self._model_uid_to_addr[rep_model_uid] = subpool_address
        finally:
            del self._model_uid_launching_guard[rep_model_uid]
        return subpool_address, store_port

    @no_type_check
    async def recover_model(self, launch_args: Dict[str, Any]):
        rep_model_uid = launch_args.get("model_uid")
        origin_uid, _ = parse_replica_model_uid(rep_model_uid)
        xavier_config: Optional[Dict[str, Any]] = launch_args.get("xavier_config", None)
        is_xavier: bool = xavier_config is not None
        supervisor_ref = await self.get_supervisor_ref(add_worker=False)
        if is_xavier:
            rank = xavier_config.get("rank")
            await supervisor_ref.call_collective_manager(
                origin_uid, "unregister_rank", rank
            )
        subpool_address = await self.launch_builtin_model(**launch_args)
        if is_xavier:
            model_ref = self._model_uid_to_model[rep_model_uid]
            await model_ref.start_transfer_for_vllm([])
            rank = xavier_config.get("rank")
            await supervisor_ref.call_collective_manager(
                origin_uid, "register_rank", rank, subpool_address, update=True
            )
