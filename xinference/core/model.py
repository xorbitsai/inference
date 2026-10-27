# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
import functools
import inspect
import json
import os
import queue
import threading
import time
import types
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Union,
    no_type_check,
)

import sse_starlette.sse
import xoscar as xo
from xoscar.api import IteratorWrapper

from ..constants import (
    XINFERENCE_DEFAULT_CANCEL_BLOCK_DURATION,
    XINFERENCE_LAUNCH_MODEL_RETRY,
)

if TYPE_CHECKING:
    from .progress_tracker import ProgressTrackerActor
    from .realtime.runtime import RealtimeRuntime
    from .worker import WorkerActor
    from ..model.llm.core import LLM
    from ..model.llm.weight_cache import WeightCachedModel
    import PIL

import logging

logger = logging.getLogger(__name__)

from ..device_utils import empty_cache
from .exceptions import InvalidModelRequestError, ModelNotReadyError
from .model_generation import ModelGenerationMixin
from .rpc_context import rpc_context
from .utils import CancelMixin, json_dumps, log_async, parse_replica_model_uid

try:
    from torch.cuda import OutOfMemoryError
except ImportError:

    class _OutOfMemoryError(Exception):
        pass

    OutOfMemoryError = _OutOfMemoryError


# !!!!! DO NOT add model_name to this list, using `register_batching_multimodal_models` below instead.
XINFERENCE_BATCHING_ALLOWED_VISION_MODELS = []

XINFERENCE_TEST_OUT_OF_MEMORY_ERROR = bool(
    os.getenv("XINFERENCE_TEST_OUT_OF_MEMORY_ERROR", False)
)


async def _await_owned_task(task: asyncio.Future):
    """Cancellation cannot abandon native work that still owns model resources."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        try:
            task.result()
        except BaseException:
            pass  # Consume native errors without replacing client cancellation.
        raise


class _RealtimeFileStream:
    def __init__(self, raw, request_id: Optional[str]):
        self.raw = raw
        self.request_id = request_id
        self.cancelled = threading.Event()
        self.done = asyncio.get_running_loop().create_future()


def register_batching_multimodal_models(*model_names: str):
    def decorator(cls):
        for name in model_names:
            if name not in XINFERENCE_BATCHING_ALLOWED_VISION_MODELS:
                XINFERENCE_BATCHING_ALLOWED_VISION_MODELS.append(name)
        return cls

    return decorator


def request_limit(fn):
    """
    Used by ModelActor.
    As a decorator, added to a ModelActor method to control
    how many requests are accessing that method at the same time.
    """

    async def wrapped_func(self, *args, **kwargs):
        if getattr(self, "_model_state", None) == "reloading":
            admission = self.__dict__.get("_reload_admission")
            if admission is None:
                raise ModelNotReadyError("Model is reloading")
            await admission.wait()
            self._require_ready()
        logger.debug(
            f"Request {fn.__name__}, current serve request count: {self._serve_count}, request limit: {self._request_limits} for the model {self.model_uid()}"
        )
        realtime = getattr(self, "_realtime_runtime", None)
        realtime_count = realtime.active_count if realtime is not None else 0
        if 1 + self._serve_count + realtime_count <= self._request_limits:
            self._serve_count += 1
        else:
            await self.record_metrics(
                "model_request_total",
                "add",
                {"labels": {**self._metrics_labels, "stream": "unknown"}, "value": 1},
            )
            await self.record_metrics(
                "model_request_errors_total",
                "add",
                {"labels": self._metrics_labels, "value": 1},
            )
            raise RuntimeError(
                f"Rate limit reached for the model. Request limit {self._request_limits} for the model: {self.model_uid()}"
            )
        start_time = time.time()
        ret = None
        _error = False
        released = False
        stream_transferred = False
        try:
            try:
                await self.record_metrics(
                    "model_serve_count",
                    "set",
                    {"labels": self._metrics_labels, "value": self._serve_count},
                )
                ret = await fn(self, *args, **kwargs)
            except Exception:
                _error = True
                raise
            finally:
                duration = time.time() - start_time
                _is_stream = ret is not None and (
                    inspect.isasyncgen(ret)
                    or inspect.isgenerator(ret)
                    or isinstance(ret, IteratorWrapper)
                )
                if not _is_stream:
                    self._serve_count = max(0, self._serve_count - 1)
                    released = True
                stream_label = "true" if _is_stream else "false"
                await self.record_metrics(
                    "model_request_total",
                    "add",
                    {
                        "labels": {**self._metrics_labels, "stream": stream_label},
                        "value": 1,
                    },
                )
                if not _is_stream:
                    await self.record_metrics(
                        "model_serve_count",
                        "set",
                        {"labels": self._metrics_labels, "value": self._serve_count},
                    )
                    logger.debug(
                        f"After request {fn.__name__}, current serve request count: {self._serve_count} for the model {self.model_uid()}"
                    )
                await self.record_metrics(
                    "model_request_duration_seconds",
                    "observe",
                    {"labels": self._metrics_labels, "value": duration},
                )
                if _error:
                    await self.record_metrics(
                        "model_request_errors_total",
                        "add",
                        {"labels": self._metrics_labels, "value": 1},
                    )
            stream_transferred = _is_stream
            return ret
        finally:
            if not released and not stream_transferred:
                self._serve_count = max(0, self._serve_count - 1)

    return wrapped_func


def oom_check(fn):
    @functools.wraps(fn)
    def _wrapper(self, *args, **kwargs):
        try:
            if XINFERENCE_TEST_OUT_OF_MEMORY_ERROR:
                raise OutOfMemoryError("Test Out of Memory Error")
            return fn(self, *args, **kwargs)
        except OutOfMemoryError as ex:
            assert self._loop is not None
            asyncio.run_coroutine_threadsafe(
                self._handle_oom_error(ex), loop=self._loop
            )

    @functools.wraps(fn)
    async def _async_wrapper(self, *args, **kwargs):
        try:
            if XINFERENCE_TEST_OUT_OF_MEMORY_ERROR:
                raise OutOfMemoryError("Test Out of Memory Error")
            return await fn(self, *args, **kwargs)
        except OutOfMemoryError as ex:
            await self._handle_oom_error(ex)

    assert not inspect.isasyncgen(fn)
    assert not inspect.isgenerator(fn)

    if asyncio.iscoroutinefunction(fn):
        return _async_wrapper
    else:
        return _wrapper


class ModelActor(ModelGenerationMixin, xo.StatelessActor, CancelMixin):
    _replica_model_uid: Optional[str]
    _realtime_runtime: Optional["RealtimeRuntime"]
    _model_state: str

    async def __pre_destroy__(self):
        await ModelActor._drain_realtime(self)
        await self.close_model_generations()
        for cache in self.__dict__.get("_generation_caches", {}).values():
            await asyncio.to_thread(cache.stop)
        from ..model.embedding.core import EmbeddingModel
        from ..model.llm.sglang.core import SGLANGModel
        from ..model.llm.transformers.core import PytorchModel as LLMPytorchModel
        from ..model.llm.vllm.core import VLLMModel as LLMVLLMModel
        from ..model.rerank.core import RerankModel

        await ModelActor._close_gpu_caches(self)

        if hasattr(self._model, "stop") and callable(self._model.stop):
            await asyncio.to_thread(self._model.stop)

        if isinstance(self._model, LLMVLLMModel):
            if self._transfer_ref is not None:
                try:
                    await xo.destroy_actor(self._transfer_ref)
                    del self._transfer_ref
                except Exception as e:
                    logger.debug(
                        f"Destroy transfer actor failed, address: {self.address}, error: {e}"
                    )

        # Free GPU memory on teardown for pytorch-backed models. Pytorch
        # embedding/rerank models are included here: they hold CUDA tensors
        # but define no stop()/close(), so without an explicit del +
        # empty_cache their VRAM is not released until the subpool process
        # exits. That leak leaves the GPU busy and can crash the next model's
        # subpool during CUDA init on relaunch (see issue #5156). The pytorch
        # format guard keeps the torch-free llama.cpp embedding/rerank paths
        # out of the torch-dependent branch below.
        if (
            isinstance(self._model, (LLMPytorchModel, LLMVLLMModel, SGLANGModel))
            and self._model.model_spec.model_format == "pytorch"
        ) or (
            isinstance(self._model, (EmbeddingModel, RerankModel))
            and self._model._model_spec.model_format == "pytorch"
        ):
            try:
                import gc

                import torch  # noqa: F401
            except ImportError:
                error_message = "Failed to import module 'torch'"
                installation_guide = [
                    "Please make sure 'torch' is installed.\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

            del self._model
            gc.collect()
            empty_cache()

    async def _drain_realtime(self):
        runtime = getattr(self, "_realtime_runtime", None)
        if runtime is not None:
            self._model_state = "draining"
            streams = list(self._realtime_file_streams)
            for stream in streams:
                stream.cancelled.set()
            await runtime.stop()
            if streams:
                await _await_owned_task(asyncio.gather(*(s.done for s in streams)))
            # File inference uses the same lock and may still own native work
            # after its client disconnects. Drain it before unloading weights.
            assert self._lock is not None
            async with self._lock:
                pass
            self._realtime_runtime = None

    def __init__(
        self,
        supervisor_address: str,
        worker_address: str,
        model: "LLM",
        replica_model_uid: str,
        request_limits: Optional[int] = None,
        xavier_config: Optional[Dict] = None,
        n_worker: Optional[int] = 1,
        shard: Optional[int] = 0,
        driver_info: Optional[dict] = None,  # for model across workers
        model_engine: Optional[str] = None,
        model_python_executable: Optional[str] = None,
    ):
        super().__init__()
        self._generation_python_executable = model_python_executable

        from ..model.llm.vllm.core import VLLMModel

        self._supervisor_address = supervisor_address
        self._worker_address = worker_address
        self._replica_model_uid = replica_model_uid
        self._model = model
        self._model_description = self._model.model_family.to_description()
        self._request_limits = (
            float("inf") if request_limits is None else request_limits
        )
        self._pending_requests: asyncio.Queue = asyncio.Queue()
        self._handle_pending_requests_task = None
        self._lock = (
            None if getattr(self._model, "allow_batch", False) else asyncio.locks.Lock()
        )
        self._worker_ref = None
        self._worker_generation = 0
        self._progress_tracker_ref = None
        self._serve_count = 0
        self._realtime_runtime = None
        self._realtime_file_streams: Set[_RealtimeFileStream] = set()
        self._realtime_start_lock = asyncio.Lock()
        model_type = self._model_description.get("model_type", "unknown")
        if model_type == "video":
            engine_label = model_engine or ""
            format_label = ""
        elif model_type in ("audio", "image", "world"):
            engine_label = model_engine or ""
            format_label = ""
        else:
            engine_label = model_engine or "unknown"
            format_label = self._model_description.get("model_format", "unknown")
        _base_uid, _rep_id = parse_replica_model_uid(self._replica_model_uid or "")
        self._metrics_labels = {
            "model_type": model_type,
            "model_uid": _base_uid,
            "replica_index": str(_rep_id),
            "model_name": self._model_description.get("model_name", "unknown"),
            "engine": engine_label,
            "worker_address": self._worker_address,
            "format": format_label,
            "quantization": self._model_description.get("quantization", "none"),
            "gpu_index": ",".join(
                str(a) for a in (self._model_description.get("accelerators") or [])
            ),
        }
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._model_state = "registering"
        # model across workers
        self._n_worker = n_worker
        self._shard = shard
        self._driver_info = driver_info

        if isinstance(self._model, VLLMModel):
            self._xavier_config = xavier_config
            self._model.set_xavier_config(xavier_config)
            self._transfer_ref = None

    async def __post_create__(self):
        self._loop = asyncio.get_running_loop()

        logger.debug("Starting ModelActor at %s, uid: %s", self.address, self.uid)

        self._handle_pending_requests_task = asyncio.create_task(
            self._handle_pending_requests()
        )

        # Report static request limit gauge
        limit_val = self._request_limits if self._request_limits != float("inf") else -1
        await self.record_metrics(
            "model_request_limit_gauge",
            "set",
            {"labels": self._metrics_labels, "value": limit_val},
        )

    def get_pid(self) -> int:
        import os

        return os.getpid()

    def _require_ready(self):
        """Guard for all inference methods: reject if not in ready state."""
        if self._model_state != "ready":
            if self._model_state in ("registering", "loading", "reloading"):
                raise ModelNotReadyError(
                    f"Model is {self._model_state}, not ready for inference"
                )
            raise RuntimeError(f"Model is in {self._model_state} state")

    def __repr__(self) -> str:
        return f"ModelActor({self._replica_model_uid})"

    def __getattr__(self, attr: str):
        return getattr(self._model, attr)

    @rpc_context
    async def decrease_serve_count(self):
        # API stream cleanup can arrive while a cancelled next() is still
        # executing. Do not release its request-limit slot before native drain.
        # This legacy RPC has no request ID, so conservatively join live leases.
        streams = list(self.__dict__.get("_realtime_file_streams", ()))
        if streams:
            await _await_owned_task(asyncio.gather(*(s.done for s in streams)))
        self._serve_count = max(0, self._serve_count - 1)
        await self.record_metrics(
            "model_serve_count",
            "set",
            {"labels": self._metrics_labels, "value": self._serve_count},
        )

    @no_type_check
    async def start_transfer_for_vllm(self, rank_addresses: List[str]):
        from ..model.llm.vllm.core import VLLMModel
        from ..model.llm.vllm.xavier.transfer import TransferActor

        assert isinstance(self._model, VLLMModel)
        rank = self._xavier_config.get("rank")  # type: ignore
        self._transfer_ref = await xo.create_actor(
            TransferActor,
            address=self.address,
            uid=f"{TransferActor.default_uid()}-{rank}",
            rank=rank,
            world_size=self._xavier_config.get("world_size"),  # type: ignore
            rank_address=self._xavier_config.get("rank_address"),  # type: ignore
            store_address=self._xavier_config.get("store_address"),  # type: ignore
            store_port=self._xavier_config.get("store_port"),  # type: ignore
            world_addresses=rank_addresses,
            cache_backend=self._xavier_config.get("cache_backend"),
            freeze_initialization_gc=(
                self._model._is_vllm_v1()
                and self._xavier_config.get("gpu_cache_bytes") is None
            ),
        )
        await self._model.init_xavier()
        logger.debug(
            f"Init transfer actor: {self._transfer_ref.address}, rank: {rank} done for vllm."  # type: ignore
        )

    async def _record_completion_metrics(
        self, duration, completion_tokens, prompt_tokens
    ):
        coros = []
        if completion_tokens > 0:
            coros.append(
                self.record_metrics(
                    "output_tokens_total_counter",
                    "add",
                    {
                        "labels": self._metrics_labels,
                        "value": completion_tokens,
                    },
                )
            )
        if prompt_tokens > 0:
            coros.append(
                self.record_metrics(
                    "input_tokens_total_counter",
                    "add",
                    {"labels": self._metrics_labels, "value": prompt_tokens},
                )
            )
        if completion_tokens > 0:
            coros.append(
                self.record_metrics(
                    "generate_tokens_total",
                    "add",
                    {
                        "labels": self._metrics_labels,
                        "value": completion_tokens,
                    },
                )
            )
        await asyncio.gather(*coros)

    def get_worker_binding(self):
        binding = {"protocol": 1, "address": self._worker_address}
        if self.__dict__.get("_serving_generation") is not None or self.__dict__.get(
            "_retiring_generations"
        ):
            binding["generation_routing"] = 1
        return binding

    def rebind_worker(self, expected_address: str, address: str, generation: int):
        """Replace the control-plane binding without touching the model runtime."""
        current = self._worker_generation
        if generation >= current and self._worker_address == address:
            # A process recovered by the new owner is already bound here.
            if generation != current:
                self._worker_ref = None
            self._worker_generation = generation
            return True
        if self._worker_address != expected_address or generation <= current:
            raise RuntimeError("Stale model worker binding")
        self._worker_address = address
        self._worker_generation = generation
        self._worker_ref = None
        self._metrics_labels["worker_address"] = address
        return True

    async def _get_worker_ref(self) -> xo.ActorRefType["WorkerActor"]:
        from .worker import WorkerActor

        while self._worker_ref is None:
            address = self._worker_address
            generation = self._worker_generation
            ref = await xo.actor_ref(address=address, uid=WorkerActor.default_uid())
            if address != self._worker_address or generation != self._worker_generation:
                continue
            self._worker_ref = ref
        return self._worker_ref

    async def _get_progress_tracker_ref(
        self,
    ) -> xo.ActorRefType["ProgressTrackerActor"]:
        from .progress_tracker import ProgressTrackerActor

        if self._progress_tracker_ref is None:
            self._progress_tracker_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=ProgressTrackerActor.default_uid()
            )
        return self._progress_tracker_ref

    async def _get_progressor(self, request_id: Optional[str]):
        from .progress_tracker import Progressor

        progressor = Progressor(
            request_id,
            await self._get_progress_tracker_ref(),
            asyncio.get_running_loop(),
        )
        await progressor.start()
        return progressor

    def is_vllm_backend(self) -> bool:
        from ..model.llm.vllm.core import VLLMModel

        return isinstance(self._model, VLLMModel)

    def is_transformers_batching_enabled(self) -> bool:
        return bool(getattr(self._model, "_batch_scheduler", None))

    def is_sglang_backend(self) -> bool:
        from ..model.llm.sglang.core import SGLANGModel

        return isinstance(self._model, SGLANGModel)

    def get_limit_mm_per_prompt(self) -> Optional[Dict[str, int]]:
        from ..model.llm.vllm.core import VLLMModel

        if not isinstance(self._model, VLLMModel):
            return None
        return self._model.get_limit_mm_per_prompt()

    def get_sglang_pd_bootstrap(self) -> dict:
        from ..model.llm.sglang.core import SGLANGModel

        if not isinstance(self._model, SGLANGModel):
            raise ValueError("SGLang P/D bootstrap requires a SGLang replica")
        return self._model.get_pd_bootstrap()

    @rpc_context
    async def release_trt_pd_request(self, request_id: str) -> None:
        from ..model.llm.trt_llm.core import TRTLLMModel

        if not isinstance(self._model, TRTLLMModel):
            raise ValueError("TRT-LLM P/D release requires a TRT-LLM replica")
        await self._model.release_pd_request(request_id)

    async def load(self):
        self._model_state = "loading"
        try:
            # Change process title for model
            import setproctitle

            setproctitle.setproctitle(f"Model: {self._replica_model_uid}")
        except ImportError:
            pass
        i = 0
        while True:
            i += 1
            try:
                if hasattr(self._model, "set_loop"):
                    self._model.set_loop(asyncio.get_running_loop())
                await asyncio.to_thread(self._load_with_redirect)
                if hasattr(self._model, "driver_info"):
                    self._driver_info = self._model.driver_info
                break
            except Exception as e:
                if (
                    i < XINFERENCE_LAUNCH_MODEL_RETRY
                    and str(e).find("busy or unavailable") >= 0
                ):
                    await asyncio.sleep(5)
                    logger.warning(
                        "Retry to load model %s: %d times", self._replica_model_uid, i
                    )
                    continue
                raise
        logger.info(f"{self} loaded")

    def _load_with_redirect(self):
        from ..deploy.utils import redirect_streams_to_logger

        with redirect_streams_to_logger():
            self._model.load()

    async def wait_for_load(self):
        if hasattr(self._model, "wait_for_load"):
            await asyncio.to_thread(self._model.wait_for_load)
        await self._start_realtime_runtime()
        self._model_state = "ready"

    async def _start_realtime_runtime(self):
        async with self._realtime_start_lock:
            await self._start_realtime_runtime_unlocked()

    async def _start_realtime_runtime_unlocked(self):
        if self._realtime_runtime is not None:
            return
        capabilities = getattr(self._model, "get_realtime_capabilities", None)
        if capabilities is None or not capabilities().get("tasks"):
            return
        if self._lock is None:
            raise ValueError("Realtime adapters require serialized inference")
        from urllib.parse import urlsplit

        from .realtime.protocol import RealtimeLimits
        from .realtime.runtime import RealtimeRuntime

        base_uid, _ = parse_replica_model_uid(self._replica_model_uid or "")
        max_sessions = int(os.environ.get("XINFERENCE_REALTIME_MAX_SESSIONS", "1"))
        if not 1 <= max_sessions <= 128:
            raise ValueError("XINFERENCE_REALTIME_MAX_SESSIONS must be in [1, 128]")
        runtime = RealtimeRuntime(
            self._model,
            base_uid,
            self._replica_model_uid or "",
            limits=RealtimeLimits(max_sessions=max_sessions),
            inference_lock=self._lock,
            admission=lambda: self._model_state == "ready"
            and self._serve_count + runtime.active_count < self._request_limits,
        )
        worker_host = urlsplit(
            self._worker_address
            if "://" in self._worker_address
            else "tcp://" + self._worker_address
        ).hostname
        host = os.environ.get(
            "XINFERENCE_REALTIME_BIND_HOST", worker_host or "127.0.0.1"
        )
        advertised = os.environ.get(
            "XINFERENCE_REALTIME_ADVERTISE_HOST", worker_host or host
        )
        if advertised in ("0.0.0.0", "::"):
            raise ValueError("Configure a routable XINFERENCE_REALTIME_ADVERTISE_HOST")
        await runtime.start(host, advertised)
        self._realtime_runtime = runtime

    def get_realtime_endpoint(self, subject: str, task: str) -> Dict[str, Any]:
        """Connect-time discovery only; never append/poll media over Actor RPC."""
        self._require_ready()
        from .realtime.protocol import RealtimeError

        if self._realtime_runtime is None:
            raise RealtimeError(
                "unsupported_task", "Model has no realtime input adapter"
            )
        return self._realtime_runtime.issue_ticket(subject, task)

    def get_realtime_status(self) -> Dict[str, Any]:
        """Low-frequency control-plane observation, not result polling."""
        runtime = self._realtime_runtime
        if runtime is None:
            return {"readiness": False, "active_sessions": 0}
        return {
            "readiness": bool(runtime.endpoint) and not runtime.draining,
            "runtime_generation": runtime.generation,
            "active_sessions": runtime.active_count,
            "completed_sessions": runtime.completed_count,
            "failed_sessions": runtime.failed_count,
            "input_queue_peak": runtime.input_queue_peak,
        }

    def _get_weight_cached_model(self) -> "WeightCachedModel":
        from ..model.llm.weight_cache import WeightCachedModel

        if not isinstance(self._model, WeightCachedModel):
            raise ValueError("Reload with retained weights requires vLLM or SGLang")
        return self._model

    def get_reload_config(self) -> Dict[str, Any]:
        import sys
        from importlib.metadata import PackageNotFoundError, version

        config = self._get_weight_cached_model().get_reload_config()
        config["python_executable"] = sys.executable
        try:
            config["engine_version"] = version(config["engine"])
        except PackageNotFoundError:
            config["engine_version"] = None
        return config

    async def validate_reload(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        self._require_ready()
        config = await asyncio.to_thread(
            self._get_weight_cached_model().validate_reload, model_config
        )
        self._reload_status: Dict[str, Any] = {}
        return {key: config[key] for key in model_config}

    def get_reload_status(self) -> Dict[str, Any]:
        return dict(getattr(self, "_reload_status", {}))

    async def reload(self, model_config: Dict[str, Any], drain_timeout: float) -> bool:
        from ..model.llm.weight_cache import ModelReloadError

        await self.validate_reload(model_config)
        model = self._get_weight_cached_model()
        self._model_state = "reloading"
        self._reload_admission = asyncio.Event()
        self._reload_status = {"stage": "draining"}
        try:
            deadline = time.monotonic() + drain_timeout
            while self._serve_count:
                if time.monotonic() >= deadline:
                    raise ModelReloadError(
                        "Timed out draining requests; the original engine is still running",
                        restored=True,
                    )
                await asyncio.sleep(0.05)
            await asyncio.to_thread(
                model.reload,
                model_config,
                lambda stage: self._reload_status.update(stage=stage),
            )
        except ModelReloadError as exc:
            self._model_state = "ready" if exc.restored else "error"
            self._reload_status.update(stage="error", restored=exc.restored)
            raise
        except Exception:
            # Validation inside reload happens before engine teardown.
            self._model_state = "ready"
            self._reload_status.update(stage="error", restored=True)
            raise
        else:
            self._model_state = "ready"
            self._reload_status.update(stage="ready")
            return not model._is_hot_patch(model_config)
        finally:
            self._reload_admission.set()

    def need_create_pools(self):
        return getattr(self._model, "need_create_pools", False)

    def set_pool_addresses(self, pool_addresses: List[str]):
        if hasattr(self._model, "set_pool_addresses"):
            self._model.set_pool_addresses(pool_addresses)

    def get_pool_addresses(self) -> Optional[List[str]]:
        if hasattr(self._model, "get_pool_addresses"):
            return self._model.get_pool_addresses()
        return None

    def set_worker_addresses(self, shard: int, worker_addresses: List[str]):
        if hasattr(self._model, "set_worker_addresses"):
            self._model.set_worker_addresses(shard, worker_addresses)

    def model_uid(self):
        return (
            self._model.model_uid
            if hasattr(self._model, "model_uid")
            else (
                self._model._model_uid
                if hasattr(self._model, "_model_uid")
                else None  # return None for UT
            )
        )

    def get_driver_info(self):
        # driver info is used for model across workers,
        # the driver model actor(always the first worker)
        # will hold driver information includes dist store etc.
        return self._driver_info

    async def _close_gpu_caches(self):
        config = getattr(self, "_xavier_config", None)
        if config is None or config.get("gpu_cache_bytes") is None:
            return
        from ..model.llm.vllm.core import VLLMModel as LLMVLLMModel

        if isinstance(self._model, LLMVLLMModel) and self._transfer_ref is not None:
            # Release imported CUDA IPC handles while EngineCore still owns the
            # allocations, after any in-flight transfer has completed.
            try:
                await self._transfer_ref.close_gpu_caches_v1()
            except Exception:
                logger.warning(
                    "Failed to close Xavier GPU caches; continuing model cleanup",
                    exc_info=True,
                )

    async def stop(self):
        self._model_state = "stopping"
        # WorkerActor calls stop() before destroy_actor(), so teardown alone is
        # too late to protect resources released by the model's stop()/close().
        await ModelActor._drain_realtime(self)
        self._model_state = "stopping"
        await self.close_model_generations()
        await ModelActor._close_gpu_caches(self)
        if hasattr(self._model, "stop"):
            await asyncio.to_thread(self._model.stop)
        elif hasattr(self._model, "close"):
            await asyncio.to_thread(self._model.close)

    async def _handle_oom_error(self, ex):
        # Reentrancy gate: on the sync batch path the OOM-triggering worker
        # thread does not await this coroutine, so before os._exit(1) actually
        # lands it can raise OOM again and concurrently schedule this handler.
        # Only let the first invocation through to avoid re-entering stop()/del.
        if getattr(self, "_oom_handling", False):
            return
        self._oom_handling = True

        error_message = (
            f"Model actor is out of memory, model id: {self.model_uid()}, error: {ex}"
        )
        logger.exception(error_message)

        async def _report_oom_status():
            worker_ref = await self._get_worker_ref()
            await worker_ref.update_model_status(
                self._replica_model_uid, last_error=error_message
            )

        try:
            await xo.wait_for(_report_oom_status(), timeout=5)
        except Exception:
            logger.warning("Failed to report OOM status before exit", exc_info=True)

        # Best-effort cleanup. At an OOM site this mostly cannot reclaim already
        # reserved GPU fragments; failures are fine because os._exit(1) below is
        # the real backstop. stop() timeout is 5s (not 30s): keep the exit delay
        # bounded so xoscar's detection and recovery are not slowed down.
        try:
            await asyncio.wait_for(self.stop(), timeout=5)
        except Exception:
            logger.warning(
                "Model stop() failed/timeout during OOM cleanup", exc_info=True
            )
        try:
            del self._model
        except AttributeError:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Critical fix: must use os._exit(1), NOT sys.exit(1).
        # sys.exit(1) only raises SystemExit, which xoscar's
        # _ErrorProcessor.__exit__ (backends/pool.py) swallows with an
        # unconditional `return True`. The process then stays alive, its
        # returncode stays None, monitor_sub_pools keeps treating it as alive,
        # and recover_sub_pool never fires. os._exit goes straight to the
        # syscall so xoscar can detect the returncode change and rebuild the
        # sub pool.
        logger.error(
            "Exiting model subprocess via os._exit(1) to trigger pool recovery"
        )
        os._exit(1)

    def _to_generator(self, output_type: str, gen: types.GeneratorType):
        start_time = time.time()
        time_to_first_token = None
        final_usage = None
        try:
            if XINFERENCE_TEST_OUT_OF_MEMORY_ERROR:
                raise OutOfMemoryError("Test Out of Memory Error")
            for v in gen:
                if time_to_first_token is None:
                    time_to_first_token = (time.time() - start_time) * 1000
                if output_type == "json":
                    final_usage = v.get("usage", None)
                    v = dict(data=json.dumps(v, ensure_ascii=False))
                else:
                    assert (
                        output_type == "binary"
                    ), f"Unknown output type '{output_type}'"
                yield sse_starlette.sse.ensure_bytes(v, None)
        except OutOfMemoryError as ex:
            assert self._loop is not None
            asyncio.run_coroutine_threadsafe(
                self._handle_oom_error(ex), loop=self._loop
            )
        finally:
            if self._loop is not None and time_to_first_token is not None:
                coro = self.record_metrics(
                    "time_to_first_token_seconds",
                    "observe",
                    {
                        "labels": {**self._metrics_labels, "stream": "true"},
                        "value": time_to_first_token / 1000,
                    },
                )
                asyncio.run_coroutine_threadsafe(coro, loop=self._loop)
            if self._loop is not None and final_usage is not None:
                coro = self._record_completion_metrics(
                    time.time() - start_time,
                    completion_tokens=final_usage["completion_tokens"],
                    prompt_tokens=final_usage["prompt_tokens"],
                )
                asyncio.run_coroutine_threadsafe(coro, loop=self._loop)

    async def _to_async_gen(self, output_type: str, gen: types.AsyncGeneratorType):
        start_time = time.time()
        time_to_first_token = None
        final_usage = None
        try:
            if XINFERENCE_TEST_OUT_OF_MEMORY_ERROR:
                raise OutOfMemoryError("Test Out of Memory Error")
            async for v in gen:
                if time_to_first_token is None:
                    time_to_first_token = (time.time() - start_time) * 1000
                final_usage = v.get("usage", None)
                if output_type == "json":
                    v = await asyncio.to_thread(json.dumps, v, ensure_ascii=False)
                    v = dict(data=v)  # noqa: F821
                else:
                    assert (
                        output_type == "binary"
                    ), f"Unknown output type '{output_type}'"
                yield await asyncio.to_thread(sse_starlette.sse.ensure_bytes, v, None)
        except OutOfMemoryError as ex:
            await self._handle_oom_error(ex)
        finally:
            coros = []
            if time_to_first_token is not None:
                coros.append(
                    self.record_metrics(
                        "time_to_first_token_seconds",
                        "observe",
                        {
                            "labels": {**self._metrics_labels, "stream": "true"},
                            "value": time_to_first_token / 1000,
                        },
                    )
                )
            if final_usage is not None:
                coros.append(
                    self._record_completion_metrics(
                        time.time() - start_time,
                        completion_tokens=final_usage["completion_tokens"],
                        prompt_tokens=final_usage["prompt_tokens"],
                    )
                )
            try:
                await asyncio.gather(*coros)
            except Exception:
                logger.exception("Failed to record metrics in _to_async_gen finally")

    async def _handle_pending_requests(self):
        logger.info("Start requests handler.")
        while True:
            pending = await self._pending_requests.get()
            gen, stream_out, stop = pending[:3]

            async def _async_wrapper(_gen):
                try:
                    # anext is only available for Python >= 3.10
                    return await _gen.__anext__()  # noqa: F821
                except StopAsyncIteration:
                    return stop
                except Exception as e:
                    return e

            def _wrapper(_gen):
                # Avoid issue: https://github.com/python/cpython/issues/112182
                try:
                    return next(_gen)
                except StopIteration:
                    return stop
                except Exception as e:
                    return e

            if len(pending) == 4:
                # For realtime-enabled adapters, lock ownership was transferred
                # from generator creation to this pending stream, not released.
                lease = pending[3]
                try:
                    while not lease.cancelled.is_set():
                        if inspect.isgenerator(gen):
                            r = await _await_owned_task(
                                asyncio.create_task(asyncio.to_thread(_wrapper, gen))
                            )
                        else:
                            r = await _await_owned_task(
                                asyncio.create_task(_async_wrapper(gen))
                            )
                        if lease.cancelled.is_set():
                            break
                        if r is stop:
                            break
                        stream_out.put_nowait(r)
                        if isinstance(r, Exception):
                            break
                finally:

                    async def close_stream():
                        try:
                            # Close both the metrics/SSE wrapper and the raw
                            # generator under the lease, including native cleanup.
                            if inspect.isgenerator(gen):
                                try:
                                    await asyncio.to_thread(gen.close)
                                finally:
                                    await asyncio.to_thread(lease.raw.close)
                            else:
                                try:
                                    await gen.aclose()
                                finally:
                                    await lease.raw.aclose()
                        finally:
                            stream_out.put_nowait(stop)
                            assert self._lock is not None
                            self._lock.release()
                            self._realtime_file_streams.discard(lease)
                            lease.done.set_result(None)

                    await _await_owned_task(asyncio.create_task(close_stream()))
                continue

            while True:
                try:
                    if inspect.isgenerator(gen):
                        r = await asyncio.to_thread(_wrapper, gen)
                    elif inspect.isasyncgen(gen):
                        r = await _async_wrapper(gen)
                    else:
                        raise Exception(
                            f"The generator {gen} should be a generator or an async generator, "
                            f"but a {type(gen)} is got."
                        )
                    stream_out.put_nowait(r)
                    if r is not stop:
                        continue
                except Exception:
                    logger.exception("stream encountered an error.")
                break

    def _queue_realtime_file_stream(self, output_type: str, raw, request_id):
        lease = _RealtimeFileStream(raw, request_id)
        stop = object()
        stream_out: Union[queue.Queue, asyncio.Queue]
        if inspect.isgenerator(raw):
            gen = self._to_generator(output_type, raw)
            stream_out = queue.Queue()

            def outputs():
                try:
                    while True:
                        item = stream_out.get()
                        if item is stop:
                            return
                        if isinstance(item, Exception):
                            raise item
                        yield item
                finally:
                    lease.cancelled.set()

            result = outputs()
        else:
            gen = self._to_async_gen(output_type, raw)
            stream_out = asyncio.Queue()

            async def async_outputs():
                try:
                    while True:
                        item = await stream_out.get()
                        if item is stop:
                            return
                        if isinstance(item, Exception):
                            raise item
                        yield item
                finally:
                    lease.cancelled.set()

            result = async_outputs()
        self._realtime_file_streams.add(lease)
        self._pending_requests.put_nowait((gen, stream_out, stop, lease))
        return result

    async def _call_wrapper_json(self, fn: Callable, *args, **kwargs):
        return await self._call_wrapper("json", fn, *args, **kwargs)

    async def _call_wrapper_binary(self, fn: Callable, *args, **kwargs):
        return await self._call_wrapper("binary", fn, *args, **kwargs)

    @oom_check
    async def _call_wrapper(self, output_type: str, fn: Callable, *args, **kwargs):
        wait_for_sync_on_cancel = kwargs.pop("_wait_for_sync_on_cancel", False) or (
            getattr(self, "_realtime_runtime", None) is not None
        )

        async def _invoke():
            if inspect.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            if not wait_for_sync_on_cancel:
                return await asyncio.to_thread(fn, *args, **kwargs)

            # World generation runs a synchronous subprocess adapter.  Keep the
            # actor task (and therefore its request-limit slot) alive until the
            # worker thread has observed cancellation and reaped that process.
            thread_task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
            return await _await_owned_task(thread_task)

        self._add_running_task(kwargs.get("request_id"))
        if getattr(self, "_realtime_runtime", None) is not None:
            assert self._lock is not None
            await self._lock.acquire()
            transferred = False
            try:
                self._require_ready()
                ret = await _invoke()
                if inspect.isgenerator(ret) or inspect.isasyncgen(ret):
                    result = self._queue_realtime_file_stream(
                        output_type, ret, kwargs.get("request_id")
                    )
                    transferred = True
                    return result
            finally:
                if not transferred:
                    self._lock.release()
        elif self._lock is None:
            ret = await _invoke()

            if inspect.isgenerator(ret):
                gen = self._to_generator(output_type, ret)
                return gen
            if inspect.isasyncgen(ret):
                gen = self._to_async_gen(output_type, ret)
                return gen
        else:
            async with self._lock:
                ret = await _invoke()

                stream_out: Union[queue.Queue, asyncio.Queue]

                if inspect.isgenerator(ret):
                    gen = self._to_generator(output_type, ret)
                    stream_out = queue.Queue()
                    stop = object()
                    self._pending_requests.put_nowait((gen, stream_out, stop))

                    def _stream_out_generator():
                        while True:
                            o = stream_out.get()
                            if o is stop:
                                break
                            elif isinstance(o, Exception):
                                raise o
                            else:
                                yield o

                    return _stream_out_generator()

                if inspect.isasyncgen(ret):
                    gen = self._to_async_gen(output_type, ret)
                    stream_out = asyncio.Queue()
                    stop = object()
                    self._pending_requests.put_nowait((gen, stream_out, stop))

                    async def _stream_out_async_gen():
                        while True:
                            o = await stream_out.get()
                            if o is stop:
                                break
                            elif isinstance(o, Exception):
                                raise o
                            else:
                                yield o

                    return _stream_out_async_gen()

        if output_type == "json":
            return await asyncio.to_thread(json_dumps, ret)
        else:
            assert output_type == "binary", f"Unknown output type '{output_type}'"
            return ret

    def _validate_cache_salt(self, args, kwargs) -> None:
        from ..cache_isolation import validate_cache_salt

        config = args[0] if args else kwargs.get("generate_config")
        if (
            isinstance(config, dict)
            and validate_cache_salt(config.get("cache_salt")) is not None
        ):
            if not getattr(self._model, "supports_cache_salt", False):
                raise ValueError("cache_salt is not supported by this model engine")

    def _get_async_request_id(self, args: tuple, kwargs: dict) -> str:
        if "request_id" in kwargs:
            return str(kwargs["request_id"])
        from ..model.llm.trt_llm.core import TRTLLMModel

        if isinstance(self._model, TRTLLMModel):
            # REST keeps TRT's public ID in the generation config. Promote it
            # before _call_wrapper registers the actor task for explicit abort.
            config = args[0] if args else kwargs.get("generate_config")
            if isinstance(config, dict) and config.get("request_id") is not None:
                return str(config["request_id"])
        return str(uuid.uuid1())

    @request_limit
    @xo.generator
    @log_async(logger=logger)
    async def generate(self, prompt: str, *args, **kwargs):
        self._require_ready()
        self._validate_cache_salt(args, kwargs)
        start_time = time.time()
        # Directly delegate to model, let model decide how to handle (batching or not)
        kwargs.pop("raw_params", None)
        if hasattr(self._model, "generate"):
            request_id = kwargs.pop("request_id", None)
            if request_id is not None and getattr(
                self._model, "_batch_scheduler", None
            ):
                # Batched Transformers models read request IDs from the generation
                # config. Keep the actor-only keyword away from the model method,
                # whose signature does not accept arbitrary keyword arguments.
                generate_config = args[0] if args else kwargs.get("generate_config")
                generate_config = dict(generate_config or {})
                generate_config["request_id"] = str(request_id)
                if args:
                    args = (generate_config, *args[1:])
                else:
                    kwargs["generate_config"] = generate_config
            response = await self._call_wrapper_json(
                self._model.generate, prompt, *args, **kwargs
            )
            await self._record_non_stream_completion_metrics(response, start_time)
            return response
        if hasattr(self._model, "async_generate"):
            kwargs["request_id"] = self._get_async_request_id(args, kwargs)
            response = await self._call_wrapper_json(
                self._model.async_generate,
                prompt,
                *args,
                **kwargs,
            )
            await self._record_non_stream_completion_metrics(response, start_time)
            return response
        raise AttributeError(f"Model {self._model.model_spec} is not for generate.")

    @request_limit
    @xo.generator
    @log_async(logger=logger)
    async def chat(self, messages: List[Dict], *args, **kwargs):
        self._require_ready()
        self._validate_cache_salt(args, kwargs)
        start_time = time.time()
        response = None
        try:
            # Directly delegate to model, let model decide how to handle (batching or not)
            kwargs.pop("raw_params", None)
            if hasattr(self._model, "chat"):
                request_id = kwargs.pop("request_id", None)
                if request_id is not None and getattr(
                    self._model, "_batch_scheduler", None
                ):
                    # Match generate(): batching reads IDs from a copied config.
                    generate_config = args[0] if args else kwargs.get("generate_config")
                    generate_config = dict(generate_config or {})
                    generate_config["request_id"] = str(request_id)
                    if args:
                        args = (generate_config, *args[1:])
                    else:
                        kwargs["generate_config"] = generate_config
                response = await self._call_wrapper_json(
                    self._model.chat, messages, *args, **kwargs
                )
                return response
            if hasattr(self._model, "async_chat"):
                kwargs["request_id"] = self._get_async_request_id(args, kwargs)
                response = await self._call_wrapper_json(
                    self._model.async_chat, messages, *args, **kwargs
                )
                return response
            raise AttributeError(f"Model {self._model.model_spec} is not for chat.")
        finally:
            await self._record_non_stream_completion_metrics(response, start_time)

    async def _record_non_stream_completion_metrics(
        self, response: Any, start_time: float
    ) -> None:
        # For the non stream result.
        record = None
        if isinstance(response, Generator) or isinstance(response, AsyncGenerator):
            record = response
        elif isinstance(response, bytes):
            record = json.loads(response)
        if record and isinstance(record, dict):
            usage = record.get("usage")
            # Some backends may not have a valid usage, we just skip them.
            if isinstance(usage, dict):
                completion_tokens = usage.get("completion_tokens")
                prompt_tokens = usage.get("prompt_tokens")
                if completion_tokens is not None and prompt_tokens is not None:
                    await self._record_completion_metrics(
                        time.time() - start_time,
                        completion_tokens,
                        prompt_tokens,
                    )
            await self.record_metrics(
                "time_to_first_token_seconds",
                "observe",
                {
                    "labels": {**self._metrics_labels, "stream": "false"},
                    "value": time.time() - start_time,
                },
            )

    @log_async(logger=logger)
    async def abort_request(
        self,
        request_id: str,
        block_duration: int = XINFERENCE_DEFAULT_CANCEL_BLOCK_DURATION,
    ) -> str:
        from ..model.scheduler.core import AbortRequestMessage

        stream_cancelled = False
        for stream in self.__dict__.get("_realtime_file_streams", ()):
            if stream.request_id == request_id:
                stream.cancelled.set()
                stream_cancelled = True

        generation_results = []
        active_error = None
        active = self.__dict__.get("_serving_generation")
        for ref in [
            active,
            *self.__dict__.get("_retiring_generations", []),
        ]:
            if ref is not None:
                try:
                    generation_results.append(
                        await ref.abort_request(request_id, block_duration)
                    )
                except Exception as error:
                    if ref == active:
                        active_error = error
                    logger.debug("Generation could not abort request", exc_info=True)
        generation_result = next(
            (
                value
                for value in generation_results
                if value != AbortRequestMessage.NO_OP.name
            ),
            AbortRequestMessage.NO_OP.name,
        )
        if (generation_results or active_error is not None) and not self._serve_count:
            if (
                active_error is not None
                and generation_result == AbortRequestMessage.NO_OP.name
            ):
                raise active_error
            return generation_result

        # Always cancel the running task first
        task = self._running_tasks.get(request_id)
        cancelled = stream_cancelled or (
            task is not None
            and not task.done()
            and task.get_name() != self._CANCEL_TASK_NAME
        )
        self._cancel_running_task(request_id, block_duration)

        # If model has abort_request method, delegate to it
        if hasattr(self._model, "abort_request"):
            result = await self._model.abort_request(request_id)
            if (
                result is not None
                and (not cancelled or result != AbortRequestMessage.NO_OP.name)
                and (
                    result != AbortRequestMessage.NO_OP.name
                    or generation_result == AbortRequestMessage.NO_OP.name
                )
                and (result != AbortRequestMessage.NO_OP.name or active_error is None)
            ):
                return result

        if cancelled or stream_cancelled:
            return AbortRequestMessage.DONE.name

        # Otherwise return NO_OP for legacy models or when model doesn't handle abort
        if (
            active_error is not None
            and generation_result == AbortRequestMessage.NO_OP.name
        ):
            raise active_error
        return generation_result

    @request_limit
    @log_async(logger=logger)
    async def create_embedding(self, input: Union[str, List[str]], *args, **kwargs):
        self._require_ready()
        kwargs.pop("request_id", None)
        if hasattr(self._model, "create_embedding"):
            return await self._call_wrapper_json(
                self._model.create_embedding, input, *args, **kwargs
            )

        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating embedding."
        )

    @request_limit
    @log_async(logger=logger, ignore_kwargs=["audio"])
    async def create_audio_embedding(self, audio: bytes, *args, **kwargs):
        self._require_ready()
        kwargs.pop("request_id", None)
        if hasattr(self._model, "create_embedding"):
            return await self._call_wrapper_json(
                self._model.create_embedding, audio, *args, **kwargs
            )

        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating audio embeddings."
        )

    @request_limit
    @log_async(logger=logger)
    async def convert_ids_to_tokens(
        self, input: Union[List, List[List]], *args, **kwargs
    ):
        kwargs.pop("request_id", None)
        if hasattr(self._model, "convert_ids_to_tokens"):
            return await self._call_wrapper_json(
                self._model.convert_ids_to_tokens, input, *args, **kwargs
            )

        raise AttributeError(f"Model {self._model.model_spec} can convert token id.")

    @request_limit
    @log_async(logger=logger)
    async def count_tokens(
        self, messages: List[Dict], generate_config: Optional[Dict] = None
    ):
        self._require_ready()
        return await self._call_wrapper_json(
            getattr(self._model, "async_count_tokens", self._model.count_tokens),
            messages,
            generate_config,
        )

    @request_limit
    @log_async(logger=logger)
    async def prepare_anthropic_request(
        self, messages: List[Dict], generate_config: Dict
    ):
        self._require_ready()
        return await self._call_wrapper_json(
            getattr(
                self._model,
                "async_prepare_anthropic_request",
                self._model.prepare_anthropic_request,
            ),
            messages,
            generate_config,
        )

    @log_async(logger=logger)
    async def get_anthropic_capabilities(self):
        self._require_ready()
        if not hasattr(self._model, "get_anthropic_capabilities"):
            return None

        def _describe():
            from dataclasses import asdict, is_dataclass

            capabilities = self._model.get_anthropic_capabilities()
            description = (
                asdict(capabilities) if is_dataclass(capabilities) else capabilities
            )
            if isinstance(description, dict):
                context_edits = description.get("supported_context_edits")
                if isinstance(context_edits, (set, frozenset)):
                    description["supported_context_edits"] = sorted(context_edits)
            return description

        return await self._call_wrapper_json(_describe)

    @request_limit
    @log_async(logger=logger)
    async def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
        return_len: Optional[bool],
        *args,
        **kwargs,
    ):
        self._require_ready()
        kwargs.pop("request_id", None)
        if hasattr(self._model, "rerank"):
            return await self._call_wrapper_json(
                self._model.rerank,
                documents,
                query,
                top_n,
                max_chunks_per_doc,
                return_documents,
                return_len,
                *args,
                **kwargs,
            )
        raise AttributeError(f"Model {self._model.model_spec} is not for reranking.")

    @request_limit
    @log_async(logger=logger, ignore_kwargs=["audio"])
    async def transcriptions(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
        **kwargs,
    ):
        self._require_ready()
        kwargs.pop("request_id", None)
        if hasattr(self._model, "transcriptions"):
            return await self._call_wrapper_json(
                getattr(
                    self._model, "async_transcriptions", self._model.transcriptions
                ),
                audio,
                language,
                prompt,
                response_format,
                temperature,
                timestamp_granularities,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating transcriptions."
        )

    @request_limit
    @log_async(logger=logger, ignore_kwargs=["audio"])
    async def translations(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
        **kwargs,
    ):
        self._require_ready()
        kwargs.pop("request_id", None)
        if hasattr(self._model, "translations"):
            return await self._call_wrapper_json(
                getattr(self._model, "async_translations", self._model.translations),
                audio,
                language,
                prompt,
                response_format,
                temperature,
                timestamp_granularities,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating translations."
        )

    @request_limit
    @xo.generator
    @log_async(logger=logger, ignore_kwargs=["prompt_speech"])
    async def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        self._require_ready()
        kwargs.pop("request_id", None)
        if hasattr(self._model, "speech"):
            from ..model.utils import resolve_media_seed

            if "seed" in kwargs:
                try:
                    seed = resolve_media_seed(kwargs["seed"])
                except ValueError as e:
                    raise InvalidModelRequestError(str(e)) from e
                if seed is None:
                    kwargs.pop("seed")
                else:
                    kwargs["seed"] = seed

            speech = getattr(self._model, "async_speech", self._model.speech)
            parameters = inspect.signature(speech).parameters
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            if "seed" not in parameters and not accepts_kwargs:
                kwargs.pop("seed", None)
            return await self._call_wrapper_binary(
                speech,
                input,
                voice,
                response_format,
                speed,
                stream,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating speech."
        )

    @request_limit
    @log_async(logger=logger)
    async def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        *args,
        **kwargs,
    ):
        self._require_ready()
        if hasattr(self._model, "text_to_image"):
            # Get progressor (don't pop request_id, let _call_wrapper handle cancellation)
            request_id = kwargs.get("request_id")
            progressor = kwargs["progressor"] = await self._get_progressor(request_id)  # type: ignore
            with progressor:
                return await self._call_wrapper_json(
                    self._model.text_to_image,
                    prompt,
                    n,
                    size,
                    response_format,
                    *args,
                    **kwargs,
                )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating image."
        )

    @request_limit
    @log_async(logger=logger)
    async def txt2img(
        self,
        **kwargs,
    ):
        self._require_ready()
        if hasattr(self._model, "txt2img"):
            kwargs.pop("_sdapi_lora_specs", None)
            if "<lora:" in (kwargs.get("prompt") or ""):
                worker = await self._get_worker_ref()
                specs = await worker.list_model_registrations("image", detailed=True)
                kwargs["_sdapi_lora_specs"] = [
                    spec for spec in specs if spec.get("model_family") == "lora"
                ]
            progressor = kwargs["progressor"] = await self._get_progressor(
                kwargs.get("request_id")
            )
            with progressor:
                return await self._call_wrapper_json(
                    self._model.txt2img,
                    **kwargs,
                )
        raise AttributeError(f"Model {self._model.model_spec} is not for txt2img.")

    @request_limit
    @log_async(
        logger=logger,
        ignore_kwargs=["image"],
    )
    async def image_to_image(
        self,
        image: "PIL.Image",
        prompt: str,
        negative_prompt: Optional[str] = None,
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "url",
        *args,
        **kwargs,
    ):
        self._require_ready()
        kwargs["negative_prompt"] = negative_prompt
        if hasattr(self._model, "image_to_image"):
            progressor = kwargs["progressor"] = await self._get_progressor(
                kwargs.pop("request_id", None)
            )
            with progressor:
                return await self._call_wrapper_json(
                    self._model.image_to_image,
                    image,
                    prompt,
                    n,
                    size,
                    response_format,
                    *args,
                    **kwargs,
                )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating image."
        )

    @request_limit
    @log_async(logger=logger)
    async def img2img(
        self,
        **kwargs,
    ):
        self._require_ready()
        if hasattr(self._model, "img2img"):
            kwargs.pop("_sdapi_lora_specs", None)
            if "<lora:" in (kwargs.get("prompt") or ""):
                worker = await self._get_worker_ref()
                specs = await worker.list_model_registrations("image", detailed=True)
                kwargs["_sdapi_lora_specs"] = [
                    spec for spec in specs if spec.get("model_family") == "lora"
                ]
            progressor = kwargs["progressor"] = await self._get_progressor(
                kwargs.get("request_id")
            )
            with progressor:
                return await self._call_wrapper_json(
                    self._model.img2img,
                    **kwargs,
                )
        raise AttributeError(f"Model {self._model.model_spec} is not for img2img.")

    @request_limit
    @log_async(
        logger=logger,
        ignore_kwargs=["image"],
    )
    async def inpainting(
        self,
        image: "PIL.Image",
        mask_image: "PIL.Image",
        prompt: str,
        negative_prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        *args,
        **kwargs,
    ):
        self._require_ready()
        kwargs["negative_prompt"] = negative_prompt
        if hasattr(self._model, "inpainting"):
            progressor = kwargs["progressor"] = await self._get_progressor(
                kwargs.pop("request_id", None)
            )
            with progressor:
                return await self._call_wrapper_json(
                    self._model.inpainting,
                    image,
                    mask_image,
                    prompt,
                    n,
                    size,
                    response_format,
                    *args,
                    **kwargs,
                )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating image."
        )

    @request_limit
    @log_async(
        logger=logger,
        ignore_kwargs=["image"],
    )
    async def ocr(
        self,
        image: "PIL.Image",
        *args,
        **kwargs,
    ):
        self._require_ready()
        if hasattr(self._model, "ocr"):
            return await self._call_wrapper_json(
                self._model.ocr,
                image,
                *args,
                **kwargs,
            )
        raise AttributeError(f"Model {self._model.model_spec} is not for ocr.")

    @request_limit
    @log_async(logger=logger, ignore_kwargs=["file_bytes"])
    async def docanalyze(
        self,
        file_bytes: bytes,
        file_name: str,
        *args,
        **kwargs,
    ):
        self._require_ready()
        if hasattr(self._model, "docanalyze"):
            return await self._call_wrapper_json(
                self._model.docanalyze,
                file_bytes,
                file_name,
                *args,
                **kwargs,
            )
        raise AttributeError(f"Model {self._model.model_spec} is not for docanalyze.")

    @request_limit
    @log_async(logger=logger, ignore_kwargs=["image"])
    async def infer(
        self,
        *args,
        **kwargs,
    ):
        self._require_ready()
        kwargs.pop("request_id", None)
        if hasattr(self._model, "infer"):
            return await self._call_wrapper_json(
                self._model.infer,
                *args,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for flexible infer."
        )

    @request_limit
    @log_async(logger=logger)
    async def text_to_video(
        self,
        prompt: str,
        n: int = 1,
        *args,
        **kwargs,
    ):
        self._require_ready()
        progressor = kwargs["progressor"] = await self._get_progressor(
            kwargs.pop("request_id", None)
        )
        with progressor:
            if hasattr(self._model, "text_to_video"):
                return await self._call_wrapper_json(
                    self._model.text_to_video,
                    prompt,
                    n,
                    *args,
                    **kwargs,
                )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating video."
        )

    @request_limit
    @log_async(logger=logger, ignore_kwargs=["video"])
    async def image_to_video(
        self,
        image: "PIL.Image",
        prompt: str,
        negative_prompt: Optional[str] = None,
        n: int = 1,
        *args,
        **kwargs,
    ):
        self._require_ready()
        kwargs["negative_prompt"] = negative_prompt
        progressor = kwargs["progressor"] = await self._get_progressor(
            kwargs.pop("request_id", None)
        )
        with progressor:
            if hasattr(self._model, "image_to_video"):
                return await self._call_wrapper_json(
                    self._model.image_to_video,
                    image,
                    prompt,
                    n,
                    *args,
                    **kwargs,
                )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating video from image."
        )

    @request_limit
    @log_async(logger=logger)
    async def flf_to_video(
        self,
        first_frame: "PIL.Image.Image",
        last_frame: "PIL.Image.Image",
        prompt: str,
        negative_prompt: Optional[str] = None,
        n: int = 1,
        *args,
        **kwargs,
    ):
        self._require_ready()
        kwargs["negative_prompt"] = negative_prompt
        progressor = kwargs["progressor"] = await self._get_progressor(
            kwargs.pop("request_id", None)
        )
        with progressor:
            if hasattr(self._model, "firstlastframe_to_video"):
                return await self._call_wrapper_json(
                    self._model.firstlastframe_to_video,
                    first_frame,
                    last_frame,
                    prompt,
                    n,
                    *args,
                    **kwargs,
                )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating video from first-last-frame."
        )

    @request_limit
    @log_async(logger=logger, ignore_kwargs=["image", "video"])
    async def world_generate(
        self,
        prompt: str,
        image: Optional[str] = None,
        video: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        self._require_ready()
        if hasattr(self._model, "world_generate"):
            request_id = kwargs.get("request_id")
            if request_id and hasattr(self._model, "register_request"):
                self._model.register_request(request_id)
            try:
                progressor = kwargs["progressor"] = await self._get_progressor(
                    request_id
                )
                with progressor:
                    return await self._call_wrapper_json(
                        self._model.world_generate,
                        prompt,
                        image,
                        video,
                        generation_config,
                        model_kwargs,
                        *args,
                        _wait_for_sync_on_cancel=True,
                        **kwargs,
                    )
            finally:
                if request_id and hasattr(self._model, "unregister_request"):
                    self._model.unregister_request(request_id)
        raise AttributeError(
            f"Model {self._model.model_spec} is not for world generation."
        )

    async def record_metrics(self, name, op, kwargs):
        worker_ref = await self._get_worker_ref()
        await worker_ref.record_metrics(name, op, kwargs)

    async def get_pending_requests_count(self):
        return self._pending_requests.qsize()

    @log_async(logger=logger)
    async def free_model_cache(self, request_id: str):
        """Free the seq kvcache reference count of the vLLM model."""
        from ..model.llm.vllm.core import VLLMChatModel as LLMVLLMChatModel
        from ..model.llm.vllm.core import VLLMModel as LLMVLLMModel

        if isinstance(self._model, LLMVLLMModel) or isinstance(
            self._model, LLMVLLMChatModel
        ):
            # self._model.free_seq_cache(request_id)
            engine = self._model._engine
            inner_engine = getattr(engine, "engine", None)
            scheduler = getattr(engine, "scheduler", None) or getattr(
                inner_engine, "scheduler", None
            )
            if scheduler:
                scheduler[0].free_seq_cache(request_id)

    @log_async(logger=logger)
    async def set_unpin_handler(
        self, model_uid: str, request_id: str, pd_model_actor_address: str
    ):
        """Set the unpin handle of the vLLM model."""
        from ..model.llm.vllm.core import VLLMChatModel as LLMVLLMChatModel
        from ..model.llm.vllm.core import VLLMModel as LLMVLLMModel

        if isinstance(self._model, LLMVLLMModel) or isinstance(
            self._model, LLMVLLMChatModel
        ):
            # For XavierEngine, need to access internal engine's scheduler
            # XavierEngine.engine -> XavierInternalEngine.scheduler
            engine = self._model._engine
            inner_engine = getattr(engine, "engine", None)
            scheduler = getattr(engine, "scheduler", None) or getattr(
                inner_engine, "scheduler", None
            )
            if scheduler:
                await scheduler[0].set_unpin_handler(
                    model_uid, request_id, pd_model_actor_address
                )
                logger.debug(
                    f"[PDModelActor] Set unpin handle for request {request_id} with pd model actor {pd_model_actor_address}"
                )

    @request_limit
    @log_async(logger=logger)
    async def controlnet_module_list(
        self,
        **kwargs,
    ):
        self._require_ready()
        if hasattr(self._model, "controlnet_module_list"):
            return await self._call_wrapper_json(
                self._model.controlnet_module_list,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for controlnet_module_list."
        )

    @request_limit
    @log_async(logger=logger)
    async def controlnet_control_types(
        self,
        **kwargs,
    ):
        self._require_ready()
        if hasattr(self._model, "controlnet_control_types"):
            return await self._call_wrapper_json(
                self._model.controlnet_control_types,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for controlnet_control_types."
        )

    @request_limit
    @log_async(logger=logger)
    async def controlnet_detect(
        self,
        **kwargs,
    ):
        self._require_ready()
        if hasattr(self._model, "controlnet_detect"):
            return await self._call_wrapper_json(
                self._model.controlnet_detect,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for controlnet_detect."
        )

    @request_limit
    @log_async(logger=logger)
    async def controlnet_model_list(
        self,
        **kwargs,
    ):
        self._require_ready()
        if hasattr(self._model, "controlnet_model_list"):
            return await self._call_wrapper_json(
                self._model.controlnet_model_list,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for controlnet_model_list."
        )


class ModelGenerationActor(ModelActor):
    """A fresh interpreter loads the model wrapper from its installed package."""

    def __init__(self, snapshot, supervisor_address, worker_address):
        from .model_generation import build_generation_model

        super().__init__(
            supervisor_address,
            worker_address,
            build_generation_model(snapshot),
            replica_model_uid=snapshot["model_uid"],
            model_engine=snapshot.get("model_engine", snapshot["engine"]),
            request_limits=snapshot["request_limits"],
            model_python_executable=snapshot.get("model_python_executable"),
        )
