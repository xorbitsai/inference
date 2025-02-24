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
import functools
import inspect
import json
import os
import queue
import time
import types
import uuid
from asyncio.queues import Queue
from asyncio.tasks import wait_for
from concurrent.futures import Future as ConcurrentFuture
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Union,
    no_type_check,
)

import sse_starlette.sse
import xoscar as xo

from ..constants import (
    XINFERENCE_DEFAULT_CANCEL_BLOCK_DURATION,
    XINFERENCE_LAUNCH_MODEL_RETRY,
    XINFERENCE_TEXT_TO_IMAGE_BATCHING_SIZE,
)

if TYPE_CHECKING:
    from .progress_tracker import ProgressTrackerActor
    from .worker import WorkerActor
    from ..model.llm.core import LLM
    from ..model.core import ModelDescription
    import PIL

import logging

logger = logging.getLogger(__name__)

from ..device_utils import empty_cache
from .utils import CancelMixin, json_dumps, log_async

try:
    from torch.cuda import OutOfMemoryError
except ImportError:

    class _OutOfMemoryError(Exception):
        pass

    OutOfMemoryError = _OutOfMemoryError


XINFERENCE_BATCHING_ALLOWED_VISION_MODELS = [
    "qwen-vl-chat",
    "cogvlm2",
    "glm-4v",
    "MiniCPM-V-2.6",
]

XINFERENCE_TEXT_TO_IMAGE_BATCHING_ALLOWED_MODELS = ["FLUX.1-dev", "FLUX.1-schnell"]
XINFERENCE_TEST_OUT_OF_MEMORY_ERROR = bool(
    os.getenv("XINFERENCE_TEST_OUT_OF_MEMORY_ERROR", False)
)


def request_limit(fn):
    """
    Used by ModelActor.
    As a decorator, added to a ModelActor method to control
    how many requests are accessing that method at the same time.
    """

    async def wrapped_func(self, *args, **kwargs):
        logger.debug(
            f"Request {fn.__name__}, current serve request count: {self._serve_count}, request limit: {self._request_limits} for the model {self.model_uid()}"
        )
        if 1 + self._serve_count <= self._request_limits:
            self._serve_count += 1
        else:
            raise RuntimeError(
                f"Rate limit reached for the model. Request limit {self._request_limits} for the model: {self.model_uid()}"
            )
        ret = None
        try:
            ret = await fn(self, *args, **kwargs)
        finally:
            if ret is not None and (
                inspect.isasyncgen(ret) or inspect.isgenerator(ret)
            ):
                # stream case, let client call model_ref to decrease self._serve_count
                pass
            else:
                self._serve_count -= 1
                logger.debug(
                    f"After request {fn.__name__}, current serve request count: {self._serve_count} for the model {self.model_uid()}"
                )
        return ret

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


class ModelActor(xo.StatelessActor, CancelMixin):
    _replica_model_uid: Optional[str]

    @classmethod
    def gen_uid(cls, model: "LLM"):
        return f"{model.__class__}-model-actor"

    async def __pre_destroy__(self):
        from ..model.embedding.core import EmbeddingModel
        from ..model.llm.sglang.core import SGLANGModel
        from ..model.llm.transformers.core import PytorchModel as LLMPytorchModel
        from ..model.llm.vllm.core import VLLMModel as LLMVLLMModel

        if self.allow_batching():
            try:
                assert self._scheduler_ref is not None
                await xo.destroy_actor(self._scheduler_ref)
                del self._scheduler_ref
            except Exception as e:
                logger.debug(
                    f"Destroy scheduler actor failed, address: {self.address}, error: {e}"
                )

        if self.allow_batching_for_text_to_image():
            try:
                assert self._text_to_image_scheduler_ref is not None
                await xo.destroy_actor(self._text_to_image_scheduler_ref)
                del self._text_to_image_scheduler_ref
            except Exception as e:
                logger.debug(
                    f"Destroy text_to_image scheduler actor failed, address: {self.address}, error: {e}"
                )

        if hasattr(self._model, "stop") and callable(self._model.stop):
            self._model.stop()

        if isinstance(self._model, LLMVLLMModel):
            if self._transfer_ref is not None:
                try:
                    await xo.destroy_actor(self._transfer_ref)
                    del self._transfer_ref
                except Exception as e:
                    logger.debug(
                        f"Destroy transfer actor failed, address: {self.address}, error: {e}"
                    )

        if (
            isinstance(self._model, (LLMPytorchModel, LLMVLLMModel, SGLANGModel))
            and self._model.model_spec.model_format == "pytorch"
        ) or isinstance(self._model, EmbeddingModel):
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

    def __init__(
        self,
        supervisor_address: str,
        worker_address: str,
        model: "LLM",
        replica_model_uid: str,
        model_description: Optional["ModelDescription"] = None,
        request_limits: Optional[int] = None,
        xavier_config: Optional[Dict] = None,
        n_worker: Optional[int] = 1,
        shard: Optional[int] = 0,
        driver_info: Optional[dict] = None,  # for model across workers
    ):
        super().__init__()
        from ..model.llm.lmdeploy.core import LMDeployModel
        from ..model.llm.sglang.core import SGLANGModel
        from ..model.llm.transformers.core import PytorchModel
        from ..model.llm.vllm.core import VLLMModel

        self._supervisor_address = supervisor_address
        self._worker_address = worker_address
        self._replica_model_uid = replica_model_uid
        self._model = model
        self._model_description = (
            model_description.to_dict() if model_description else {}
        )
        self._request_limits = (
            float("inf") if request_limits is None else request_limits
        )
        self._pending_requests: asyncio.Queue = asyncio.Queue()
        self._handle_pending_requests_task = None
        self._lock = (
            None
            if isinstance(
                self._model, (PytorchModel, VLLMModel, SGLANGModel, LMDeployModel)
            )
            else asyncio.locks.Lock()
        )
        self._worker_ref = None
        self._progress_tracker_ref = None
        self._serve_count = 0
        self._metrics_labels = {
            "type": self._model_description.get("model_type", "unknown"),
            "model": self.model_uid(),
            "node": self._worker_address,
            "format": self._model_description.get("model_format", "unknown"),
            "quantization": self._model_description.get("quantization", "none"),
        }
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # model across workers
        self._n_worker = n_worker
        self._shard = shard
        self._driver_info = driver_info

        self._scheduler_ref = None
        self._text_to_image_scheduler_ref = None

        if isinstance(self._model, VLLMModel):
            self._xavier_config = xavier_config
            self._model.set_xavier_config(xavier_config)
            self._transfer_ref = None

    async def __post_create__(self):
        self._loop = asyncio.get_running_loop()

        self._handle_pending_requests_task = asyncio.create_task(
            self._handle_pending_requests()
        )

        if self.allow_batching():
            from .scheduler import SchedulerActor

            self._scheduler_ref = await xo.create_actor(
                SchedulerActor,
                address=self.address,
                uid=SchedulerActor.gen_uid(self.model_uid(), self._model.rep_id),
            )

        if self.allow_batching_for_text_to_image():
            from ..model.image.scheduler.flux import FluxBatchSchedulerActor

            self._text_to_image_scheduler_ref = await xo.create_actor(
                FluxBatchSchedulerActor,
                address=self.address,
                uid=FluxBatchSchedulerActor.gen_uid(self.model_uid()),
            )

    def __repr__(self) -> str:
        return f"ModelActor({self._replica_model_uid})"

    def decrease_serve_count(self):
        self._serve_count -= 1

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
            generate_throughput = completion_tokens / duration
            coros.append(
                self.record_metrics(
                    "generate_throughput",
                    "set",
                    {
                        "labels": self._metrics_labels,
                        "value": generate_throughput,
                    },
                )
            )
        await asyncio.gather(*coros)

    async def _get_worker_ref(self) -> xo.ActorRefType["WorkerActor"]:
        from .worker import WorkerActor

        if self._worker_ref is None:
            self._worker_ref = await xo.actor_ref(
                address=self._worker_address, uid=WorkerActor.default_uid()
            )
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

    async def _get_progressor(self, request_id: str):
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

    def allow_batching(self) -> bool:
        from ..model.llm.transformers.core import PytorchModel

        model_ability = self._model_description.get("model_ability", [])

        condition = isinstance(self._model, PytorchModel)
        if condition and ("vision" in model_ability or "audio" in model_ability):
            if (
                self._model.model_family.model_name
                in XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
                or self._model.model_family.model_family
                in XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
            ):
                return True
            else:
                logger.warning(
                    f"Currently for multimodal models, "
                    f"xinference only supports {', '.join(XINFERENCE_BATCHING_ALLOWED_VISION_MODELS)} for batching. "
                    f"Your model {self._model.model_family.model_name} with model family {self._model.model_family.model_family} is disqualified."
                )
                return False
        return condition

    def allow_batching_for_text_to_image(self) -> bool:
        from ..model.image.stable_diffusion.core import DiffusionModel

        condition = XINFERENCE_TEXT_TO_IMAGE_BATCHING_SIZE is not None and isinstance(
            self._model, DiffusionModel
        )

        if condition:
            model_name = self._model._model_spec.model_name  # type: ignore
            if model_name in XINFERENCE_TEXT_TO_IMAGE_BATCHING_ALLOWED_MODELS:
                return True
            else:
                logger.warning(
                    f"Currently for image models with text_to_image ability, "
                    f"xinference only supports {', '.join(XINFERENCE_TEXT_TO_IMAGE_BATCHING_ALLOWED_MODELS)} for batching. "
                    f"Your model {model_name} is disqualified."
                )
                return False
        return condition

    async def load(self):
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
                self._model.load()
                if hasattr(self._model, "driver_info"):
                    self._driver_info = self._model.driver_info
                break
            except Exception as e:
                if (
                    i < XINFERENCE_LAUNCH_MODEL_RETRY
                    and str(e).find("busy or unavailable") >= 0
                ):
                    await asyncio.sleep(5)
                    logger.warning("Retry to load model {model_uid}: %d times", i)
                    continue
                raise
        if self.allow_batching():
            await self._scheduler_ref.set_model(self._model)
            logger.debug(
                f"Batching enabled for model: {self.model_uid()}, max_num_seqs: {self._model.get_max_num_seqs()}"
            )
        if self.allow_batching_for_text_to_image():
            await self._text_to_image_scheduler_ref.set_model(self._model)
            logger.debug(
                f"Batching enabled for model: {self.model_uid()}, max_num_images: {self._model.get_max_num_images_for_batching()}"
            )
        logger.info(f"{self} loaded")

    async def wait_for_load(self):
        if hasattr(self._model, "wait_for_load"):
            self._model.wait_for_load()

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

    async def _handle_oom_error(self, ex):
        error_message = (
            f"Model actor is out of memory, model id: {self.model_uid()}, error: {ex}"
        )
        logger.exception(error_message)
        worker_ref = await self._get_worker_ref()
        await worker_ref.update_model_status(
            self._replica_model_uid, last_error=error_message
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
                    "time_to_first_token",
                    "set",
                    {"labels": self._metrics_labels, "value": time_to_first_token},
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
                        "time_to_first_token",
                        "set",
                        {"labels": self._metrics_labels, "value": time_to_first_token},
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
            await asyncio.gather(*coros)

    async def _handle_pending_requests(self):
        logger.info("Start requests handler.")
        while True:
            gen, stream_out, stop = await self._pending_requests.get()

            async def _async_wrapper(_gen):
                try:
                    # anext is only available for Python >= 3.10
                    return await _gen.__anext__()  # noqa: F821
                except StopAsyncIteration:
                    return stop

            def _wrapper(_gen):
                # Avoid issue: https://github.com/python/cpython/issues/112182
                try:
                    return next(_gen)
                except StopIteration:
                    return stop

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

    async def _call_wrapper_json(self, fn: Callable, *args, **kwargs):
        return await self._call_wrapper("json", fn, *args, **kwargs)

    async def _call_wrapper_binary(self, fn: Callable, *args, **kwargs):
        return await self._call_wrapper("binary", fn, *args, **kwargs)

    @oom_check
    async def _call_wrapper(self, output_type: str, fn: Callable, *args, **kwargs):
        self._add_running_task(kwargs.get("request_id"))
        if self._lock is None:
            if inspect.iscoroutinefunction(fn):
                ret = await fn(*args, **kwargs)
            else:
                ret = await asyncio.to_thread(fn, *args, **kwargs)

            if inspect.isgenerator(ret):
                gen = self._to_generator(output_type, ret)
                return gen
            if inspect.isasyncgen(ret):
                gen = self._to_async_gen(output_type, ret)
                return gen
        else:
            async with self._lock:
                if inspect.iscoroutinefunction(fn):
                    ret = await fn(*args, **kwargs)
                else:
                    ret = await asyncio.to_thread(fn, *args, **kwargs)

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
                            else:
                                yield o

                    return _stream_out_async_gen()

        if output_type == "json":
            return await asyncio.to_thread(json_dumps, ret)
        else:
            assert output_type == "binary", f"Unknown output type '{output_type}'"
            return ret

    @request_limit
    @xo.generator
    @log_async(logger=logger)
    async def generate(self, prompt: str, *args, **kwargs):
        if self.allow_batching():
            # not support request_id
            kwargs.pop("request_id", None)
            return await self.handle_batching_request(
                prompt, "generate", *args, **kwargs
            )
        else:
            kwargs.pop("raw_params", None)
            if hasattr(self._model, "generate"):
                # not support request_id
                kwargs.pop("request_id", None)
                return await self._call_wrapper_json(
                    self._model.generate, prompt, *args, **kwargs
                )
            if hasattr(self._model, "async_generate"):
                if "request_id" not in kwargs:
                    kwargs["request_id"] = str(uuid.uuid1())
                else:
                    # model only accept string
                    kwargs["request_id"] = str(kwargs["request_id"])
                return await self._call_wrapper_json(
                    self._model.async_generate,
                    prompt,
                    *args,
                    **kwargs,
                )
            raise AttributeError(f"Model {self._model.model_spec} is not for generate.")

    @staticmethod
    async def _queue_consumer(
        queue: Queue, timeout: Optional[float] = None
    ) -> AsyncIterator[Any]:
        from .scheduler import (
            XINFERENCE_STREAMING_ABORT_FLAG,
            XINFERENCE_STREAMING_DONE_FLAG,
            XINFERENCE_STREAMING_ERROR_FLAG,
        )

        while True:
            # TODO: timeout setting
            res = await wait_for(queue.get(), timeout)
            if res == XINFERENCE_STREAMING_DONE_FLAG:
                break
            elif res == XINFERENCE_STREAMING_ABORT_FLAG:
                raise RuntimeError(
                    f"This request has been cancelled by another `abort_request` request."
                )
            elif isinstance(res, str) and res.startswith(
                XINFERENCE_STREAMING_ERROR_FLAG
            ):
                raise RuntimeError(res[len(XINFERENCE_STREAMING_ERROR_FLAG) :])
            else:
                yield res

    @staticmethod
    def _get_stream_from_args(*args) -> bool:
        assert args[0] is None or isinstance(args[0], dict)
        return False if args[0] is None else args[0].get("stream", False)

    async def handle_batching_request(
        self, prompt_or_messages: Union[str, List[Dict]], call_ability, *args, **kwargs
    ):
        """
        The input parameter `prompt_or_messages`:
        - when the model_ability is `generate`, it's `prompt`, which is str type.
        - when the model_ability is `chat`, it's `messages`, which is List[Dict] type.
        """
        stream = self._get_stream_from_args(*args)
        assert self._scheduler_ref is not None
        if stream:
            assert self._scheduler_ref is not None
            queue: Queue[Any] = Queue()
            ret = self._queue_consumer(queue)
            await self._scheduler_ref.add_request(
                prompt_or_messages, queue, call_ability, *args, **kwargs
            )
            gen = self._to_async_gen("json", ret)
            return gen
        else:
            from .scheduler import XINFERENCE_NON_STREAMING_ABORT_FLAG

            assert self._loop is not None
            future = ConcurrentFuture()
            await self._scheduler_ref.add_request(
                prompt_or_messages, future, call_ability, *args, **kwargs
            )
            fut = asyncio.wrap_future(future, loop=self._loop)
            result = await fut
            if result == XINFERENCE_NON_STREAMING_ABORT_FLAG:
                raise RuntimeError(
                    f"This request has been cancelled by another `abort_request` request."
                )
            return await asyncio.to_thread(json_dumps, result)

    @request_limit
    @xo.generator
    @log_async(logger=logger)
    async def chat(self, messages: List[Dict], *args, **kwargs):
        start_time = time.time()
        response = None
        try:
            if self.allow_batching():
                # not support request_id
                kwargs.pop("request_id", None)
                return await self.handle_batching_request(
                    messages, "chat", *args, **kwargs
                )
            else:
                kwargs.pop("raw_params", None)
                if hasattr(self._model, "chat"):
                    # not support request_id
                    kwargs.pop("request_id", None)
                    response = await self._call_wrapper_json(
                        self._model.chat, messages, *args, **kwargs
                    )
                    return response
                if hasattr(self._model, "async_chat"):
                    if "request_id" not in kwargs:
                        kwargs["request_id"] = str(uuid.uuid1())
                    else:
                        # model only accept string
                        kwargs["request_id"] = str(kwargs["request_id"])
                    response = await self._call_wrapper_json(
                        self._model.async_chat, messages, *args, **kwargs
                    )
                    return response
                raise AttributeError(f"Model {self._model.model_spec} is not for chat.")
        finally:
            # For the non stream result.
            record = None
            if isinstance(response, Generator) or isinstance(response, AsyncGenerator):
                record = response
            elif isinstance(response, bytes):
                record = json.loads(response)
            if record and isinstance(record, dict):
                usage = record["usage"]
                # Some backends may not have a valid usage, we just skip them.
                completion_tokens = usage["completion_tokens"]
                prompt_tokens = usage["prompt_tokens"]
                await self._record_completion_metrics(
                    time.time() - start_time,
                    completion_tokens,
                    prompt_tokens,
                )

    async def abort_request(
        self,
        request_id: str,
        block_duration: int = XINFERENCE_DEFAULT_CANCEL_BLOCK_DURATION,
    ) -> str:
        from .utils import AbortRequestMessage

        self._cancel_running_task(request_id, block_duration)
        if self.allow_batching():
            if self._scheduler_ref is None:
                return AbortRequestMessage.NOT_FOUND.name
            return await self._scheduler_ref.abort_request(request_id)
        elif self.allow_batching_for_text_to_image():
            if self._text_to_image_scheduler_ref is None:
                return AbortRequestMessage.NOT_FOUND.name
            return await self._text_to_image_scheduler_ref.abort_request(request_id)
        return AbortRequestMessage.NO_OP.name

    @request_limit
    @log_async(logger=logger)
    async def create_embedding(self, input: Union[str, List[str]], *args, **kwargs):
        kwargs.pop("request_id", None)
        if hasattr(self._model, "create_embedding"):
            return await self._call_wrapper_json(
                self._model.create_embedding, input, *args, **kwargs
            )

        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating embedding."
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
        kwargs.pop("request_id", None)
        if hasattr(self._model, "transcriptions"):
            return await self._call_wrapper_json(
                self._model.transcriptions,
                audio,
                language,
                prompt,
                response_format,
                temperature,
                timestamp_granularities,
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
        kwargs.pop("request_id", None)
        if hasattr(self._model, "translations"):
            return await self._call_wrapper_json(
                self._model.translations,
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
        kwargs.pop("request_id", None)
        if hasattr(self._model, "speech"):
            return await self._call_wrapper_binary(
                self._model.speech,
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

    async def handle_image_batching_request(self, unique_id, *args, **kwargs):
        size = args[2]
        if XINFERENCE_TEXT_TO_IMAGE_BATCHING_SIZE != size:
            raise RuntimeError(
                f"The image size: {size} of text_to_image for batching "
                f"must be the same as the environment variable: {XINFERENCE_TEXT_TO_IMAGE_BATCHING_SIZE} you set."
            )
        assert self._loop is not None
        future = ConcurrentFuture()
        await self._text_to_image_scheduler_ref.add_request(
            unique_id, future, *args, **kwargs
        )
        fut = asyncio.wrap_future(future, loop=self._loop)
        result = await fut
        return await asyncio.to_thread(json_dumps, result)

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
        if hasattr(self._model, "text_to_image"):
            if self.allow_batching_for_text_to_image():
                unique_id = kwargs.pop("request_id", None)
                return await self.handle_image_batching_request(
                    unique_id, prompt, n, size, response_format, *args, **kwargs
                )
            else:
                progressor = kwargs["progressor"] = await self._get_progressor(
                    kwargs.pop("request_id", None)
                )
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
        if hasattr(self._model, "txt2img"):
            progressor = kwargs["progressor"] = await self._get_progressor(
                kwargs.pop("request_id", None)
            )
            with progressor:
                return await self._call_wrapper_json(
                    self._model.txt2img,
                    **kwargs,
                )
        raise AttributeError(f"Model {self._model.model_spec} is not for txt2img.")

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
        if hasattr(self._model, "img2img"):
            progressor = kwargs["progressor"] = await self._get_progressor(
                kwargs.pop("request_id", None)
            )
            with progressor:
                return await self._call_wrapper_json(
                    self._model.img2img,
                    **kwargs,
                )
        raise AttributeError(f"Model {self._model.model_spec} is not for img2img.")

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
        if hasattr(self._model, "ocr"):
            return await self._call_wrapper_json(
                self._model.ocr,
                image,
                *args,
                **kwargs,
            )
        raise AttributeError(f"Model {self._model.model_spec} is not for ocr.")

    @request_limit
    @log_async(logger=logger, ignore_kwargs=["image"])
    async def infer(
        self,
        **kwargs,
    ):
        kwargs.pop("request_id", None)
        if hasattr(self._model, "infer"):
            return await self._call_wrapper_json(
                self._model.infer,
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
        kwargs.pop("request_id", None)
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

    async def record_metrics(self, name, op, kwargs):
        worker_ref = await self._get_worker_ref()
        await worker_ref.record_metrics(name, op, kwargs)

    async def get_pending_requests_count(self):
        return self._pending_requests.qsize()
