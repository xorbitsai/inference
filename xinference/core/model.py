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
import functools
import inspect
import json
import os
import queue
import time
import types
import uuid
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
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
)

if TYPE_CHECKING:
    from .progress_tracker import ProgressTrackerActor
    from .worker import WorkerActor
    from ..model.llm.core import LLM
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


# !!!!! DO NOT add model_name to this list, using `register_batching_multimodal_models` below instead.
XINFERENCE_BATCHING_ALLOWED_VISION_MODELS = []

XINFERENCE_TEXT_TO_IMAGE_BATCHING_ALLOWED_MODELS = ["FLUX.1-dev", "FLUX.1-schnell"]
XINFERENCE_TEST_OUT_OF_MEMORY_ERROR = bool(
    os.getenv("XINFERENCE_TEST_OUT_OF_MEMORY_ERROR", False)
)


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

    async def __pre_destroy__(self):
        from ..model.embedding.core import EmbeddingModel
        from ..model.llm.sglang.core import SGLANGModel
        from ..model.llm.transformers.core import PytorchModel as LLMPytorchModel
        from ..model.llm.vllm.core import VLLMModel as LLMVLLMModel

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
        request_limits: Optional[int] = None,
        xavier_config: Optional[Dict] = None,
        n_worker: Optional[int] = 1,
        shard: Optional[int] = 0,
        driver_info: Optional[dict] = None,  # for model across workers
    ):
        super().__init__()

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

    def __repr__(self) -> str:
        return f"ModelActor({self._replica_model_uid})"

    def __getattr__(self, attr: str):
        return getattr(self._model, attr)

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

    def is_sglang_backend(self) -> bool:
        from ..model.llm.sglang.core import SGLANGModel

        return isinstance(self._model, SGLANGModel)

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
                if hasattr(self._model, "set_loop"):
                    self._model.set_loop(asyncio.get_running_loop())
                await asyncio.to_thread(self._model.load)
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
        logger.info(f"{self} loaded")

    async def wait_for_load(self):
        if hasattr(self._model, "wait_for_load"):
            await asyncio.to_thread(self._model.wait_for_load)

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

    @request_limit
    @xo.generator
    @log_async(logger=logger)
    async def generate(self, prompt: str, *args, **kwargs):
        # Directly delegate to model, let model decide how to handle (batching or not)
        kwargs.pop("raw_params", None)
        if hasattr(self._model, "generate"):
            # not support request_id for generate
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

    @request_limit
    @xo.generator
    @log_async(logger=logger)
    async def chat(self, messages: List[Dict], *args, **kwargs):
        start_time = time.time()
        response = None
        try:
            # Directly delegate to model, let model decide how to handle (batching or not)
            kwargs.pop("raw_params", None)
            if hasattr(self._model, "chat"):
                # Only remove request_id if model doesn't have batch scheduler
                if not (
                    hasattr(self._model, "_batch_scheduler")
                    and self._model._batch_scheduler
                ):
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
        from ..model.scheduler.core import AbortRequestMessage

        # Always cancel the running task first
        self._cancel_running_task(request_id, block_duration)

        # If model has abort_request method, delegate to it
        if hasattr(self._model, "abort_request"):
            result = await self._model.abort_request(request_id)
            if result is not None:
                return result

        # Otherwise return NO_OP for legacy models or when model doesn't handle abort
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
        *args,
        **kwargs,
    ):
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
    @log_async(logger=logger)
    async def image_to_video(
        self,
        image: "PIL.Image",
        prompt: str,
        negative_prompt: Optional[str] = None,
        n: int = 1,
        *args,
        **kwargs,
    ):
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

    async def record_metrics(self, name, op, kwargs):
        worker_ref = await self._get_worker_ref()
        await worker_ref.record_metrics(name, op, kwargs)

    async def get_pending_requests_count(self):
        return self._pending_requests.qsize()
