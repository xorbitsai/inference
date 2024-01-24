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
import time
import types
import weakref
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

import sse_starlette.sse
import xoscar as xo

if TYPE_CHECKING:
    from .worker import WorkerActor
    from ..model.llm.core import LLM
    from ..model.core import ModelDescription
    import PIL

import logging

logger = logging.getLogger(__name__)

from .utils import json_dumps, log_async

try:
    from torch.cuda import OutOfMemoryError
except ImportError:

    class _OutOfMemoryError(Exception):
        pass

    OutOfMemoryError = _OutOfMemoryError


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
        if self._request_limits is not None:
            if 1 + self._serve_count <= self._request_limits:
                self._serve_count += 1
            else:
                raise RuntimeError(
                    f"Rate limit reached for the model. Request limit {self._request_limits} for the model: {self.model_uid()}"
                )
        try:
            ret = await fn(self, *args, **kwargs)
        finally:
            if self._request_limits is not None:
                self._serve_count -= 1
            logger.debug(
                f"After request {fn.__name__}, current serve request count: {self._serve_count} for the model {self.model_uid()}"
            )
        return ret

    return wrapped_func


def oom_check(fn):
    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except OutOfMemoryError:
            logger.exception("Model actor is out of memory.")
            os._exit(1)

    @functools.wraps(fn)
    async def _async_wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except OutOfMemoryError:
            logger.exception("Model actor is out of memory.")
            os._exit(1)

    assert not inspect.isasyncgen(fn)
    assert not inspect.isgenerator(fn)

    if asyncio.iscoroutinefunction(fn):
        return _async_wrapper
    else:
        return _wrapper


class ModelActor(xo.StatelessActor):
    @classmethod
    def gen_uid(cls, model: "LLM"):
        return f"{model.__class__}-model-actor"

    async def __pre_destroy__(self):
        from ..model.embedding.core import EmbeddingModel
        from ..model.llm.pytorch.core import PytorchModel as LLMPytorchModel
        from ..model.llm.vllm.core import VLLMModel as LLMVLLMModel

        if (
            isinstance(self._model, (LLMPytorchModel, LLMVLLMModel))
            and self._model.model_spec.model_format == "pytorch"
        ) or isinstance(self._model, EmbeddingModel):
            try:
                import gc

                import torch
            except ImportError:
                error_message = "Failed to import module 'torch'"
                installation_guide = [
                    "Please make sure 'torch' is installed.\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

            del self._model
            gc.collect()
            torch.cuda.empty_cache()

    def __init__(
        self,
        worker_address: str,
        model: "LLM",
        model_description: Optional["ModelDescription"] = None,
        request_limits: Optional[int] = None,
    ):
        super().__init__()
        from ..model.llm.pytorch.core import PytorchModel
        from ..model.llm.pytorch.spec_model import SpeculativeModel
        from ..model.llm.vllm.core import VLLMModel

        self._worker_address = worker_address
        self._model = model
        self._model_description = (
            model_description.to_dict() if model_description else {}
        )
        self._request_limits = request_limits

        self._generators: Dict[str, Union[Iterator, AsyncGenerator]] = {}
        self._current_generator = lambda: None
        self._lock = (
            None
            if isinstance(self._model, (PytorchModel, SpeculativeModel, VLLMModel))
            else asyncio.locks.Lock()
        )
        self._worker_ref = None
        self._serve_count = 0
        self._metrics_labels = {
            "type": self._model_description.get("model_type", "unknown"),
            "model": self.model_uid(),
            "node": self._worker_address,
            "format": self._model_description.get("model_format", "unknown"),
            "quantization": self._model_description.get("quantization", "none"),
        }
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def __post_create__(self):
        self._loop = asyncio.get_running_loop()

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
                address=self._worker_address, uid=WorkerActor.uid()
            )
        return self._worker_ref

    def is_vllm_backend(self) -> bool:
        from ..model.llm.vllm.core import VLLMModel

        return isinstance(self._model, VLLMModel)

    def load(self):
        self._model.load()

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

    def _to_json_generator(self, gen: types.GeneratorType):
        start_time = time.time()
        time_to_first_token = None
        final_usage = None
        try:
            for v in gen:
                if time_to_first_token is None:
                    time_to_first_token = (time.time() - start_time) * 1000
                final_usage = v.pop("usage", None)
                v = dict(data=json.dumps(v))
                yield sse_starlette.sse.ensure_bytes(v, None)
        except OutOfMemoryError:
            logger.exception(
                "Model actor is out of memory, model id: %s", self.model_uid()
            )
            os._exit(1)
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

    async def _to_json_async_gen(self, gen: types.AsyncGeneratorType):
        start_time = time.time()
        time_to_first_token = None
        final_usage = None
        try:
            async for v in gen:
                if time_to_first_token is None:
                    time_to_first_token = (time.time() - start_time) * 1000
                final_usage = v.pop("usage", None)
                v = await asyncio.to_thread(json.dumps, v)
                v = dict(data=v)  # noqa: F821
                yield await asyncio.to_thread(sse_starlette.sse.ensure_bytes, v, None)
        except OutOfMemoryError:
            logger.exception(
                "Model actor is out of memory, model id: %s", self.model_uid()
            )
            os._exit(1)
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

    @oom_check
    async def _call_wrapper(self, fn: Callable, *args, **kwargs):
        if self._lock is None:
            if inspect.iscoroutinefunction(fn):
                ret = await fn(*args, **kwargs)
            else:
                ret = await asyncio.to_thread(fn, *args, **kwargs)
        else:
            async with self._lock:
                if inspect.iscoroutinefunction(fn):
                    ret = await fn(*args, **kwargs)
                else:
                    ret = await asyncio.to_thread(fn, *args, **kwargs)

        if self._lock is not None and self._current_generator():
            raise Exception("Parallel generation is not supported by ggml.")

        if inspect.isgenerator(ret):
            gen = self._to_json_generator(ret)
            self._current_generator = weakref.ref(gen)
            return gen
        if inspect.isasyncgen(ret):
            gen = self._to_json_async_gen(ret)
            self._current_generator = weakref.ref(gen)
            return gen
        return await asyncio.to_thread(json_dumps, ret)

    @log_async(logger=logger)
    @request_limit
    @xo.generator
    async def generate(self, prompt: str, *args, **kwargs):
        if hasattr(self._model, "generate"):
            return await self._call_wrapper(
                self._model.generate, prompt, *args, **kwargs
            )
        if hasattr(self._model, "async_generate"):
            return await self._call_wrapper(
                self._model.async_generate, prompt, *args, **kwargs
            )
        raise AttributeError(f"Model {self._model.model_spec} is not for generate.")

    @log_async(logger=logger)
    @request_limit
    @xo.generator
    async def chat(self, prompt: str, *args, **kwargs):
        start_time = time.time()
        response = None
        try:
            if hasattr(self._model, "chat"):
                response = await self._call_wrapper(
                    self._model.chat, prompt, *args, **kwargs
                )
                return response
            if hasattr(self._model, "async_chat"):
                response = await self._call_wrapper(
                    self._model.async_chat, prompt, *args, **kwargs
                )
                return response
            raise AttributeError(f"Model {self._model.model_spec} is not for chat.")
        finally:
            # For the non stream result.
            if response is not None and isinstance(response, dict):
                usage = response["usage"]
                # Some backends may not have a valid usage, we just skip them.
                completion_tokens = usage["completion_tokens"]
                prompt_tokens = usage["prompt_tokens"]
                await self._record_completion_metrics(
                    time.time() - start_time,
                    completion_tokens,
                    prompt_tokens,
                )

    @log_async(logger=logger)
    @request_limit
    async def create_embedding(self, input: Union[str, List[str]], *args, **kwargs):
        if hasattr(self._model, "create_embedding"):
            return await self._call_wrapper(
                self._model.create_embedding, input, *args, **kwargs
            )

        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating embedding."
        )

    @log_async(logger=logger)
    @request_limit
    async def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
        *args,
        **kwargs,
    ):
        if hasattr(self._model, "rerank"):
            return await self._call_wrapper(
                self._model.rerank,
                documents,
                query,
                top_n,
                max_chunks_per_doc,
                return_documents,
                *args,
                **kwargs,
            )
        raise AttributeError(f"Model {self._model.model_spec} is not for reranking.")

    @log_async(logger=logger, args_formatter=lambda _, kwargs: kwargs.pop("audio"))
    @request_limit
    async def transcriptions(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
    ):
        if hasattr(self._model, "transcriptions"):
            return await self._call_wrapper(
                self._model.transcriptions,
                audio,
                language,
                prompt,
                response_format,
                temperature,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating transcriptions."
        )

    @log_async(logger=logger, args_formatter=lambda _, kwargs: kwargs.pop("audio"))
    @request_limit
    async def translations(
        self,
        audio: bytes,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
    ):
        if hasattr(self._model, "translations"):
            return await self._call_wrapper(
                self._model.translations,
                audio,
                prompt,
                response_format,
                temperature,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating translations."
        )

    @log_async(logger=logger)
    @request_limit
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
            return await self._call_wrapper(
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

    async def image_to_image(
        self,
        image: "PIL.Image",
        prompt: str,
        negative_prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        *args,
        **kwargs,
    ):
        if hasattr(self._model, "image_to_image"):
            return await self._call_wrapper(
                self._model.image_to_image,
                image,
                prompt,
                negative_prompt,
                n,
                size,
                response_format,
                *args,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating image."
        )

    async def record_metrics(self, name, op, kwargs):
        worker_ref = await self._get_worker_ref()
        await worker_ref.record_metrics(name, op, kwargs)
