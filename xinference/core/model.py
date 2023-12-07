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
import inspect
import os
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

import xoscar as xo

if TYPE_CHECKING:
    from ..model.llm.core import LLM
    from ..types import ChatCompletionChunk, CompletionChunk
    import PIL

import logging

logger = logging.getLogger(__name__)

from .utils import json_dumps, log_async

T = TypeVar("T")


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


class IteratorWrapper(Generic[T]):
    def __init__(self, uid: str, model_actor_addr: str, model_actor_uid: str):
        self._uid = uid
        self._model_actor_addr = model_actor_addr
        self._model_actor_uid = model_actor_uid
        self._model_actor_ref: Optional[xo.ActorRefType["ModelActor"]] = None

    async def destroy(self):
        if self._model_actor_ref is None:
            self._model_actor_ref = await xo.actor_ref(
                address=self._model_actor_addr, uid=self._model_actor_uid
            )
        assert self._model_actor_ref is not None
        return await self._model_actor_ref.destroy_generator(self._uid)

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        if self._model_actor_ref is None:
            self._model_actor_ref = await xo.actor_ref(
                address=self._model_actor_addr, uid=self._model_actor_uid
            )

        try:
            assert self._model_actor_ref is not None
            return await self._model_actor_ref.next(self._uid)
        except Exception as e:
            if "StopIteration" in str(e):
                raise StopAsyncIteration
            else:
                raise


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

    def __init__(self, model: "LLM", request_limits: Optional[int] = None):
        super().__init__()
        from ..model.llm.pytorch.core import PytorchModel
        from ..model.llm.pytorch.spec_model import SpeculativeModel
        from ..model.llm.vllm.core import VLLMModel

        self._model = model
        self._request_limits = request_limits

        self._generators: Dict[str, Union[Iterator, AsyncGenerator]] = {}
        self._lock = (
            None
            if isinstance(self._model, (PytorchModel, SpeculativeModel, VLLMModel))
            else asyncio.locks.Lock()
        )
        self._serve_count = 0

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

    async def _wrap_generator(self, ret: Any):
        if inspect.isgenerator(ret) or inspect.isasyncgen(ret):
            if self._lock is not None and self._generators:
                raise Exception("Parallel generation is not supported by ggml.")
            generator_uid = str(uuid.uuid1())
            self._generators[generator_uid] = ret

            return IteratorWrapper(
                uid=generator_uid,
                model_actor_addr=self.address,
                model_actor_uid=self.uid,
            )
        else:
            return ret

    async def _call_wrapper(self, _wrapper: Callable):
        assert not (
            inspect.iscoroutinefunction(_wrapper)
            or inspect.isasyncgenfunction(_wrapper)
        )
        if self._lock is None:
            return await asyncio.to_thread(_wrapper)
        else:
            async with self._lock:
                return await asyncio.to_thread(_wrapper)

    async def _call_async_wrapper(self, _wrapper: Callable):
        return await asyncio.create_task(_wrapper())

    @log_async(logger=logger)
    @request_limit
    async def generate(self, prompt: str, *args, **kwargs):
        if not hasattr(self._model, "generate") and not hasattr(
            self._model, "async_generate"
        ):
            raise AttributeError(f"Model {self._model.model_spec} is not for generate.")

        def _wrapper():
            return self._wrap_generator(
                getattr(self._model, "generate")(prompt, *args, **kwargs)
            )

        async def _async_wrapper():
            # for vLLM.
            return self._wrap_generator(
                await getattr(self._model, "async_generate")(prompt, *args, **kwargs)
            )

        if hasattr(self._model, "generate"):
            return await self._call_wrapper(_wrapper)
        else:
            return await self._call_async_wrapper(_async_wrapper)

    @log_async(logger=logger)
    @request_limit
    async def chat(self, prompt: str, *args, **kwargs):
        if not hasattr(self._model, "chat") and not hasattr(self._model, "async_chat"):
            raise AttributeError(f"Model {self._model.model_spec} is not for chat.")

        def _wrapper():
            return self._wrap_generator(
                getattr(self._model, "chat")(prompt, *args, **kwargs)
            )

        async def _async_wrapper():
            # for vLLM.
            return self._wrap_generator(
                await getattr(self._model, "async_chat")(prompt, *args, **kwargs)
            )

        if hasattr(self._model, "async_chat"):
            return await self._call_async_wrapper(_async_wrapper)
        else:
            return await self._call_wrapper(_wrapper)

    @log_async(logger=logger)
    @request_limit
    async def create_embedding(self, input: Union[str, List[str]], *args, **kwargs):
        if not hasattr(self._model, "create_embedding"):
            raise AttributeError(
                f"Model {self._model.model_spec} is not for creating embedding."
            )

        def _wrapper():
            data = getattr(self._model, "create_embedding")(input, *args, **kwargs)
            return json_dumps(data)

        return await self._call_wrapper(_wrapper)

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
        if not hasattr(self._model, "rerank"):
            raise AttributeError(
                f"Model {self._model.model_spec} is not for reranking."
            )

        def _wrapper():
            data = getattr(self._model, "rerank")(
                documents,
                query,
                top_n,
                max_chunks_per_doc,
                return_documents,
                *args,
                **kwargs,
            )
            return json_dumps(data)

        return await self._call_wrapper(_wrapper)

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
        if not hasattr(self._model, "text_to_image"):
            raise AttributeError(
                f"Model {self._model.model_spec} is not for creating image."
            )

        def _wrapper():
            return getattr(self._model, "text_to_image")(
                prompt, n, size, response_format, *args, **kwargs
            )

        return await self._call_wrapper(_wrapper)

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
        if not hasattr(self._model, "image_to_image"):
            raise AttributeError(
                f"Model {self._model.model_spec} is not for creating image."
            )

        def _wrapper():
            return getattr(self._model, "image_to_image")(
                image,
                prompt,
                negative_prompt,
                n,
                size,
                response_format,
                *args,
                **kwargs,
            )

        return await self._call_wrapper(_wrapper)

    @log_async(logger=logger)
    async def next(
        self, generator_uid: str
    ) -> Union["ChatCompletionChunk", "CompletionChunk"]:
        assert generator_uid in self._generators
        stop = object()
        gen = self._generators[generator_uid]

        try:
            from torch.cuda import OutOfMemoryError
        except ImportError:

            class _OutOfMemoryError(Exception):
                pass

            OutOfMemoryError = _OutOfMemoryError

        def _wrapper():
            try:
                return next(gen)
            except OutOfMemoryError:
                logger.exception(
                    "Model actor is out of memory, model id: %s", self.model_uid()
                )
                os._exit(1)
            except StopIteration:
                return stop

        async def _async_wrapper():
            try:
                # anext is only available for Python >= 3.10
                return await gen.__anext__()  # noqa: F821
            except StopAsyncIteration:
                return stop

        if inspect.isgenerator(gen):
            r = await self._call_wrapper(_wrapper)
        elif inspect.isasyncgen(gen):
            # for vLLM.
            r = await self._call_async_wrapper(_async_wrapper)
        else:
            raise TypeError(
                f"Unexpected type {type(gen)}, expecting generator or async generator"
            )

        if r is stop:
            self._generators.pop(generator_uid, None)
            raise Exception("StopIteration")
        else:
            return r

    @log_async(logger=logger)
    async def destroy_generator(self, generator_uid: str):
        self._generators.pop(generator_uid, None)
