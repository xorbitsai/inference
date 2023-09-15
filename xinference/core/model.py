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

import logging

logger = logging.getLogger(__name__)


T = TypeVar("T")


class IteratorWrapper(Generic[T]):
    def __init__(self, uid: str, model_actor_addr: str, model_actor_uid: str):
        self._uid = uid
        self._model_actor_addr = model_actor_addr
        self._model_actor_uid = model_actor_uid
        self._model_actor_ref: Optional[xo.ActorRefType["ModelActor"]] = None

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

    def __init__(self, model: "LLM"):
        super().__init__()
        from ..model.llm.pytorch.core import PytorchModel
        from ..model.llm.vllm.core import VLLMModel

        self._model = model

        self._generators: Dict[str, Union[Iterator, AsyncGenerator]] = {}
        self._lock = (
            None
            if isinstance(self._model, (PytorchModel, VLLMModel))
            else asyncio.locks.Lock()
        )

    def load(self):
        self._model.load()

    async def _wrap_generator(self, ret: Any):
        if inspect.isgenerator(ret) or inspect.isasyncgen(ret):
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
        if self._lock is None:
            return await asyncio.to_thread(_wrapper)
        else:
            async with self._lock:
                return await asyncio.to_thread(_wrapper)

    async def _call_async_wrapper(self, _wrapper: Callable):
        return await asyncio.create_task(_wrapper())

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

        if hasattr(self._model, "generate"):
            return await self._call_wrapper(_wrapper)
        else:
            return await self._call_async_wrapper(_async_wrapper)

    async def create_embedding(self, input: Union[str, List[str]], *args, **kwargs):
        if not hasattr(self._model, "create_embedding"):
            raise AttributeError(
                f"Model {self._model.model_spec} is not for creating embedding."
            )

        async def _wrapper():
            return getattr(self._model, "create_embedding")(input, *args, **kwargs)

        return await self._call_wrapper(_wrapper)

    async def next(
        self, generator_uid: str
    ) -> Union["ChatCompletionChunk", "CompletionChunk"]:
        assert generator_uid in self._generators
        stop = object()
        gen = self._generators[generator_uid]

        def _wrapper():
            try:
                return next(gen)
            except StopIteration:
                return stop

        async def _async_wrapper():
            try:
                return await anext(gen)
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
