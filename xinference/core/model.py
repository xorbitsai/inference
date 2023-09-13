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

        if (
            isinstance(self._model, LLMPytorchModel)
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

        self._model = model
        self._generators: Dict[str, Iterator] = {}
        self._lock = (
            None if isinstance(self._model, PytorchModel) else asyncio.locks.Lock()
        )

    def load(self):
        self._model.load()

    async def _wrap_generator(self, ret: Any):
        if inspect.isgenerator(ret):
            generator_uid = str(uuid.uuid1())
            self._generators[generator_uid] = ret
            return IteratorWrapper(
                uid=generator_uid,
                model_actor_addr=self.address,
                model_actor_uid=self.uid,
            )
        else:
            return ret

    async def _call_wrapper(self, _wrapper):
        if self._lock is None:
            return await asyncio.to_thread(_wrapper)
        else:
            async with self._lock:
                return await asyncio.to_thread(_wrapper)

    async def generate(self, prompt: str, *args, **kwargs):
        if not hasattr(self._model, "generate"):
            raise AttributeError(f"Model {self._model.model_spec} is not for generate.")

        def _wrapper():
            return self._wrap_generator(
                getattr(self._model, "generate")(prompt, *args, **kwargs)
            )

        return await self._call_wrapper(_wrapper)

    async def chat(self, prompt: str, *args, **kwargs):
        if not hasattr(self._model, "chat"):
            raise AttributeError(f"Model {self._model.model_spec} is not for chat.")

        def _wrapper():
            return self._wrap_generator(
                getattr(self._model, "chat")(prompt, *args, **kwargs)
            )

        return await self._call_wrapper(_wrapper)

    async def create_embedding(self, input: Union[str, List[str]], *args, **kwargs):
        if not hasattr(self._model, "create_embedding"):
            raise AttributeError(
                f"Model {self._model.model_spec} is not for creating embedding."
            )

        def _wrapper():
            return getattr(self._model, "create_embedding")(input, *args, **kwargs)

        return await self._call_wrapper(_wrapper)

    async def next(
        self, generator_uid: str
    ) -> Union["ChatCompletionChunk", "CompletionChunk"]:
        assert generator_uid in self._generators
        stop = object()

        def _wrapper():
            try:
                return next(self._generators[generator_uid])
            except StopIteration:
                return stop

        r = await self._call_wrapper(_wrapper)
        if r is stop:
            self._generators.pop(generator_uid, None)
            raise Exception("StopIteration")
        else:
            return r
