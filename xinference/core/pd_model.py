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
import copy
import logging
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

import xoscar as xo

from .utils import log_async

if TYPE_CHECKING:
    from .model import ModelActor

logger = logging.getLogger(__name__)


class SchedulingPolicy(ABC):
    """调度策略抽象基类"""

    @abstractmethod
    def schedule(self) -> xo.ActorRefType["ModelActor"]:
        raise NotImplementedError("Scheduling Policy is not set.")

    @abstractmethod
    def update_replicas(self, model_replicas: List[xo.ActorRefType["ModelActor"]]):
        raise NotImplementedError("Scheduling Policy is not set.")


class RoundRobinSchedulingPolicy(SchedulingPolicy):
    """轮询调度策略"""

    def __init__(self, model_replicas: List[xo.ActorRefType["ModelActor"]]):
        self._model_replicas = model_replicas
        self._model_replicas_cycle = iter(self._model_replicas)
        super().__init__()

    def schedule(self) -> xo.ActorRefType["ModelActor"]:
        if not self._model_replicas:
            raise RuntimeError("No model replicas available for scheduling")
        try:
            return next(self._model_replicas_cycle)
        except StopIteration:
            self._model_replicas_cycle = iter(self._model_replicas)
            return next(self._model_replicas_cycle)

    def update_replicas(self, model_replicas: List[xo.ActorRefType["ModelActor"]]):
        """更新副本列表，重置轮询迭代器"""
        self._model_replicas = model_replicas
        self._model_replicas_cycle = iter(self._model_replicas)

    def __repr__(self) -> str:
        return f"RoundRobinSchedulingPolicy({len(self._model_replicas)} replicas)"


class PDModelActor(xo.StatelessActor):
    """PD分离模型Actor - 支持多P多D"""

    @classmethod
    def default_uid(cls):
        return "pd-model-actor"

    def __init__(
        self,
        model_uid: str,
        scheduling_policy: Callable[
            [List[xo.ActorRefType["ModelActor"]]], SchedulingPolicy
        ] = RoundRobinSchedulingPolicy,
    ):
        super().__init__()
        # Prefill request map, used to skip the timeout task for specific request id.
        self._request_set: Set[str] = set()

        self._model_uid = model_uid

        # 使用字典存储副本：{replica_uid: actor_ref}
        self._prefill_replicas: Dict[str, xo.ActorRefType["ModelActor"]] = {}
        self._decode_replicas: Dict[str, xo.ActorRefType["ModelActor"]] = {}

        self._prefill_policy: Optional[SchedulingPolicy] = None
        self._decode_policy: Optional[SchedulingPolicy] = None
        self._scheduling_policy: Callable[
            [List[xo.ActorRefType["ModelActor"]]], SchedulingPolicy
        ] = scheduling_policy

        logger.info(
            f"Initialize PDModelActor for model {model_uid} using {scheduling_policy.__name__}"
        )

    async def __post_create__(self):
        pass

    async def __pre_destroy__(self):
        await asyncio.gather(
            *[
                self.free_prefill_model_cache(request_id)
                for request_id in list(self._request_set)
            ],
            return_exceptions=True,
        )
        await super().__pre_destroy__()

    async def add_prefill_actor(
        self,
        replica_uid: str,
        actor: xo.ActorRefType["ModelActor"],
    ):
        """添加 Prefill Actor"""
        if self._prefill_replicas.get(replica_uid) != actor:
            self._prefill_replicas[replica_uid] = actor
            # 更新调度策略的副本列表
            if self._prefill_policy:
                self._prefill_policy.update_replicas(
                    list(self._prefill_replicas.values())
                )
            else:
                self._prefill_policy = self._scheduling_policy(
                    list(self._prefill_replicas.values())
                )
            logger.info(
                f"Added prefill actor {replica_uid}, total prefill actors: {len(self._prefill_replicas)}"
            )
        else:
            logger.debug(f"Prefill actor {replica_uid} already exists, skipping")

    async def add_decode_actor(
        self,
        replica_uid: str,
        actor: xo.ActorRefType["ModelActor"],
    ):
        """添加 Decode Actor"""
        if self._decode_replicas.get(replica_uid) != actor:
            self._decode_replicas[replica_uid] = actor
            # 更新调度策略的副本列表
            if self._decode_policy:
                self._decode_policy.update_replicas(
                    list(self._decode_replicas.values())
                )
            else:
                self._decode_policy = self._scheduling_policy(
                    list(self._decode_replicas.values())
                )
            logger.info(
                f"Added decode actor {replica_uid}, total decode actors: {len(self._decode_replicas)}"
            )
        else:
            logger.debug(f"Decode actor {replica_uid} already exists, skipping")

    async def remove_prefill_actor(self, replica_uid: str):
        """移除 Prefill Actor"""
        if replica_uid in self._prefill_replicas:
            del self._prefill_replicas[replica_uid]
            # 更新调度策略的副本列表
            if self._prefill_replicas and self._prefill_policy:
                self._prefill_policy.update_replicas(
                    list(self._prefill_replicas.values())
                )
            elif self._prefill_replicas:
                self._prefill_policy = self._scheduling_policy(
                    list(self._prefill_replicas.values())
                )
            else:
                self._prefill_policy = None
            logger.info(
                f"Removed prefill actor {replica_uid}, remaining prefill actors: {len(self._prefill_replicas)}"
            )
        else:
            logger.warning(f"Prefill actor {replica_uid} not found, skipping")

    async def remove_decode_actor(self, replica_uid: str):
        """移除 Decode Actor"""
        if replica_uid in self._decode_replicas:
            del self._decode_replicas[replica_uid]
            # 更新调度策略的副本列表
            if self._decode_replicas and self._decode_policy:
                self._decode_policy.update_replicas(
                    list(self._decode_replicas.values())
                )
            elif self._decode_replicas:
                self._decode_policy = self._scheduling_policy(
                    list(self._decode_replicas.values())
                )
            else:
                self._decode_policy = None
            logger.info(
                f"Removed decode actor {replica_uid}, remaining decode actors: {len(self._decode_replicas)}"
            )
        else:
            logger.warning(f"Decode actor {replica_uid} not found, skipping")

    def get_prefill_actor(self, replica_uid: str) -> xo.ActorRefType["ModelActor"]:
        """获取指定的 Prefill Actor"""
        return self._prefill_replicas.get(replica_uid)

    def get_decode_actor(self, replica_uid: str) -> xo.ActorRefType["ModelActor"]:
        """获取指定的 Decode Actor"""
        return self._decode_replicas.get(replica_uid)

    def has_prefill_replica(self, replica_uid: str) -> bool:
        """检查 Prefill 副本是否存在"""
        return replica_uid in self._prefill_replicas

    def has_decode_replica(self, replica_uid: str) -> bool:
        """检查 Decode 副本是否存在"""
        return replica_uid in self._decode_replicas

    def get_all_replica_uids(self) -> dict:
        """获取所有副本 UID"""
        return {
            "prefill": list(self._prefill_replicas.keys()),
            "decode": list(self._decode_replicas.keys()),
        }

    def __repr__(self) -> str:
        return f"PDModelActor(Prefill: {len(self._prefill_replicas)}, Decode: {len(self._decode_replicas)})"

    async def decrease_serve_count(self):
        # The stream wrapper releases its selected decode replica's slot.
        # API callers still invoke this compatibility method on the router.
        pass

    @log_async(logger=logger)
    async def free_prefill_model_cache(self, request_id: str):
        """释放prefill模型缓存"""
        logger.debug(
            f"[PDModelActor] Free prefill model cache for request {request_id}"
        )
        if request_id in self._request_set:
            self._request_set.remove(request_id)
            if self._prefill_replicas:
                await asyncio.gather(
                    *[
                        model.free_model_cache(request_id)
                        for model in self._prefill_replicas.values()
                    ],
                    return_exceptions=True,
                )
        else:
            logger.warning(
                f"[request {request_id}] Prefill model cache has been freed already"
            )
            return

    async def is_vllm_backend(self):
        return True

    async def _infer(self, method, inputs, *args, **kwargs):
        if not self._prefill_policy or not self._decode_policy:
            from .model import ModelNotReadyError

            raise ModelNotReadyError(
                "Both prefill and decode replicas must be available"
            )
        kwargs = dict(kwargs)
        request_id = str(kwargs.get("request_id") or uuid.uuid4().hex)
        kwargs["request_id"] = request_id
        if request_id in self._request_set:
            raise ValueError(f"Request {request_id} is already running")
        if args and args[0] is not None and not isinstance(args[0], dict):
            raise TypeError("Generation config must be a dict or None")
        prefill = self._prefill_policy.schedule()
        decode = self._decode_policy.schedule()
        prefill_args = list(copy.deepcopy(args))
        if not prefill_args or prefill_args[0] is None:
            prefill_args = [{}] + prefill_args[1:]
        prefill_args[0]["max_tokens"] = 1
        prefill_args[0]["stream"] = False
        prefill_kwargs = copy.deepcopy(kwargs)
        if isinstance(prefill_kwargs.get("raw_params"), dict):
            prefill_kwargs["raw_params"].update(max_tokens=1, stream=False)
        self._request_set.add(request_id)
        try:
            result = await getattr(prefill, method)(
                inputs, *prefill_args, **prefill_kwargs
            )
            if hasattr(result, "__aiter__"):
                async for _ in result:
                    pass
            if request_id not in self._request_set:
                raise asyncio.CancelledError(f"PD request {request_id} was aborted")
            await decode.set_unpin_handler(self._model_uid, request_id, self.address)
            result = await getattr(decode, method)(inputs, *args, **kwargs)
        except BaseException:
            await self.free_prefill_model_cache(request_id)
            raise
        if not hasattr(result, "__aiter__"):
            await self.free_prefill_model_cache(request_id)
            return result

        async def stream():
            try:
                async for chunk in result:
                    yield chunk
            finally:
                try:
                    if hasattr(result, "aclose"):
                        await result.aclose()
                    elif hasattr(result, "destroy"):
                        await result.destroy()
                finally:
                    try:
                        await decode.decrease_serve_count()
                    finally:
                        await self.free_prefill_model_cache(request_id)

        return stream()

    @xo.generator
    async def generate(self, prompt: str, *args, **kwargs):
        return await self._infer("generate", prompt, *args, **kwargs)

    @xo.generator
    async def chat(self, messages, *args, **kwargs):
        return await self._infer("chat", messages, *args, **kwargs)

    async def abort_request(self, request_id, block_duration=30):
        try:
            results = await asyncio.gather(
                *[
                    model.abort_request(request_id, block_duration)
                    for model in [
                        *self._prefill_replicas.values(),
                        *self._decode_replicas.values(),
                    ]
                ],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return "DONE" if "DONE" in results else "NOT_FOUND"
        finally:
            await self.free_prefill_model_cache(request_id)

    async def get_pd_info(self):
        """获取PD分离信息"""
        return {
            "model_uid": self._model_uid,
            "prefill_count": len(self._prefill_replicas),
            "decode_count": len(self._decode_replicas),
            "prefill_replica_uids": list(self._prefill_replicas.keys()),
            "decode_replica_uids": list(self._decode_replicas.keys()),
            "prefill_policy": str(self._prefill_policy),
            "decode_policy": str(self._decode_policy),
        }
