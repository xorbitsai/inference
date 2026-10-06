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
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

import xoscar as xo

from .rpc_context import actor_call
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
        transport_backend: str = "xavier",
        model_engine: str = "vllm",
    ):
        super().__init__()
        # Prefill request map, used to skip the timeout task for specific request id.
        self._request_set: Set[str] = set()
        self._direct_transfers: dict[str, dict] = {}

        self._model_uid = model_uid
        self._transport_backend = transport_backend
        self._model_engine = (model_engine or "vllm").lower()
        self._direct_handoff = transport_backend == "xavier"

        # 使用字典存储副本：{replica_uid: actor_ref}
        self._prefill_replicas: Dict[str, xo.ActorRefType["ModelActor"]] = {}
        self._decode_replicas: Dict[str, xo.ActorRefType["ModelActor"]] = {}
        self._sglang_bootstrap: dict[str, dict] = {}

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
            if self._model_engine == "sglang" and self._transport_backend == "nixl":
                self._sglang_bootstrap[replica_uid] = await actor_call(
                    actor, "get_sglang_pd_bootstrap"
                )
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
            self._sglang_bootstrap.pop(replica_uid, None)
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
        handoff = self._direct_transfers.pop(request_id, None)
        if handoff:
            try:
                if handoff.get("engine") == "sglang":
                    ref = await xo.actor_ref(
                        address=handoff["address"], uid=handoff["uid"]
                    )
                    if handoff.get("mode") == "gpu":
                        await ref.release(handoff["room"])
                    else:
                        await ref.release_handoff(handoff["ticket"])
                else:
                    await self._abandon_vllm_handoff(handoff)
            except Exception:
                logger.warning("Failed to abandon Xavier handoff", exc_info=True)
        if request_id in self._request_set:
            self._request_set.remove(request_id)
        else:
            logger.warning(
                f"[request {request_id}] Prefill model cache has been freed already"
            )
            return

    @staticmethod
    async def _abandon_vllm_handoff(handoff):
        from ..model.llm.vllm.xavier.transfer import TransferActor

        ref = await xo.actor_ref(
            address=handoff["address"],
            uid=f"{TransferActor.default_uid()}-{handoff['rank']}",
        )
        # D owns claimed tickets until its writes drain. Only reclaim
        # a handoff that was never accepted by D (router failure/abort).
        await ref.abandon_direct_gpu_v1(handoff["ticket"])

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
        if "generate_config" in kwargs:
            if args:
                raise TypeError("Generation config supplied twice")
            args = (kwargs.pop("generate_config"),)
        if args and args[0] is not None and not isinstance(args[0], dict):
            raise TypeError("Generation config must be a dict or None")
        if args and args[0] and args[0].get("n", 1) != 1:
            # Handoff leases cover one decoder, not parallel sampling children.
            raise ValueError("PD KV handoff currently requires n=1")
        if self._model_engine == "sglang":
            return await self._infer_sglang(method, inputs, args, kwargs, request_id)
        prefill = self._prefill_policy.schedule()
        decode = self._decode_policy.schedule()
        logger.debug(
            "PD route: request=%s prefill=%s decode=%s backend=%s",
            request_id,
            prefill.uid,
            decode.uid,
            self._transport_backend,
        )
        prefill_args = list(copy.deepcopy(args))
        if not prefill_args or prefill_args[0] is None:
            prefill_args = [{}] + prefill_args[1:]
        prefill_args[0]["max_tokens"] = 1
        prefill_args[0]["stream"] = False
        prefill_args[0]["n"] = 1
        prefill_args[0]["_pd_kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
        }
        prefill_kwargs = copy.deepcopy(kwargs)
        if isinstance(prefill_kwargs.get("raw_params"), dict):
            prefill_kwargs["raw_params"].update(max_tokens=1, stream=False)
        self._request_set.add(request_id)
        prefill_start = time.perf_counter()
        try:
            result = await actor_call(
                prefill, method, inputs, *prefill_args, **prefill_kwargs
            )
            if hasattr(result, "__aiter__"):
                async for _ in result:
                    pass
            logger.debug(
                "PD prefill complete: request=%s backend=%s elapsed_s=%.6f",
                request_id,
                self._transport_backend,
                time.perf_counter() - prefill_start,
            )
            payload = json.loads(result) if isinstance(result, (bytes, str)) else result
            transfer = (
                payload.get("_pd_kv_transfer_params")
                if isinstance(payload, dict)
                else None
            )
            if self._direct_handoff and isinstance(transfer, dict):
                handoff = transfer.get("xavier_direct") or transfer.get("sglang_xavier")
                if handoff:
                    self._direct_transfers[request_id] = handoff
            if not isinstance(transfer, dict) or not transfer.get("do_remote_prefill"):
                raise RuntimeError("PD prefill did not return KV transfer metadata")
            decode_args = list(copy.deepcopy(args))
            if not decode_args or decode_args[0] is None:
                decode_args = [{}] + decode_args[1:]
            decode_args[0]["_pd_kv_transfer_params"] = transfer
            args = tuple(decode_args)
            if request_id not in self._request_set:
                raise asyncio.CancelledError(f"PD request {request_id} was aborted")
            result = await actor_call(
                decode,
                method,
                inputs,
                *args,
                _rpc_operation_request_id=request_id,
                **kwargs,
            )
        except BaseException:
            await self.free_prefill_model_cache(request_id)
            raise
        if not hasattr(result, "__aiter__"):
            # A successful D response has already completed its handoff.
            self._direct_transfers.pop(request_id, None)
            await self.free_prefill_model_cache(request_id)
            return result

        async def stream():
            try:
                async for chunk in result:
                    yield chunk
                self._direct_transfers.pop(request_id, None)
            finally:
                try:
                    if hasattr(result, "aclose"):
                        await result.aclose()
                    elif hasattr(result, "destroy"):
                        await result.destroy()
                finally:
                    try:
                        await actor_call(
                            decode,
                            "decrease_serve_count",
                            _rpc_operation_request_id=request_id,
                        )
                    finally:
                        await self.free_prefill_model_cache(request_id)

        return stream()

    async def _infer_sglang(self, method, inputs, args, kwargs, request_id):
        """Native SGLang owns live P/D slots until the selected transport drains."""
        prefill = self._prefill_policy.schedule()
        decode = self._decode_policy.schedule()
        native = self._transport_backend == "nixl"
        bootstrap = (
            self._sglang_bootstrap[
                next(
                    uid for uid, ref in self._prefill_replicas.items() if ref == prefill
                )
            ]
            if native
            else dict(address=self.address, uid=f"xavier-cache-{self._model_uid}")
        )
        handoff = dict(
            engine="sglang",
            mode="nixl" if native else "gpu",
            room=uuid.uuid4().int % (2**63 - 1) + 1,
            **bootstrap,
        )
        transfer_key = "sglang_nixl" if native else "sglang_xavier"
        self._request_set.add(request_id)
        if not native:
            self._direct_transfers[request_id] = handoff
        prefill_args = list(copy.deepcopy(args)) or [{}]
        prefill_args[0] = dict(prefill_args[0] or {})
        prefill_args[0].update(
            max_tokens=1,
            stream=False,
            n=1,
            _pd_kv_transfer_params={"do_remote_decode": True, transfer_key: handoff},
        )
        decode_args = list(copy.deepcopy(args)) or [{}]
        decode_args[0] = dict(decode_args[0] or {})
        decode_args[0]["_pd_kv_transfer_params"] = {
            "do_remote_prefill": True,
            transfer_key: handoff,
        }
        prefill_kwargs = copy.deepcopy(kwargs)
        if isinstance(prefill_kwargs.get("raw_params"), dict):
            prefill_kwargs["raw_params"].update(max_tokens=1, stream=False)
        p_task = asyncio.create_task(
            actor_call(prefill, method, inputs, *prefill_args, **prefill_kwargs)
        )
        d_task = asyncio.create_task(
            actor_call(
                decode,
                method,
                inputs,
                *decode_args,
                _rpc_operation_request_id=request_id,
                **kwargs,
            )
        )
        result = None

        async def close_decode_stream():
            try:
                if hasattr(result, "aclose"):
                    await result.aclose()
                elif hasattr(result, "destroy"):
                    await result.destroy()
            finally:
                await actor_call(
                    decode,
                    "decrease_serve_count",
                    _rpc_operation_request_id=request_id,
                )

        async def stop():
            # Abort both native requests before cancellation so their schedulers
            # drain GPU work before reclaiming source or destination slots.
            await asyncio.gather(
                *(
                    actor_call(
                        replica,
                        "abort_request",
                        request_id,
                        30,
                        _rpc_operation_request_id=request_id,
                    )
                    for replica in (prefill, decode)
                ),
                return_exceptions=True,
            )
            p_task.cancel()
            d_task.cancel()
            await asyncio.gather(p_task, d_task, return_exceptions=True)
            await self.free_prefill_model_cache(request_id)

        try:
            await asyncio.wait({p_task, d_task}, return_when=asyncio.FIRST_COMPLETED)
            if p_task.done():
                await p_task
            result = await d_task
            if not hasattr(result, "__aiter__"):
                await p_task
                self._direct_transfers.pop(request_id, None)
                await self.free_prefill_model_cache(request_id)
                return result
        except BaseException:
            await stop()
            # The decode RPC can transfer stream ownership before the prefill
            # task fails, including when both tasks finish in the same turn.
            if d_task.done() and not d_task.cancelled() and d_task.exception() is None:
                result = d_task.result()
                if hasattr(result, "__aiter__"):
                    try:
                        await close_decode_stream()
                    except Exception:
                        logger.warning(
                            "Failed to close SGLang decode stream", exc_info=True
                        )
            raise

        async def stream():
            completed = False
            next_chunk = None
            try:
                iterator = result.__aiter__()
                while True:
                    next_chunk = asyncio.create_task(anext(iterator))
                    if not p_task.done():
                        await asyncio.wait(
                            {p_task, next_chunk}, return_when=asyncio.FIRST_COMPLETED
                        )
                    if p_task.done():
                        await p_task
                    try:
                        chunk = await next_chunk
                    except StopAsyncIteration:
                        break
                    # Native D output follows GPU transfer completion.
                    await p_task
                    yield chunk
                await p_task
                completed = True
                self._direct_transfers.pop(request_id, None)
            finally:
                if next_chunk is not None:
                    next_chunk.cancel()
                    await asyncio.gather(next_chunk, return_exceptions=True)
                try:
                    await close_decode_stream()
                finally:
                    if completed:
                        await self.free_prefill_model_cache(request_id)
                    else:
                        await stop()

        return stream()

    @xo.generator
    @log_async(logger=logger)
    async def generate(self, prompt: str, *args, **kwargs):
        return await self._infer("generate", prompt, *args, **kwargs)

    @xo.generator
    @log_async(logger=logger)
    async def chat(self, messages, *args, **kwargs):
        return await self._infer("chat", messages, *args, **kwargs)

    @log_async(logger=logger)
    async def abort_request(self, request_id, block_duration=30):
        try:
            results = await asyncio.gather(
                *[
                    actor_call(
                        model,
                        "abort_request",
                        request_id,
                        block_duration,
                        _rpc_operation_request_id=request_id,
                    )
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
            "transport_backend": self._transport_backend,
            "prefill_count": len(self._prefill_replicas),
            "decode_count": len(self._decode_replicas),
            "prefill_replica_uids": list(self._prefill_replicas.keys()),
            "decode_replica_uids": list(self._decode_replicas.keys()),
            "prefill_policy": str(self._prefill_policy),
            "decode_policy": str(self._decode_policy),
        }
