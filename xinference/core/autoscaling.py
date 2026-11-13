# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Admission and replica policy for autoscaled Embedding deployments."""

import asyncio
import json
import os
import time
from collections import deque
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

from .._compat import PYDANTIC_V2, BaseModel

if PYDANTIC_V2:
    from pydantic.v1 import StrictInt, StrictStr, root_validator, validator
else:
    from pydantic import StrictInt, StrictStr, root_validator, validator

import xoscar as xo
from async_timeout import timeout

from .control_session import ControlSessionRejected
from .exceptions import ModelNotReadyError
from .replica_config import DeviceConfig, ReplicaConfig

if TYPE_CHECKING:
    from .supervisor import SupervisorActor

POOL_ENV = "XINFERENCE_AUTOSCALING_GPU_POOLS"
CONTROL_INTERVAL = 0.25


class EmbeddingOverloadedError(RuntimeError):
    """The request could not obtain an embedding slot before its deadline."""


class AutoscalingPolicy(BaseModel):
    class Config:
        extra = "forbid"

    gpu_pool: Optional[StrictStr] = None
    min_replicas: StrictInt = 1
    max_replicas: Optional[StrictInt] = None
    concurrency_per_replica: StrictInt = 32
    queue_capacity: StrictInt = 256
    queue_timeout_ms: StrictInt = 10000
    scale_up_wait_ms: StrictInt = 250
    idle_timeout_seconds: StrictInt = 300

    @validator(
        "min_replicas",
        "max_replicas",
        "concurrency_per_replica",
        "queue_capacity",
        "queue_timeout_ms",
        "scale_up_wait_ms",
        "idle_timeout_seconds",
    )
    def positive(cls, value):
        if value is not None and value < 1:
            raise ValueError("Value must be a positive integer")
        return value

    @root_validator(skip_on_failure=True)
    def validate_bounds(cls, values):
        minimum = values["min_replicas"]
        maximum = values["max_replicas"]
        if maximum is not None and maximum < minimum:
            raise ValueError("max_replicas must be at least min_replicas")
        if values["scale_up_wait_ms"] >= values["queue_timeout_ms"]:
            raise ValueError("scale_up_wait_ms must be below queue_timeout_ms")
        return values


class GPUPoolConfig(BaseModel):
    __root__: Dict[StrictStr, Dict[StrictStr, List[StrictInt]]]

    @root_validator(skip_on_failure=True)
    def validate_devices(cls, values):
        for name, workers in values["__root__"].items():
            if not name.strip() or not workers:
                raise ValueError(
                    "Each GPU pool must have a name and at least one worker"
                )
            for address, devices in workers.items():
                if not address.strip() or not devices or min(devices) < 0:
                    raise ValueError(
                        "Each GPU pool worker must have non-negative GPU indexes"
                    )
                if len(set(devices)) != len(devices):
                    raise ValueError("Duplicate GPU index in pool")
        return values


def gpu_pools() -> Dict[str, Dict[str, List[int]]]:
    return GPUPoolConfig.parse_obj(json.loads(os.getenv(POOL_ENV, "{}"))).__root__


async def select_placements(
    owner: "SupervisorActor", pool_name: Optional[str], count: int
) -> List[ReplicaConfig]:
    pools = gpu_pools()
    if pool_name is not None and pool_name not in pools:
        raise ValueError(f"Unknown GPU pool: {pool_name}")
    pool = None if pool_name is None else pools[pool_name]
    workers = [
        (address, worker)
        for address, worker in list(owner._worker_address_to_worker.items())
        if pool is None or address in pool
    ]
    snapshots = await asyncio.gather(
        *(
            asyncio.wait_for(worker.get_gpu_allocation_status(), 2)
            for _, worker in workers
        ),
        return_exceptions=True,
    )
    placements = []
    for (address, worker), allocation in sorted(
        zip(workers, snapshots), key=lambda item: item[0][0]
    ):
        if (
            isinstance(allocation, BaseException)
            or owner._worker_address_to_worker.get(address) is not worker
        ):
            continue
        devices = set(allocation["total"])
        if pool is not None:
            devices.intersection_update(pool[address])
        occupied = {
            int(device)
            for key in ("models", "user_specified")
            for device, models in allocation[key].items()
            if models
        }
        for device in sorted(devices - occupied):
            placements.append(
                ReplicaConfig(
                    devices=[DeviceConfig(worker_ip=address, n_gpu=1, gpu_idx=[device])]
                )
            )
            if len(placements) == count:
                return placements
    return placements


class EmbeddingAutoscaler:
    def __init__(self, policy: AutoscalingPolicy):
        self.policy = policy
        self.waiters: Deque[Tuple[float, asyncio.Event]] = deque()
        self.last_pressure = time.monotonic()
        self.retry_after = 0.0
        self.reason: Optional[str] = None
        self.operation: Optional[str] = None
        self.task: Optional[asyncio.Task] = None
        self.closed = False
        self.saturated = False
        self.draining: set[int] = set()

    def close(self):
        self.closed = True
        for _, turn in self.waiters:
            turn.set()

    def oldest_wait(self):
        if self.waiters:
            return time.monotonic() - self.waiters[0][0]
        return 0.0

    async def admit(
        self, owner: "SupervisorActor", model_uid: str, reservation_id: str
    ):
        if len(self.waiters) >= self.policy.queue_capacity:
            raise EmbeddingOverloadedError("Embedding request queue is full")
        waiter = (time.monotonic(), asyncio.Event())
        self.waiters.append(waiter)
        if len(self.waiters) == 1:
            waiter[1].set()
        try:
            async with timeout(self.policy.queue_timeout_ms / 1000):
                return await self._admit_head(
                    owner, model_uid, reservation_id, waiter[1]
                )
        except asyncio.TimeoutError:
            raise EmbeddingOverloadedError(
                "Timed out waiting for an embedding replica"
            ) from None
        finally:
            self.waiters.remove(waiter)
            if self.waiters:
                self.waiters[0][1].set()
            else:
                self.saturated = False

    async def _admit_head(self, owner, model_uid, reservation_id, turn):
        await turn.wait()
        while not self.closed:
            info = owner._model_uid_to_replica_info.get(model_uid)
            ids = [] if info is None else info.active_replica_ids
            # Start at the next replica in the normal round-robin order.
            if ids:
                first = next(info.scheduler)
                offset = ids.index(first)
                ids = ids[offset:] + ids[:offset]
            saturated = False
            for replica_id in ids:
                if replica_id in self.draining:
                    continue
                try:
                    async with timeout(1):
                        model = await owner._get_replica_model(model_uid, replica_id)
                        if await model.reserve_embedding_request(
                            reservation_id, self.policy.concurrency_per_replica
                        ):
                            self.saturated = False
                            return model
                        saturated = True
                except (ModelNotReadyError, asyncio.TimeoutError):
                    continue
                except (ConnectionError, OSError, xo.ActorNotExist, xo.ServerClosed):
                    continue
            self.saturated = saturated
            if saturated:
                self.last_pressure = time.monotonic()
            await asyncio.sleep(0.02)
        raise ModelNotReadyError(f"Embedding deployment {model_uid} was stopped")

    async def run(self, owner: "SupervisorActor", model_uid: str) -> None:
        while not self.closed:
            await asyncio.sleep(CONTROL_INTERVAL)
            if self.closed:
                break
            try:
                await owner.reconcile_embedding_replicas(model_uid)
            except ControlSessionRejected:
                # A retained supervisor must wait for controller authority.
                self.operation = None

    def status(self, ready):
        return {
            "policy": self.policy.dict(),
            "ready_replicas": ready,
            "queued_requests": len(self.waiters),
            "oldest_wait_ms": round(self.oldest_wait() * 1000),
            "operation": self.operation,
            "reason": self.reason,
        }
