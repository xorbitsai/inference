# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Keep Supervisor RPC admission separate from a replaceable control process."""

import asyncio
import copy
import logging
import threading
import time
from typing import Any, Optional

import xoscar as xo
from xoscar.context import get_context

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1

# These reads resolve routes or poll the current Router configuration. They
# remain available while management drains and the target rebuilds its registry.
INFERENCE_READS = {
    "get_model",
    "describe_model",
    "describe_running_model",
    "get_model_status",
    "list_models",
    "list_virtual_models",
    "resolve_token_router_runtime",
    "get_token_router_config_after",
}

# These reports update volatile liveness/observations, not model ownership or
# desired configuration. Queuing them would trigger reconnects or expire routes.
LIVENESS_CALLS = {
    "receive_heartbeat",
    "report_worker_status",
    "heartbeat_token_router_instance",
    "heartbeat_token_router_node",
}

# Let callbacks and pending configuration ACKs finish before the final snapshot.
# ACKs arriving after that snapshot must instead reach the selected target.
DRAIN_CALLBACKS = {
    "add_worker",
    "remove_worker",
    "get_worker_resource_policy",
    "worker_handoff_protocol",
    "can_recover_model",
    "mark_replica_dead",
    "register_pd_replica",
    "unregister_pd_replica",
    "call_collective_manager",
    "restore_model_reload_config",
    "plan_worker_relaunch",
    "report_worker_relaunch",
    "prepare_worker_handoff",
    "commit_worker_handoff",
    "confirm_worker_handoff",
    "abort_worker_handoff",
    "ack_token_router_config",
}


class SupervisorGateway(xo.StatelessActor):
    """Retain the public actor identity and forward the original RPC envelope."""

    def __init__(self):
        super().__init__()
        self.target: Any = None
        self.phase = "frozen"
        self.changed = asyncio.Condition()
        self.controls = 0
        self.readers: dict = {}
        self._terminal_callbacks: set[asyncio.Task] = set()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        async def forward(*args, **kwargs):
            return await self.__on_receive__((name, 0, args, kwargs))

        return forward

    async def __on_receive__(self, message):
        method = message[0]
        if method in {
            "prepare_supervisor_switch",
            "select_supervisor",
            "release_supervisor",
            "drain_supervisor_reads",
        }:
            return await super().__on_receive__(message)
        if method == "mark_replica_dead" and self.phase == "frozen":
            # This is a one-shot fact, not a cancellable management request.
            # Keep it until the selected owner can apply it, even if the
            # Worker's bounded RPC times out while snapshot/restore is running.
            task = asyncio.create_task(self._forward(message))
            self._terminal_callbacks.add(task)
            task.add_done_callback(self._terminal_callback_done)
            return await asyncio.shield(task)
        return await self._forward(message)

    def _terminal_callback_done(self, task):
        self._terminal_callbacks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "Failed to deliver dead-replica notification", exc_info=task.exception()
            )

    async def __pre_destroy__(self):
        tasks = tuple(self._terminal_callbacks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _forward(self, message):
        method = message[0]
        reading = method in INFERENCE_READS or method in LIVENESS_CALLS
        async with self.changed:
            await self.changed.wait_for(
                lambda: self.target is not None
                and (
                    reading
                    or self.phase == "open"
                    or (self.phase == "closing" and method in DRAIN_CALLBACKS)
                )
            )
            target = self.target
            if reading:
                self.readers[target.address] = self.readers.get(target.address, 0) + 1
            else:
                self.controls += 1
        try:
            return await get_context().send(target, message)
        finally:
            async with self.changed:
                if reading:
                    self.readers[target.address] -= 1
                else:
                    self.controls -= 1
                self.changed.notify_all()

    async def prepare_supervisor_switch(self, final=True):
        async with self.changed:
            self.phase = "closing"
            await self.changed.wait_for(lambda: self.controls == 0)
            if final:
                self.phase = "frozen"

    async def select_supervisor(self, target):
        async with self.changed:
            self.target = target
            self.changed.notify_all()

    async def release_supervisor(self):
        async with self.changed:
            self.phase = "open"
            self.changed.notify_all()

    async def drain_supervisor_reads(self, address):
        async with self.changed:
            await self.changed.wait_for(lambda: self.readers.get(address, 0) == 0)
            self.readers.pop(address, None)


class SupervisorMaintenance:
    """Quiesce background maintenance across the actor and health-check loops."""

    def __init__(self):
        self.lock = threading.Lock()
        self.paused = False
        self.active = 0

    def begin(self) -> bool:
        with self.lock:
            if self.paused:
                return False
            self.active += 1
            return True

    def end(self) -> None:
        with self.lock:
            self.active -= 1

    def pause(self) -> None:
        with self.lock:
            self.paused = True

    def resume(self) -> None:
        with self.lock:
            self.paused = False

    def idle(self) -> bool:
        with self.lock:
            return self.active == 0


# Actor references point at the stable service pool or at retained Workers.
# No task, lock, model object, interpreter code, or database handle is copied.
ROUTING_FIELDS = (
    "_pd_model_mapping",
    "_pd_roles",
    "_xavier_cache_mapping",
    "_xavier_source_mapping",
    "_block_tracker_mapping",
    "_collective_manager_mapping",
    "_model_reload_status",
    "_unexpected_down_replicas",
    "_autostart_model_states",
    "_token_router_runtime_cursors",
    "_worker_status",
    "_worker_metadata",
    "_worker_model_gpu_memory",
    "_worker_model_gpu_memory_update_time",
)


class SupervisorUpgradeState:
    def __init__(self, supervisor: Any):
        self.supervisor = supervisor
        self.maintenance = SupervisorMaintenance()
        self._router_snapshot_time: Optional[float] = None

    def _require_idle(self):
        owner = self.supervisor
        manager = getattr(owner, "_worker_relaunch_manager", None)
        if (
            owner._workers_launching
            or owner._initial_model_launches
            or any(not task.done() for task in owner._async_launch_tasks)
            or owner._model_reload_tasks
            or owner._cache_uid_to_task
            or owner._autostart_lock.locked()
            or any(
                owner._ownership_registry().protected(address)
                for address in owner._ownership_registry().pending
            )
            or (
                manager is not None
                and any(not task.done() for task in manager.tasks.values())
            )
        ):
            raise RuntimeError(
                "Supervisor has an active launch, download, model reload, or Worker relaunch; retry after it finishes"
            )

    async def pause(self):
        owner = self.supervisor
        self.maintenance.pause()
        try:
            while not self.maintenance.idle():
                await asyncio.sleep(0.01)
            self._require_idle()
            task = owner._autostart_task
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        except BaseException:
            self.resume()
            raise

    def snapshot(self):
        if not self.maintenance.paused or not self.maintenance.idle():
            raise RuntimeError("Supervisor maintenance is not paused")
        self._require_idle()
        owner = self.supervisor
        from .worker_handoff import CustomModelState

        return {
            "protocol": PROTOCOL_VERSION,
            "local_deployment": owner._local_deployment,
            "local_worker_address": owner._local_worker_address,
            "captured_at": time.time(),
            "workers": list(owner._worker_address_to_worker),
            "incarnations": dict(owner._ownership_registry().incarnations),
            "replicas": {
                uid: [
                    ref.address
                    for ref in (refs if isinstance(refs, (tuple, list)) else [refs])
                ]
                for uid, refs in owner._replica_model_uid_to_worker.items()
                if not uid.endswith("-rank0")
            },
            "coordinators": {
                uid: ref.address
                for uid, ref in owner._replica_model_uid_to_worker.items()
                if uid.endswith("-rank0")
            },
            "shards": {
                uid: {shard: ref.address for shard, ref in shards.items()}
                for uid, shards in owner._replica_model_uid_to_worker_shards.items()
            },
            "routes": copy.deepcopy(
                {name: getattr(owner, name) for name in ROUTING_FIELDS}
            ),
            "router_instances": copy.deepcopy(owner._token_router_registry._instances),
            "system_settings": dict(owner._system_settings),
            "uptime": owner._uptime,
            "custom_models": CustomModelState.capture(owner),
        }

    def resume(self):
        self.refresh_liveness()
        self.maintenance.resume()
        self.supervisor._schedule_autostart()

    def refresh_liveness(self):
        from ..constants import XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD

        now = time.time()
        for status in self.supervisor._worker_status.values():
            status.update_time = now
            status.failure_remaining_count = XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD
        if self._router_snapshot_time is not None:
            # Preserve each runtime's age at capture, including offline ones.
            # Only the target's frozen preparation time is excluded; reports
            # keep reaching the source throughout the drain and rollback.
            elapsed = max(0, now - self._router_snapshot_time)
            for instance in self.supervisor._token_router_registry._instances.values():
                instance["last_heartbeat"] += elapsed
            self._router_snapshot_time = None

    async def restore(self, snapshot):
        if snapshot.get("protocol") != PROTOCOL_VERSION:
            raise ValueError("Unsupported Supervisor handoff protocol")
        owner = self.supervisor
        from ..extensions import get_extensions
        from .worker import WorkerActor
        from .worker_handoff import CustomModelState

        owner._local_deployment = snapshot.get("local_deployment")
        owner._local_worker_address = snapshot.get("local_worker_address")

        CustomModelState.restore(owner, snapshot["custom_models"])

        async def inspect_worker(address):
            worker = await xo.actor_ref(address=address, uid=WorkerActor.default_uid())
            incarnation, models, resources = await asyncio.gather(
                xo.wait_for(worker.worker_incarnation(), 10),
                xo.wait_for(worker.list_models(), 10),
                xo.wait_for(worker.get_extension_resources(), 10),
            )
            expected = snapshot["incarnations"].get(address)
            if expected is not None and incarnation != expected:
                raise RuntimeError(
                    "Worker ownership changed while preparing Supervisor"
                )
            if set((resources or {}).get("extensions", {})) != {
                item.name for item in get_extensions()
            }:
                raise ValueError("Worker and Supervisor extensions must match")
            expected_uids = {
                uid
                for uid, addresses in snapshot["replicas"].items()
                if address in addresses
            }
            if set(models) != expected_uids:
                raise RuntimeError(
                    "Worker model inventory changed while preparing Supervisor"
                )
            coordinators = [
                uid
                for uid, endpoint in snapshot["coordinators"].items()
                if endpoint == address
            ]
            # Rank-zero actors have no model spec and are intentionally absent
            # from list_models(). Validate their retained actor handles instead.
            await asyncio.gather(
                *(xo.wait_for(worker.get_model(uid), 10) for uid in coordinators)
            )
            replicas = []
            for uid in models:
                addresses = snapshot["replicas"][uid]
                shards = snapshot["shards"].get(uid, {})
                shard = next(
                    (rank for rank, endpoint in shards.items() if endpoint == address),
                    addresses.index(address),
                )
                replicas.append(
                    {
                        "replica_model_uid": uid,
                        "n_worker": len(addresses),
                        "shard": shard,
                    }
                )
            return (
                address,
                worker,
                {
                    "incarnation": incarnation,
                    "resources": resources,
                    "replicas": replicas,
                    "coordinators": coordinators,
                },
            )

        workers = await asyncio.gather(
            *(inspect_worker(address) for address in snapshot["workers"])
        )
        owner._initialize_worker_ownership()
        observed: dict = {}
        observed_coordinators = {}
        for address, worker, state in workers:
            owner._worker_address_to_worker[address] = worker
            owner._ownership_registry().incarnations[address] = state["incarnation"]
            owner._extension_worker_resources[address] = state["resources"]
            owner._rebuild_worker_replica_state(worker, state["replicas"])
            for replica in state["replicas"]:
                uid = replica["replica_model_uid"]
                observed.setdefault(uid, []).append(address)
            for uid in state["coordinators"]:
                owner._replica_model_uid_to_worker[uid] = worker
                observed_coordinators[uid] = address
        if {uid: sorted(addresses) for uid, addresses in observed.items()} != {
            uid: sorted(addresses) for uid, addresses in snapshot["replicas"].items()
        }:
            raise RuntimeError(
                "Worker model inventory changed while preparing Supervisor"
            )
        if observed_coordinators != snapshot["coordinators"]:
            raise RuntimeError(
                "Worker coordinator inventory changed while preparing Supervisor"
            )
        for name in ROUTING_FIELDS:
            setattr(owner, name, snapshot["routes"][name])
        owner._token_router_registry._instances = snapshot["router_instances"]
        self._router_snapshot_time = snapshot["captured_at"]
        owner._system_settings = snapshot["system_settings"]
        from .system_settings_store import SystemSettings, apply_system_settings

        apply_system_settings(SystemSettings.from_dict(owner._system_settings))
        owner._uptime = snapshot["uptime"]
        owner._supervisor_restored = True
