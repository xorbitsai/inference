# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Keep Supervisor RPC admission separate from a replaceable control process."""

import asyncio
import copy
import threading
from typing import Any

import xoscar as xo
from xoscar.context import get_context

PROTOCOL_VERSION = 1

# These calls only resolve an existing model route. They must remain available
# while control operations drain and the replacement rebuilds its registry.
INFERENCE_READS = {
    "get_model",
    "describe_model",
    "describe_running_model",
    "get_model_status",
    "resolve_token_router_runtime",
}

# A management RPC may call a Worker which calls back into the Supervisor.
# Admit those callbacks until the outstanding management RPCs have completed.
WORKER_CALLBACKS = {
    "add_worker",
    "remove_worker",
    "report_worker_status",
    "receive_heartbeat",
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
        reading = method in INFERENCE_READS
        async with self.changed:
            await self.changed.wait_for(
                lambda: self.target is not None
                and (
                    reading
                    or self.phase == "open"
                    or (self.phase == "closing" and method in WORKER_CALLBACKS)
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

    def _require_idle(self):
        owner = self.supervisor
        manager = getattr(owner, "_worker_relaunch_manager", None)
        if (
            owner._workers_launching
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
            "workers": list(owner._worker_address_to_worker),
            "incarnations": dict(owner._ownership_registry().incarnations),
            "replicas": {
                uid: [
                    ref.address
                    for ref in (refs if isinstance(refs, (tuple, list)) else [refs])
                ]
                for uid, refs in owner._replica_model_uid_to_worker.items()
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
        self.maintenance.resume()
        self.supervisor._schedule_autostart()

    async def restore(self, snapshot):
        if snapshot.get("protocol") != PROTOCOL_VERSION:
            raise ValueError("Unsupported Supervisor handoff protocol")
        owner = self.supervisor
        from ..extensions import get_extensions
        from .worker import WorkerActor
        from .worker_handoff import CustomModelState

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
                },
            )

        workers = await asyncio.gather(
            *(inspect_worker(address) for address in snapshot["workers"])
        )
        owner._initialize_worker_ownership()
        observed: dict = {}
        for address, worker, state in workers:
            owner._worker_address_to_worker[address] = worker
            owner._ownership_registry().incarnations[address] = state["incarnation"]
            owner._extension_worker_resources[address] = state["resources"]
            owner._rebuild_worker_replica_state(worker, state["replicas"])
            for replica in state["replicas"]:
                uid = replica["replica_model_uid"]
                observed.setdefault(uid, []).append(address)
                if uid.endswith("-rank0"):
                    owner._replica_model_uid_to_worker[uid] = worker
        if {uid: sorted(addresses) for uid, addresses in observed.items()} != {
            uid: sorted(addresses) for uid, addresses in snapshot["replicas"].items()
        }:
            raise RuntimeError(
                "Worker model inventory changed while preparing Supervisor"
            )
        for name in ROUTING_FIELDS:
            setattr(owner, name, snapshot["routes"][name])
        owner._token_router_registry._instances = snapshot["router_instances"]
        owner._system_settings = snapshot["system_settings"]
        from .system_settings_store import SystemSettings, apply_system_settings

        apply_system_settings(SystemSettings.from_dict(owner._system_settings))
        owner._uptime = snapshot["uptime"]
        owner._supervisor_restored = True
