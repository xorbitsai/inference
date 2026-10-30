# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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

"""Serialize worker ownership changes and fence messages from retired owners."""

import asyncio
import functools
import time
from pathlib import Path
from typing import Any, Dict, Set

import xoscar as xo

from .supervisor_journal import LocalSupervisorJournal, supervisor_journal
from .worker_relaunch import SupervisorRelaunchMixin
from .worker_state import WorkerJournalError


async def call_worker_route(
    owner: Any, uid: str, ref: Any, method: str, **kwargs: Any
) -> Any:
    """Retry an idempotent read if its original owner was replaced in flight."""
    from ..constants import XINFERENCE_GET_MODEL_RPC_TIMEOUT
    from .rpc_context import actor_call

    try:
        return await xo.wait_for(
            actor_call(ref, method, **kwargs), XINFERENCE_GET_MODEL_RPC_TIMEOUT
        )
    except Exception:
        current = owner._replica_model_uid_to_worker.get(uid)
        if isinstance(current, (tuple, list)):
            current = current[0] if current else None
        if current is None or current == ref:
            raise
        return await xo.wait_for(
            actor_call(current, method, **kwargs), XINFERENCE_GET_MODEL_RPC_TIMEOUT
        )


class WorkerOwnership:
    """Keep transfer identities durable without serializing model objects or credentials."""

    def __init__(self, path=None, *, store=None):
        self.path = Path(path) if path is not None else None
        self.store = (
            store
            if store is not None
            else (LocalSupervisorJournal(self.path) if self.path is not None else None)
        )
        self.incarnations: Dict[str, str] = {}
        self.retired: Dict[str, Set[str]] = {}
        self.pending: dict = {}
        self.completed: dict = {}
        self._locks: dict = {}
        if self.store is not None:
            data = self.store.load({"version": 1})
            if not isinstance(data, dict) or data.get("version") != 1:
                raise WorkerJournalError("Unsupported Worker ownership journal")
            self.retired = {
                key: set(value) for key, value in data.get("retired", {}).items()
            }
            self.completed = {
                key: tuple(value) for key, value in data.get("completed", {}).items()
            }
            for source, record in data.get("pending", {}).items():
                self.pending[source] = dict(
                    record,
                    key=tuple(record["key"]),
                    deadline=time.monotonic()
                    + max(0, record["expires_at"] - time.time()),
                    recovered=True,
                )

    def checkpoint(self, *, pending=None, retired=None, completed=None):
        pending = self.pending if pending is None else pending
        retired = self.retired if retired is None else retired
        completed = self.completed if completed is None else completed
        if self.store is not None:
            self.store.save(
                {
                    "version": 1,
                    "pending": {
                        source: {
                            "key": record["key"],
                            "extensions": record["extensions"],
                            "models": record.get("models", []),
                            "committed": record["committed"],
                            "expires_at": time.time()
                            + record["deadline"]
                            - time.monotonic(),
                        }
                        for source, record in pending.items()
                    },
                    "retired": {key: sorted(value) for key, value in retired.items()},
                    "completed": completed,
                },
            )
        self.pending, self.retired, self.completed = pending, retired, completed

    def discard(self, source):
        if source in self.pending:
            pending = dict(self.pending)
            pending.pop(source)
            self.checkpoint(pending=pending)

    def set_pending(self, source, record):
        pending = dict(self.pending)
        pending[source] = record
        self.checkpoint(pending=pending)

    def lock(self, address):
        return self._locks.setdefault(address, asyncio.Lock())

    def check(self, address, incarnation, registration=False):
        if address in self.retired and (
            incarnation is None or incarnation in self.retired[address]
        ):
            raise RuntimeError("Worker ownership was transferred")
        current = self.incarnations.get(address)
        if (
            current
            and (not registration or incarnation is None)
            and incarnation != current
        ):
            raise RuntimeError("Stale worker incarnation")

    def protected(self, address):
        state = self.pending.get(address)
        return state is not None and (
            state["committed"] or time.monotonic() < state["deadline"]
        )


def shared_registry_operation(method):
    """Serialize shared catalog writes with the Worker's handoff reservation."""

    @functools.wraps(method)
    async def checked(self, *args, **kwargs):
        address = getattr(self, "_shared_worker_address", None)
        if address is None:
            return await method(self, *args, **kwargs)
        registry = self._ownership_registry()
        async with registry.lock(address):
            if address in registry.pending:
                raise RuntimeError(
                    "Worker handoff is in progress; retry after it finishes"
                )
            return await method(self, *args, **kwargs)

    return checked


def worker_message(*, registration=False):
    def decorate(method):
        @functools.wraps(method)
        async def checked(self, worker_address, *args, incarnation=None, **kwargs):
            registry = self._ownership_registry()
            async with registry.lock(worker_address):
                registry.check(worker_address, incarnation, registration)
                if registration and incarnation:
                    from .worker import WorkerActor

                    endpoint = await xo.actor_ref(
                        address=worker_address, uid=WorkerActor.default_uid()
                    )
                    if (
                        await xo.wait_for(endpoint.worker_incarnation(), 5)
                        != incarnation
                    ):
                        raise RuntimeError("Stale worker registration")
                result = await method(self, worker_address, *args, **kwargs)
                if registration and incarnation:
                    previous = registry.incarnations.get(worker_address)
                    if previous is not None and previous != incarnation:
                        registry.retired.setdefault(worker_address, set()).add(previous)
                    registry.incarnations[worker_address] = incarnation
                return result

        return checked

    return decorate


class SupervisorHandoffMixin(SupervisorRelaunchMixin):
    _replica_model_uid_to_worker: Dict[str, Any]

    def _initialize_worker_ownership(self):
        from ..constants import XINFERENCE_HOME

        path = Path(XINFERENCE_HOME) / "worker-handoffs.json"
        self._worker_ownership = WorkerOwnership(
            path, store=supervisor_journal(self, "worker_ownership", path)
        )

    def _ownership_registry(self):
        registry = getattr(self, "_worker_ownership", None)
        if registry is None:
            registry = self._worker_ownership = WorkerOwnership()
        return registry

    def worker_handoff_protocol(self):
        from .worker_handoff import PROTOCOL_VERSION

        return PROTOCOL_VERSION

    async def _remove_dead_worker(self, address, expected_ref):
        """Health checks share the ownership gate with registration and handoff."""
        loop = getattr(self, "_actor_loop", None)
        if loop is not None and loop is not asyncio.get_running_loop():
            future = asyncio.run_coroutine_threadsafe(
                self._remove_dead_worker(address, expected_ref), loop
            )
            return await asyncio.wrap_future(future)
        registry = self._ownership_registry()
        async with registry.lock(address):
            if (
                self._worker_address_to_worker.get(address) is not None
                and self._worker_address_to_worker[address] != expected_ref
            ) or registry.protected(address):
                return False
            registry.discard(address)
            await self._handle_dead_worker(address)
            self._worker_status.pop(address, None)
            self._worker_address_to_worker.pop(address, None)
            self._extension_worker_resources.pop(address, None)
            self._discard_worker_metadata(address)
            self._clear_worker_model_gpu_memory(address)
            return True

    def can_recover_model(self, worker_address: str, model_uid: str) -> bool:
        refs = self._replica_model_uid_to_worker.get(model_uid)
        if refs is None:
            return True
        if not isinstance(refs, (tuple, list)):
            refs = [refs]
        return any(ref.address == worker_address for ref in refs)

    async def prepare_worker_handoff(self, snapshot, timeout):
        registry = self._ownership_registry()
        async with registry.lock(snapshot.source):
            registry.check(snapshot.source, snapshot.source_incarnation)
            if snapshot.source not in self._worker_address_to_worker:
                raise ValueError("Source worker is not registered")
            existing = registry.pending.get(snapshot.source)
            if existing and not registry.protected(snapshot.source):
                registry.discard(snapshot.source)
                existing = None
            if existing:
                if existing["key"] == snapshot.transfer_key:
                    if not existing["committed"]:
                        registry.set_pending(
                            snapshot.source,
                            dict(existing, deadline=time.monotonic() + timeout),
                        )
                    return True
                raise RuntimeError("Another worker replacement is in progress")
            if snapshot.target in self._worker_address_to_worker or any(
                state["key"][3] == snapshot.target
                for state in registry.pending.values()
            ):
                raise ValueError("Replacement worker is already registered or reserved")
            from .worker import WorkerActor

            target = await xo.actor_ref(
                address=snapshot.target, uid=WorkerActor.default_uid()
            )
            resources = await xo.wait_for(target.get_extension_resources(), timeout)
            expected = self._extension_worker_resources.get(snapshot.source)
            if set(resources.get("extensions", {})) != set(
                (expected or {}).get("extensions", {})
            ):
                raise ValueError("Worker extensions must match during replacement")
            replicas = [
                self._get_model_uid_and_replica_index(uid)
                for uid in snapshot.state["_model_uid_to_model"]
            ]
            registry.set_pending(
                snapshot.source,
                {
                    "key": snapshot.transfer_key,
                    "extensions": sorted(resources.get("extensions", {})),
                    "models": sorted(
                        {replica[0] for replica in replicas if replica is not None}
                    ),
                    "deadline": time.monotonic() + timeout,
                    "committed": False,
                },
            )
            return True

    async def abort_worker_handoff(self, source, ticket):
        registry = self._ownership_registry()
        async with registry.lock(source):
            state = registry.pending.get(source)
            if state and state["key"][1] == ticket:
                if state["committed"]:
                    raise RuntimeError(
                        "Committed worker ownership cannot be rolled back"
                    )
                registry.discard(source)

    async def confirm_worker_handoff(self, source, ticket):
        registry = self._ownership_registry()
        async with registry.lock(source):
            state = registry.pending.get(source)
            if state and state["key"][1] == ticket:
                registry.set_pending(source, dict(state, committed=True))

    async def commit_worker_handoff(self, snapshot):
        from .worker import WorkerActor

        registry = self._ownership_registry()
        async with registry.lock(snapshot.source):
            if snapshot.ticket in registry.completed:
                if registry.completed[snapshot.ticket] != (
                    snapshot.source,
                    snapshot.target,
                ):
                    raise RuntimeError("Worker handoff ticket changed")
                return True
            state = registry.pending.get(snapshot.source)
            if not state or state["key"] != snapshot.transfer_key:
                raise RuntimeError("No matching worker reservation")
            if (
                not state.get("recovered")
                and snapshot.source not in self._worker_address_to_worker
            ):
                raise RuntimeError("Source worker is no longer registered")
            registry.check(snapshot.source, snapshot.source_incarnation)
            worker = await xo.actor_ref(
                address=snapshot.target, uid=WorkerActor.default_uid()
            )
            if not await worker.worker_handoff_ready(snapshot.transfer_key):
                raise RuntimeError("Replacement worker is not ready")
            resources = await worker.get_extension_resources()
            if set(resources.get("extensions", {})) != set(state["extensions"]):
                raise ValueError("Worker extensions must match during replacement")

            cache_tracker = getattr(self, "_cache_tracker_ref", None)
            if cache_tracker is not None:
                await xo.wait_for(
                    cache_tracker.rebind_worker(snapshot.source, snapshot.target), 5
                )

            pending = dict(registry.pending)
            pending.pop(snapshot.source)
            retired = dict(registry.retired)
            retired[snapshot.source] = retired.get(snapshot.source, set()) | {
                snapshot.source_incarnation
            }
            completed = dict(
                registry.completed,
                **{snapshot.ticket: (snapshot.source, snapshot.target)},
            )
            registry.checkpoint(pending=pending, retired=retired, completed=completed)
            if getattr(self, "_local_worker_address", None) == snapshot.source:
                self._local_worker_address = snapshot.target
            if getattr(self, "_shared_worker_address", None) == snapshot.source:
                self._shared_worker_address = None

            # No awaits from validation to publication: route replacement and
            # old-owner fencing become visible together on the actor loop.
            def replace(ref):
                return worker if ref.address == snapshot.source else ref

            for uid, refs in self._replica_model_uid_to_worker.items():
                if isinstance(refs, (list, tuple)):
                    self._replica_model_uid_to_worker[uid] = tuple(
                        replace(ref) for ref in refs
                    )
                else:
                    self._replica_model_uid_to_worker[uid] = replace(refs)
            for shards in self._replica_model_uid_to_worker_shards.values():
                for shard, ref in shards.items():
                    shards[shard] = replace(ref)
            for info in self._model_uid_to_replica_info.values():
                for index, refs in info.replica_to_worker_refs.items():
                    info.replica_to_worker_refs[index] = [replace(ref) for ref in refs]
            self._worker_address_to_worker.pop(snapshot.source, None)
            self._worker_address_to_worker[snapshot.target] = worker
            status = self._worker_status.pop(snapshot.source, None)
            if status is not None:
                status.update_time = time.time()
                self._worker_status[snapshot.target] = status
            self._extension_worker_resources.pop(snapshot.source, None)
            self._extension_worker_resources[snapshot.target] = resources
            self._discard_worker_metadata(snapshot.source)
            self._clear_worker_model_gpu_memory(snapshot.source)
            self._list_models_cache.pop(snapshot.source, None)
            self._reverse_ping_failures.pop(snapshot.source, None)
            self._invalidate_list_models_debounce_cache()
            registry.incarnations.pop(snapshot.source, None)
            registry.incarnations[snapshot.target] = snapshot.target_incarnation
            return True
