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
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

import xoscar as xo

from ..constants import XINFERENCE_HEALTH_CHECK_TIMEOUT
from .supervisor_journal import LocalSupervisorJournal, supervisor_journal
from .worker_relaunch import SupervisorRelaunchMixin
from .worker_state import WorkerJournalError

logger = logging.getLogger(__name__)

_UNSPECIFIED_WORKER_GENERATION = object()


async def call_worker_route(
    owner: Any,
    uid: str,
    ref: Any,
    method: str,
    *,
    rpc_timeout: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """Retry an idempotent read if its original owner was replaced in flight."""
    from ..constants import XINFERENCE_GET_MODEL_RPC_TIMEOUT
    from .rpc_context import actor_call

    if rpc_timeout is None:
        rpc_timeout = XINFERENCE_GET_MODEL_RPC_TIMEOUT

    async def call(worker: Any) -> Any:
        task = asyncio.create_task(actor_call(worker, method, **kwargs))
        try:
            return await xo.wait_for(task, rpc_timeout)
        except asyncio.CancelledError:
            # Cancelling xo.wait_for does not cancel its actor call. Forward the
            # cancellation without waiting for a stalled worker to acknowledge it.
            task.cancel()
            task.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )
            raise

    try:
        return await call(ref)
    except Exception:
        current = owner._replica_model_uid_to_worker.get(uid)
        if isinstance(current, (tuple, list)):
            current = current[0] if current else None
        if current is None or current == ref:
            raise
        return await call(current)


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
                            **(
                                {"worker_id": record["worker_id"]}
                                if "worker_id" in record
                                else {}
                            ),
                            "models": record.get("models", []),
                            "committed": record["committed"],
                            "target_process": record.get("target_process"),
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
                if registration and registry.protected(worker_address):
                    state = registry.pending[worker_address]
                    if incarnation != state["key"][5]:
                        raise RuntimeError("Worker handoff source incarnation changed")
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

    async def _remove_dead_worker(
        self, address, expected_ref, expected_generation=_UNSPECIFIED_WORKER_GENERATION
    ):
        """Health checks share the ownership gate with registration and handoff."""
        if expected_generation is _UNSPECIFIED_WORKER_GENERATION:
            expected_generation = getattr(self, "_worker_metadata_generation", {}).get(
                address
            )
        loop = getattr(self, "_actor_loop", None)
        if loop is not None and loop is not asyncio.get_running_loop():
            future = asyncio.run_coroutine_threadsafe(
                self._remove_dead_worker(address, expected_ref, expected_generation),
                loop,
            )
            return await asyncio.wrap_future(future)
        registry = self._ownership_registry()
        async with registry.lock(address):
            if (
                (
                    self._worker_address_to_worker.get(address) is not None
                    and self._worker_address_to_worker[address] != expected_ref
                )
                or getattr(self, "_worker_metadata_generation", {}).get(address)
                != expected_generation
                or registry.protected(address)
            ):
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

    def _check_worker_name(self, worker_id, address, *, source=None):
        if not worker_id:
            return
        owner = self._pending_worker_names.get(worker_id)
        if owner is None:
            registry = self._ownership_registry()
            for other_source, state in registry.pending.items():
                if (
                    other_source != source
                    and state.get("worker_id") == worker_id
                    and registry.protected(other_source)
                ):
                    registered = self._extension_worker_resources.get(address, {})
                    if (
                        other_source == address
                        and address in self._worker_address_to_worker
                        and registered.get("worker_id") == worker_id
                    ):
                        continue
                    owner = state["key"][3]
                    break
        if owner is None:
            owner = next(
                (
                    registered_address
                    for registered_address, resources in self._extension_worker_resources.copy().items()
                    if registered_address not in (address, source)
                    and registered_address in self._worker_address_to_worker
                    and resources.get("worker_id") == worker_id
                ),
                None,
            )
        if owner is not None:
            raise RuntimeError(
                f"Worker name {worker_id!r} is already registered or being "
                f"registered at {owner}; refusing registration from {address}"
            )

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
            from .worker import WorkerActor

            target = await xo.actor_ref(
                address=snapshot.target, uid=WorkerActor.default_uid()
            )
            resources = await xo.wait_for(target.get_extension_resources(), timeout)
            try:
                target_process = await xo.wait_for(
                    target.worker_handoff_process_identity(snapshot.target_incarnation),
                    5,
                )
            except (AttributeError, asyncio.TimeoutError):
                target_process = None
            if snapshot.target in self._worker_address_to_worker or any(
                state["key"][3] == snapshot.target
                # A commit may be awaiting cache rebind as its deadline passes.
                and (registry.protected(source) or registry.lock(source).locked())
                for source, state in registry.pending.items()
            ):
                raise ValueError("Replacement worker is already registered or reserved")
            expected = self._extension_worker_resources.get(snapshot.source)
            if set(resources.get("extensions", {})) != set(
                (expected or {}).get("extensions", {})
            ):
                raise ValueError("Worker extensions must match during replacement")
            worker_id = resources.get("worker_id")
            self._check_worker_name(worker_id, snapshot.target, source=snapshot.source)
            replicas = [
                self._get_model_uid_and_replica_index(uid)
                for uid in snapshot.state["_model_uid_to_model"]
            ]
            registry.set_pending(
                snapshot.source,
                {
                    "key": snapshot.transfer_key,
                    "extensions": sorted(resources.get("extensions", {})),
                    "worker_id": worker_id,
                    "target_process": target_process,
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
        from .worker import WorkerActor
        from .worker_handoff import HandoffPending

        registry = self._ownership_registry()
        async with registry.lock(source):
            state = registry.pending.get(source)
            if not state or state["key"][1] != ticket or state["committed"]:
                return
            key = state["key"]
            registry.check(source, key[5])
            if not registry.protected(source):
                target = key[3]
                # A v1 source reports completion only after transferring its
                # subprocesses. Require evidence from the original target
                # before restoring an expired reservation.
                if (
                    not state.get("recovered")
                    and source not in self._worker_address_to_worker
                ):
                    raise RuntimeError("Worker handoff reservation expired")
                worker = await xo.wait_for(
                    xo.actor_ref(address=target, uid=WorkerActor.default_uid()), 5
                )
                if not await xo.wait_for(worker.worker_handoff_ready(key), 5):
                    try:
                        in_progress = await xo.wait_for(
                            worker.worker_handoff_in_progress(key), 5
                        )
                    except AttributeError:
                        # Older v1 Workers only expose the readiness check.
                        in_progress = (
                            await xo.wait_for(worker.worker_incarnation(), 5) == key[6]
                        )
                    if in_progress:
                        raise HandoffPending("Replacement Worker is still transferring")
                    raise RuntimeError("Worker handoff reservation expired")
                if target in self._worker_address_to_worker or any(
                    other_source != source
                    and reservation["key"][3] == target
                    and (
                        registry.protected(other_source)
                        or registry.lock(other_source).locked()
                    )
                    for other_source, reservation in registry.pending.items()
                ):
                    raise RuntimeError("Worker handoff reservation expired")
                self._check_worker_name(state.get("worker_id"), target, source=source)
            registry.set_pending(source, dict(state, committed=True))

    def get_pending_worker_handoff(self, source):
        state = self._ownership_registry().pending.get(source)
        if state is None:
            raise KeyError(source)
        key = state["key"]
        return {
            "ticket": key[1],
            "source": source,
            "target": key[3],
            "target_incarnation": key[6],
            "target_process": state.get("target_process"),
            "committed": state["committed"],
        }

    async def recover_confirmed_worker_handoff(
        self, source, ticket, target_incarnation
    ):
        registry = self._ownership_registry()
        state = registry.pending.get(source)
        if (
            state is None
            or not state["committed"]
            or state["key"][1] != ticket
            or state["key"][6] != target_incarnation
        ):
            raise RuntimeError("No matching confirmed worker handoff")
        if not await xo.wait_for(
            self._remove_abandoned_committed_handoff(
                source, state, operator_verified=True
            ),
            XINFERENCE_HEALTH_CHECK_TIMEOUT,
        ):
            raise RuntimeError("Worker handoff changed during recovery")
        self._schedule_autostart()
        return {"recovered": True}

    async def _handoff_target_still_online(self, key):
        from .worker import WorkerActor

        async def probe():
            worker = await xo.actor_ref(address=key[3], uid=WorkerActor.default_uid())
            return await worker.worker_incarnation() == key[6]

        try:
            return await xo.wait_for(probe(), 5)
        except (OSError, xo.ActorNotExist, xo.ServerClosed):
            # Without positive evidence, recovery relies on the operator's
            # verification of the exact process on the target host.
            return False

    async def _remove_abandoned_committed_handoff(
        self, source, expected, *, operator_verified=False
    ):
        from .worker_handoff import (
            worker_handoff_process_alive,
            worker_handoff_process_exited,
        )

        registry = self._ownership_registry()
        async with registry.lock(source):
            state = registry.pending.get(source)
            if state is not expected:
                return False
            identity = state.get("target_process")
            key = state["key"]
            if operator_verified:
                if worker_handoff_process_alive(
                    identity
                ) or await self._handoff_target_still_online(key):
                    raise RuntimeError("Original replacement Worker is still running")
            elif not worker_handoff_process_exited(identity):
                return False
            # The confirmed transfer made the original source retire. Remove
            # its stale routes before releasing the journaled reservation.
            await self._handle_dead_worker(
                source, reconcile_models=state.get("models", ())
            )
            self._worker_status.pop(source, None)
            self._worker_address_to_worker.pop(source, None)
            self._extension_worker_resources.pop(source, None)
            self._discard_worker_metadata(source)
            self._clear_worker_model_gpu_memory(source)
            pending = dict(registry.pending)
            del pending[source]
            retired = dict(registry.retired)
            for address, incarnation in ((source, key[5]), (key[3], key[6])):
                retired[address] = retired.get(address, set()) | {incarnation}
            registry.checkpoint(pending=pending, retired=retired)
            return True

    async def _try_remove_abandoned_committed_handoff(
        self, source, state, *, timeout=None
    ):
        return await xo.wait_for(
            self._remove_abandoned_committed_handoff(source, state),
            XINFERENCE_HEALTH_CHECK_TIMEOUT if timeout is None else timeout,
        )

    async def _check_abandoned_worker_handoffs(self):
        registry = getattr(self, "_worker_ownership", None)
        if registry is None or not registry.pending:
            return
        loop = getattr(self, "_actor_loop", None)
        if loop is not None and loop is not asyncio.get_running_loop():
            future = asyncio.run_coroutine_threadsafe(
                self._check_abandoned_worker_handoffs(), loop
            )
            await asyncio.wrap_future(future)
            return

        sources = [
            source for source, state in registry.pending.items() if state["committed"]
        ]
        cursor = getattr(self, "_abandoned_handoff_scan_next", None)
        start = sources.index(cursor) if cursor in sources else 0
        deadline = time.monotonic() + XINFERENCE_HEALTH_CHECK_TIMEOUT
        for offset in range(len(sources)):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            index = (start + offset) % len(sources)
            source = sources[index]
            self._abandoned_handoff_scan_next = sources[(index + 1) % len(sources)]
            state = registry.pending.get(source)
            if state is None or not state["committed"]:
                continue
            try:
                if await self._try_remove_abandoned_committed_handoff(
                    source, state, timeout=remaining
                ):
                    self._schedule_autostart()
            except Exception:
                logger.warning(
                    "Failed to clean abandoned handoff for Worker %s",
                    source,
                    exc_info=True,
                )

    async def _wait_for_worker_handoffs(self, model_uid):
        from .worker_handoff import HandoffPending

        registry = getattr(self, "_worker_ownership", None)
        if registry is None:
            return False
        for source, state in list(registry.pending.items()):
            if model_uid not in state.get("models", ()):
                continue
            if state["committed"]:
                try:
                    await self._try_remove_abandoned_committed_handoff(source, state)
                except Exception:
                    logger.warning(
                        "Failed to clean abandoned handoff for Worker %s",
                        source,
                        exc_info=True,
                    )
                # Recheck the model after cleanup or concurrent publication.
                return True
            if registry.protected(source):
                return True
            if not state.get("recovered"):
                continue
            try:
                await self.confirm_worker_handoff(source, state["key"][1])
            except (ConnectionRefusedError, xo.ActorNotExist):
                continue
            except (OSError, xo.ServerClosed, HandoffPending):
                # A transient transport error does not establish that the
                # target has lost its transferred subprocesses.
                return True
            except (RuntimeError, ValueError):
                continue
            # The replacement coordinator retries publication after reconnecting.
            # Recheck even if it published while confirmation was in flight.
            return True
        return False

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
            if not registry.protected(snapshot.source):
                raise RuntimeError("Worker handoff reservation expired")
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
            if not registry.protected(snapshot.source):
                raise RuntimeError("Worker handoff reservation expired")
            if set(resources.get("extensions", {})) != set(state["extensions"]):
                raise ValueError("Worker extensions must match during replacement")

            worker_id = resources.get("worker_id")
            if "worker_id" in state and state["worker_id"] != worker_id:
                raise ValueError("Replacement worker name changed during handoff")
            self._check_worker_name(worker_id, snapshot.target, source=snapshot.source)
            if worker_id:
                self._pending_worker_names[worker_id] = snapshot.target
            try:
                cache_tracker = getattr(self, "_cache_tracker_ref", None)
                if cache_tracker is not None:
                    await xo.wait_for(
                        cache_tracker.rebind_worker(snapshot.source, snapshot.target), 5
                    )
            finally:
                if worker_id:
                    self._pending_worker_names.pop(worker_id, None)

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
