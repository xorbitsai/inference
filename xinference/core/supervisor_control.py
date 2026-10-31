# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Optional process-bound Supervisor admission and Worker control propagation.

This module does not elect a controller or persist another copy of ownership.
Each process keeps its original session, including queued/background commands.
Extensions own workflow ordering; this module supplies only local primitives.
"""

import asyncio
import contextlib
import inspect
import math
import uuid
from contextvars import ContextVar
from types import MethodType
from typing import Any, Optional, Tuple

from .control_invocation import ControlInvocation, invoke_control_hook
from .control_session import (
    CONTROL_CONTEXT_KEY,
    NATIVE_HANDOFF_RPCS,
    READ_ONLY_WORKER_RPCS,
    ControlContext,
    ControlSessionRejected,
    WorkerControlFence,
    _BoundWorkerControlMethod,
)
from .lifecycle import WorkerLifecycle, WorkerPhase
from .supervisor_upgrade import INFERENCE_READS, LIVENESS_CALLS

MODEL_SERVICE_FIELDS = (
    "_pd_model_mapping",
    "_pd_roles",
    "_xavier_cache_mapping",
    "_xavier_source_mapping",
    "_block_tracker_mapping",
    "_collective_manager_mapping",
)
_service_host: ContextVar[Optional[Tuple[Any, str]]] = ContextVar(
    "xinference_model_service_host", default=None
)
_supervisor_calls: ContextVar[tuple] = ContextVar("supervisor_calls", default=())


@contextlib.contextmanager
def _supervisor_call_scope(control):
    """Mark nested business helpers separately from transport admission."""
    token = _supervisor_calls.set(
        (*_supervisor_calls.get(), (control, asyncio.current_task()))
    )
    try:
        yield
    finally:
        _supervisor_calls.reset(token)


def model_service_address(owner):
    current = _service_host.get()
    return (
        current[1]
        if current is not None and current[0] is owner
        else owner.control_address
    )


READS = (
    INFERENCE_READS
    | LIVENESS_CALLS
    | {
        "default_uid",
        "worker_handoff_protocol",
        "get_worker_resource_policy",
        "get_status",
        "get_workers_info",
        "get_supervisor_info",
        "record_metrics",
        "get_autostart_config",
        "get_autostart_model_summary",
        "get_cluster_device_info",
        "get_builtin_prompts",
        "get_builtin_families",
        "get_devices_count",
        "get_cluster_metrics_data",
        "list_model_registrations",
        "get_model_registration",
        "recommend_model",
        "query_engines_by_model_name",
        "get_model_versions",
        "get_model_version_count",
        "get_launch_builtin_model_progress",
        "get_launch_builtin_model_progress_details",
        "get_cache_builtin_model_progress_details",
        "list_model_downloads",
        "get_instance_info",
        "get_replica_statuses",
        "get_instance_count",
        "get_model_reload_config",
        "get_model_reload_status",
        "is_local_deployment",
        "list_cached_models",
        "list_deletable_models",
        "list_virtual_envs",
        "list_virtual_env_packages",
        "get_progress",
        "get_progress_snapshot",
        "supervisor_upgrade_info",
    }
)
SERVICE_CALLS = {
    "controller_session_status",
    "supervisor_extension_operation",
    "add_worker",
    "activate_supervisor",
    "prepare_supervisor_activation",
}
BLOCKED = NATIVE_HANDOFF_RPCS | {
    "pause_supervisor",
    "resume_supervisor",
    "restore_supervisor",
    "get_supervisor_snapshot",
    "relaunch_worker",
}


class ControlledWorkerRef:
    """Transport-transparent reference; all unclassified methods require control."""

    def __init__(self, control, ref):
        self._control, self._ref = control, ref

    def __getattr__(self, name):
        method = getattr(self._ref, name)
        if (
            name.startswith("_")
            or name in READ_ONLY_WORKER_RPCS
            or not callable(method)
        ):
            return method
        command = self._control.capture()

        def call(*args, **kwargs):
            if CONTROL_CONTEXT_KEY in kwargs:
                raise ControlSessionRejected("caller_control_override")

            async def invoke():
                async with self._control.fence.command(command):

                    async def execute(args=args, kwargs=kwargs):
                        if CONTROL_CONTEXT_KEY in kwargs:
                            raise ControlSessionRejected("caller_control_override")
                        return await method(
                            *args, **{**kwargs, CONTROL_CONTEXT_KEY: command.to_dict()}
                        )

                    return await invoke_control_hook(
                        self._control.authority,
                        "around_worker_call",
                        ControlInvocation(
                            self._ref.address, name, command, args, kwargs
                        ),
                        execute,
                    )

            return invoke()

        return call

    def __reduce__(self):
        # Never serialize a provider, connection or process context to clients
        # or into an upgrade snapshot. Restored references must be rebound.
        return self._ref.__reduce__()

    def __eq__(self, other):
        return self._ref == (
            other._ref if isinstance(other, ControlledWorkerRef) else other
        )

    def __hash__(self):
        return hash(self._ref)


class SupervisorControl:
    def __init__(self, owner, authority):
        self.owner, self.authority, self.session = owner, authority, authority.session
        self.fence = WorkerControlFence(authority, WorkerLifecycle(active=True))
        self.fence.session = self.session
        self._workers = {}
        self.prepared = False
        self.has_served = False

    @property
    def ready(self):
        return self.fence.ready

    def capture(self):
        return self.fence.bound_context() or ControlContext(
            self.session, uuid.uuid4().hex
        )

    def worker(self, ref):
        if isinstance(ref, ControlledWorkerRef):
            ref = ref._ref
        key = (ref.address, ref.uid)
        if key not in self._workers:
            self._workers[key] = ControlledWorkerRef(self, ref)
        return self._workers[key]

    async def check(self):
        with self.fence.local_scope(self.capture()):
            await self.fence.checkpoint()

    async def register(self, ref):
        await self.fence._validate(self.session, ("fencing", "reconciling", "active"))
        if callable(getattr(self.authority, "worker_registered", None)):
            await self.invoke("worker_registered", ref, timeout=60)
        return self.worker(ref)

    async def invoke(self, name, *args, timeout=10):
        method = getattr(self.authority, name)
        try:
            if inspect.iscoroutinefunction(method):
                return await asyncio.wait_for(method(*args), timeout)
            return await asyncio.wait_for(asyncio.to_thread(method, *args), timeout)
        except ControlSessionRejected:
            raise
        except Exception:
            raise ControlSessionRejected("authority_unavailable") from None

    async def operation(self, operation, payload):
        if not isinstance(operation, str) or not isinstance(payload, dict):
            raise ValueError("Expected an extension operation and object payload")
        method = getattr(self.authority, "execute", None)
        if not inspect.iscoroutinefunction(method):
            raise ControlSessionRejected("extension_operation_unsupported")
        await self.fence._validate(self.session, ("fencing", "reconciling", "active"))
        try:
            return await method(operation, payload)
        except ControlSessionRejected:
            raise
        except Exception:
            raise ControlSessionRejected("extension_operation_failed") from None


class SupervisorControlPort:
    """Process-local extension capabilities; never a remote recovery workflow.

    No method discovers endpoints, contacts Workers, selects observations, changes
    authority phase, or combines these primitives into a workflow.
    """

    def __init__(self, owner):
        self._owner = owner

    def __reduce__(self):
        raise TypeError("Supervisor control ports are process-local")

    def _control(self):
        control = self._owner._controller_control
        if control is None:
            raise ControlSessionRejected("authority_not_configured")
        return control

    def status(self):
        return self._control().fence.status()

    async def pause(self, timeout=30):
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Expected a finite positive drain timeout")
        control = self._control()
        await control.fence._validate(
            control.session, ("fencing", "reconciling", "active")
        )
        # Check and close admission without yielding. Launch orchestration can
        # span untracked child tasks between individually admitted Worker RPCs.
        # Refuse a busy pause before fencing their readiness or rollback calls.
        self._owner._upgrade_state._require_idle()
        if control.fence.lifecycle._tasks:
            raise RuntimeError(
                "Supervisor has active management work; retry after it finishes"
            )
        control.fence.ready, control.prepared = False, False
        try:
            await control.fence.lifecycle.drain(timeout)
        finally:
            if control.fence.lifecycle.phase == WorkerPhase.DRAINING:
                control.fence.lifecycle.phase = WorkerPhase.ACTIVE
        await self._owner._upgrade_state.pause()
        control.prepared = True

    async def restore_worker(self, ref, replicas, incarnation, coordinators=()):
        control = self._control()
        await control.fence._validate(
            control.session, ("fencing", "reconciling", "active")
        )
        if not control.prepared or control.ready:
            raise ControlSessionRejected("control_not_paused")
        ref = control.worker(ref)
        owner = self._owner
        owner._worker_address_to_worker[ref.address] = ref
        owner._ownership_registry().incarnations[ref.address] = incarnation
        await owner._restore_worker_inventory(ref, replica_states=replicas)
        for uid in coordinators:
            if not isinstance(uid, str) or not uid.endswith("-rank0"):
                raise ValueError("Invalid coordinator identity")
            owner._replica_model_uid_to_worker[uid] = ref

    async def resume(self):
        control = self._control()
        await control.fence._validate(control.session, ("active",))
        if not control.prepared or control.ready:
            raise ControlSessionRejected("control_not_paused")
        try:
            await self._owner._activate_supervisor()
            self._owner._upgrade_state.resume()
        except BaseException:
            control.fence.ready = False
            raise
        control.prepared = False
        # Publish only after all activation awaits and maintenance resume have
        # succeeded. Background launches are scheduled after this atomic commit.
        control.fence.ready = control.has_served = True
        self._owner._schedule_autostart()
        return self.status()

    def settings(self):
        return dict(self._owner._system_settings)

    def catalog(self):
        from .worker_handoff import CustomModelState

        return CustomModelState.capture(self._owner)

    def workers(self):
        return list(self._owner._worker_address_to_worker.values())

    def restore_catalog(self, payload):
        from .worker_handoff import CustomModelState

        CustomModelState.restore(self._owner, payload, replace=True)

    def restore_settings(self, payload):
        from .system_settings_store import SystemSettings, apply_system_settings

        parsed = SystemSettings.from_dict(payload)
        self._owner._system_settings = parsed.to_dict()
        apply_system_settings(parsed)

    async def call(self, method, *args, **kwargs):
        if not isinstance(method, str) or method.startswith("_"):
            raise ValueError("Expected a public Supervisor method")
        control = self._control()
        async with control.fence.command(control.capture()):
            with _supervisor_call_scope(control):
                value = getattr(self._owner, method)(*args, **kwargs)
                return await value if inspect.isawaitable(value) else value

    async def with_service_address(self, address, call):
        if not isinstance(address, str) or not address:
            raise ValueError("Expected an actor address")
        token = _service_host.set((self._owner, address))
        try:
            return await call()
        finally:
            _service_host.reset(token)

    def service_address(self):
        return model_service_address(self._owner)

    def model_services(self, model_uid):
        return {
            name: getattr(self._owner, name, {}).get(model_uid)
            for name in MODEL_SERVICE_FIELDS
        }

    def restore_model_services(self, model_uid, services):
        if not isinstance(services, dict) or set(services) - set(MODEL_SERVICE_FIELDS):
            raise ValueError("Invalid model service state")
        for name, value in services.items():
            mapping = getattr(self._owner, name)
            if value is None:
                mapping.pop(model_uid, None)
            else:
                mapping[model_uid] = value

    def close_admission(self):
        self._control().fence.ready = False


def bind_supervisor_control(owner, name, method):
    control = owner._controller_control

    def guarded(self, *args, **kwargs):
        # Remote dispatch may hold the fence before resolving the outer method.
        nested = (
            control.fence.bound_context() is not None
            and (control, asyncio.current_task()) in _supervisor_calls.get()
        )
        supplied = kwargs.pop(CONTROL_CONTEXT_KEY, None)
        if name in BLOCKED:
            raise ControlSessionRejected("handoff_requires_session_transfer")
        # Nested synchronous helpers preserve their ordinary return type.
        if nested and not inspect.iscoroutinefunction(method):
            control.fence.check_local()
            return method(*args, **kwargs)
        command = (
            control.capture()
            if supplied is None
            else ControlContext.from_dict(supplied)
        )

        async def run():
            async with control.fence.command(command):

                async def execute(args=args, kwargs=kwargs):
                    value = method(*args, **kwargs)
                    return await value if inspect.isawaitable(value) else value

                with _supervisor_call_scope(control):
                    if nested:
                        return await execute()
                    return await invoke_control_hook(
                        control.authority,
                        "around_supervisor_call",
                        ControlInvocation(
                            self.control_address, name, command, args, kwargs
                        ),
                        execute,
                    )

        return run()

    return MethodType(_BoundWorkerControlMethod(owner, method, guarded), owner)
