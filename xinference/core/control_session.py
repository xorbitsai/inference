# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Optional controller-session admission for retained Workers.

The authority is supplied by a process-local extension. It must validate against
the current authority on every call and raise on expiry or unavailability. This
module does not elect a controller, discover an endpoint, persist a second copy
of ownership, or retry a command. Installing a session first closes admission,
then drains the existing Worker lifecycle before acknowledging the new owner.

Callers attach ControlContext.to_dict() under CONTROL_CONTEXT_KEY on mutating
Worker RPCs; metadata is removed before invoking the business method. The
operation ID is correlation, not a deduplication or exactly-once receipt. There
is no automatic Supervisor propagation. Reads retain their ordinary arguments.
The factory hook is create_worker_control_authority(WorkerControlAuthorityContext).
Its validator must scope authority to the intended Worker/cluster, use bounded
I/O and raise on denial. Blocking validators run off the actor loop.

Worker/model process replacement is rejected in this optional mode until its
session/resource-transfer protocol is negotiated. Without a provider, existing
replacement and resource callbacks keep their legacy behavior. Local fence
status is diagnostic, not current lease proof or full service readiness.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import math
import re
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Tuple

PROTOCOL_VERSION = 1
CONTROL_CONTEXT_KEY = "_xinference_control_context"
# All other RPCs require explicit control context. New methods cannot silently
# become unguarded merely because a mutation list was not updated.
READ_ONLY_WORKER_RPCS = frozenset(
    {
        "ping",
        # Liveness never registers a Worker or changes its model ownership.
        "heartbeat",
        "default_uid",
        "get_devices_count",
        # ModelActors report inference metrics independently of controller ownership.
        "record_metrics",
        "get_model",
        "get_model_for_abort",
        "describe_model",
        "list_models",
        "get_model_status",
        "get_model_count",
        "get_launch_args",
        "get_model_launch_status",
        "get_model_reload_config",
        "get_model_reload_status",
        "is_model_vllm_backend",
        "get_gpu_allocation_status",
        "get_node_metadata",
        "get_workers_info",
        "get_extension_resources",
        "read_runtime_logs",
        "list_model_registrations",
        "get_model_registration",
        "query_engines_by_model_name",
        "get_model_recommendation_info",
        "list_cached_models",
        "list_deletable_models",
        "list_virtual_envs",
        "list_virtual_env_packages",
        "worker_incarnation",
        "worker_process_host",
        "model_generation_protocol",
        "worker_handoff_protocol",
        "worker_handoff_status",
        "worker_handoff_ready",
        "worker_replacement_safe_to_stop",
        "worker_package_upgrade_protocol",
        "worker_relaunch_state",
        "worker_relaunch_process",
    }
)
# These callbacks validate their model incarnation instead of controller state.
MODEL_DIAGNOSTIC_RPCS = frozenset({"report_model_oom"})
# Native Worker/model generation replacement transfers execution or schedules
# callbacks outside the current Worker operation. Session/resource transfer must
# be negotiated before this optional mode can support it. Legacy replacement is
# unchanged without an authority.
NATIVE_HANDOFF_RPCS = frozenset(
    {
        "check_worker_handoff_support",
        "prepare_worker_handoff",
        "abort_worker_handoff",
        "commit_worker_handoff",
        "finish_worker_handoff",
        "replace_worker",
        "launch_worker_replacement",
        "reload_model",
        "retire_model_generation_pool",
    }
)
_binding: ContextVar[Optional[Tuple[Any, "ControlContext"]]] = ContextVar(
    "xinference_worker_control", default=None
)


class _BoundWorkerControlMethod:
    def __init__(self, owner: Any, method: Any, dispatch: Any):
        self.__self__ = owner
        self.__wrapped__ = method
        self._dispatch = dispatch

    def __call__(self, owner: Any, *args, **kwargs):
        return self._dispatch(owner, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        method = object.__getattribute__(self, "__wrapped__")
        if (
            name in {"batch", "delay"}
            and getattr(
                object.__getattribute__(self, "__self__"), "_control_fence", None
            )
            is not None
        ):
            raise ControlSessionRejected("unsupported_batch")
        # Preserve the callable's interface, including descriptors/dynamic
        # attributes that functools.wraps cannot copy from its __dict__.
        return getattr(method, name)


def bind_worker_control_method(owner: Any, name: str, method: Any) -> Any:
    """Give LocalActorRef the same admission boundary as remote dispatch.

    Xoscar resolves local references with getattr(actor, name), bypassing
    __on_receive__. Keep synchronous nested calls synchronous, and keep the
    bound method's actor identity so StatelessActor remains lock-free.
    """

    def guarded(self, *args, **kwargs):
        fence = getattr(self, "_control_fence", None)
        supplied = kwargs.pop(CONTROL_CONTEXT_KEY, None)
        if fence is None:
            if supplied is not None:
                raise ControlSessionRejected("authority_not_configured")
            return method(*args, **kwargs)
        current = fence.bound_context()
        if supplied is not None:
            try:
                current = ControlContext.from_dict(supplied)
            except (TypeError, ValueError):
                raise ControlSessionRejected("invalid_context") from None
        if current is None:
            raise ControlSessionRejected("missing_context")
        if name in NATIVE_HANDOFF_RPCS:
            raise ControlSessionRejected("handoff_requires_session_transfer")
        # Only an explicitly bound, already admitted task may use nested
        # synchronous cleanup. Caller metadata always takes the normal gate.
        if supplied is None and getattr(method, "_management_cleanup", False) is True:
            fence.check_cleanup()
            return method(*args, **kwargs)
        if supplied is None and not inspect.iscoroutinefunction(method):
            fence.check_local()
            if asyncio.current_task() not in fence.lifecycle._tasks:
                raise ControlSessionRejected("missing_admission")
            return method(*args, **kwargs)

        async def admitted():
            async with fence.command(current):
                result = method(*args, **kwargs)
                return await result if inspect.isawaitable(result) else result

        return admitted()

    return MethodType(_BoundWorkerControlMethod(owner, method, guarded), owner)


class ControlSessionRejected(RuntimeError):
    """A stable diagnostic, never an underlying provider/connection exception."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Worker control rejected: {reason}")

    def __reduce__(self):
        return type(self), (self.reason,)


def _identifier(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        raise ValueError("Invalid controller-session identifier")


@dataclass(frozen=True)
class ControllerSession:
    cluster_id: str
    candidate_id: str
    owner_session: str
    term: int

    def __post_init__(self) -> None:
        for value in (self.cluster_id, self.candidate_id, self.owner_session):
            _identifier(value)
        if type(self.term) is not int or not 1 <= self.term < (1 << 63):
            raise ValueError("Invalid controller term")

    def to_dict(self) -> Dict[str, Any]:
        return {"version": PROTOCOL_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: Any) -> ControllerSession:
        if (
            not isinstance(value, dict)
            or set(value)
            != {"version", "cluster_id", "candidate_id", "owner_session", "term"}
            or type(value["version"]) is not int
            or value["version"] != PROTOCOL_VERSION
        ):
            raise ValueError("Unsupported controller-session protocol")
        return cls(**{key: item for key, item in value.items() if key != "version"})


@dataclass(frozen=True)
class ControlContext:
    session: ControllerSession
    operation_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.session, ControllerSession):
            raise ValueError("Invalid controller session")
        _identifier(self.operation_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "session": self.session.to_dict(),
            "operation_id": self.operation_id,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ControlContext:
        if (
            not isinstance(value, dict)
            or set(value) != {"version", "session", "operation_id"}
            or type(value["version"]) is not int
            or value["version"] != PROTOCOL_VERSION
        ):
            raise ValueError("Unsupported control-context protocol")
        return cls(ControllerSession.from_dict(value["session"]), value["operation_id"])


class WorkerControlFence:
    def __init__(self, authority: Any, lifecycle: Any, validation_timeout: float = 10):
        if not callable(getattr(authority, "validate", None)):
            raise TypeError("Worker control authority must implement validate")
        self.authority = authority
        self.lifecycle = lifecycle
        if (
            isinstance(validation_timeout, bool)
            or not isinstance(validation_timeout, (int, float))
            or not math.isfinite(validation_timeout)
            or validation_timeout <= 0
        ):
            raise ValueError("Invalid authority validation timeout")
        self.validation_timeout = validation_timeout
        self.session: Optional[ControllerSession] = None
        self.ready = False
        self._install_lock = asyncio.Lock()

    def bound_context(self) -> Optional[ControlContext]:
        current = _binding.get()
        return current[1] if current is not None and current[0] is self else None

    def capture(self) -> ControlContext:
        current = self.bound_context()
        if current is not None:
            return current
        return self.capture_event()

    def capture_event(self) -> ControlContext:
        """Bind a newly detected Worker event, ignoring a monitor's old context."""
        if self.session is None or not self.ready:
            raise ControlSessionRejected("not_installed")
        return ControlContext(self.session, uuid.uuid4().hex)

    @contextlib.contextmanager
    def local_scope(self, command: Optional[ControlContext] = None) -> Iterator[None]:
        token = _binding.set((self, command or self.capture()))
        try:
            yield
        finally:
            _binding.reset(token)

    def check_local(self) -> ControlContext:
        command = self.bound_context()
        if command is None:
            raise ControlSessionRejected("missing_context")
        if command.session != self.session:
            raise ControlSessionRejected("stale_session")
        if not self.ready:
            raise ControlSessionRejected("fence_pending")
        return command

    def check_cleanup(self) -> None:
        """An already admitted task can release its own resources during drain.

        The new owner cannot receive an acknowledgement until this task leaves
        the existing lifecycle. An untracked late callback cannot use this path.
        """
        command = self.bound_context()
        if command is None or asyncio.current_task() not in self.lifecycle._tasks:
            raise ControlSessionRejected("cleanup_not_admitted")
        if self.ready and command.session != self.session:
            raise ControlSessionRejected("stale_session")

    async def _validate(
        self, session: ControllerSession, phases: Tuple[str, ...]
    ) -> None:
        async def invoke():
            validate = self.authority.validate
            if inspect.iscoroutinefunction(validate):
                return await validate(session, phases=phases)
            result = await asyncio.to_thread(validate, session, phases=phases)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("Synchronous authority returned an awaitable")
            return result

        try:
            result = await asyncio.wait_for(invoke(), self.validation_timeout)
            if result is False:
                raise ControlSessionRejected("authority_rejected")
        except ControlSessionRejected:
            raise
        except Exception:
            raise ControlSessionRejected("authority_unavailable") from None

    async def checkpoint(self) -> None:
        command = self.check_local()
        await self._validate(command.session, ("active",))
        # Validation itself yields: a newer installation can have closed this
        # session while the authority request was in flight.
        self.check_local()

    @contextlib.asynccontextmanager
    async def command(self, command: ControlContext) -> AsyncIterator[None]:
        with self.local_scope(command):
            await self.checkpoint()
            async with self.lifecycle.operation():
                yield

    def _check_install_order(self, session: ControllerSession) -> None:
        if self.session is None:
            return
        if session.cluster_id != self.session.cluster_id:
            raise ControlSessionRejected("cluster_changed")
        if session.term < self.session.term or (
            session.term == self.session.term and session != self.session
        ):
            raise ControlSessionRejected("stale_session")

    async def install(
        self, session: ControllerSession, timeout: float
    ) -> Dict[str, Any]:
        from .lifecycle import WorkerPhase

        if not isinstance(session, ControllerSession):
            raise ValueError("Invalid controller session")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Invalid controller fence timeout")
        async with self._install_lock:
            self._check_install_order(session)
            await self._validate(session, ("fencing", "reconciling", "active"))
            self._check_install_order(session)
            if self.lifecycle.phase != WorkerPhase.ACTIVE:
                raise ControlSessionRejected("worker_not_active")
            if self.session == session and self.ready:
                return self.status()
            # Never roll back this assignment on timeout, cancellation or loss
            # of authority. Old sessions remain rejected even if drain fails.
            self.session = session
            self.ready = False
            try:
                await self.lifecycle.drain(timeout)
                await self._validate(session, ("fencing", "reconciling", "active"))
                self.ready = True
            finally:
                if self.lifecycle.phase == WorkerPhase.DRAINING:
                    self.lifecycle.phase = WorkerPhase.ACTIVE
            return self.status()

    def status(self) -> Dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "installed": self.session is not None,
            "term": self.session.term if self.session else None,
            "ready": self.ready,
        }


async def check_control_session(owner: Any) -> None:
    fence = getattr(owner, "_control_fence", None)
    if fence is not None:
        await fence.checkpoint()


async def run_worker_control_callback(owner: Any, method: Any) -> Any:
    fence = getattr(owner, "_control_fence", None)
    if fence is None:
        return await method()
    # The status isolation has its own event loop. Capture when the Worker
    # schedules its callback, then admit/drain it on the actor loop.
    captured = fence.capture()

    async def run():
        async with fence.command(captured):
            return await method()

    return await owner._run_on_actor_loop(run)
