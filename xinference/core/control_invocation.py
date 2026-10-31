# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Optional process-local interception of an already admitted control call."""

import asyncio
import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable

PROTOCOL_VERSION = 1

_active: ContextVar[tuple] = ContextVar("control_invocations", default=())


@dataclass(frozen=True)
class ControlInvocation:
    target: str
    method: str
    command: Any
    args: tuple
    kwargs: dict

    def __post_init__(self):
        object.__setattr__(self, "kwargs", dict(self.kwargs))


async def invoke_control_hook(
    authority: Any, hook: str, invocation: ControlInvocation, call: Callable
) -> Any:
    callback = getattr(authority, hook, None)
    if callback is None:
        return await call()
    if not inspect.iscoroutinefunction(callback):
        raise TypeError("Control invocation callbacks must be asynchronous")
    # Background tasks inherit ContextVars, but each new task owns its calls.
    key = (
        id(authority),
        hook,
        invocation.target,
        invocation.method,
        asyncio.current_task(),
    )
    if key in _active.get():
        return await call()
    token = _active.set((*_active.get(), key))
    try:
        return await callback(invocation, call)
    finally:
        _active.reset(token)
