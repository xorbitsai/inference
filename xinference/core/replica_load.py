# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

"""The load contract between engine adapters, model actors, and replica policy."""

import math
from typing import Awaitable, NamedTuple, Optional, Protocol, Union


class EngineLoad(NamedTuple):
    """Backend work counts in engine units, such as sequences or decode jobs."""

    waiting: int = 0
    backlog_seconds: float = 0.0
    running: int = 0


class EngineLoadReporter(Protocol):
    """Implement on a concrete adapter with `replica_autoscaling = "native"`."""

    def enable_engine_load_reporting(self) -> None:
        """Enable instrumentation before the engine loads."""
        ...

    def get_engine_load(
        self,
    ) -> Union[Optional[EngineLoad], Awaitable[Optional[EngineLoad]]]:
        """Report native work, including waiting inputs hidden from REST admission.

        Keep sampling cheap and non-blocking. Return None when unavailable.
        Provide a sample at startup once initialized, including when idle.
        Report work until the engine releases it, including after cancellation.
        """
        ...


def validate_engine_load(load: EngineLoad) -> None:
    if (
        not isinstance(load, EngineLoad)
        or type(load.waiting) is not int
        or type(load.running) is not int
        or min(load.waiting, load.running) < 0
        or not math.isfinite(load.backlog_seconds)
        or load.backlog_seconds < 0
        or (not load.waiting and load.backlog_seconds != 0)
    ):
        raise ValueError("Invalid autoscaling engine load snapshot")


class ReplicaLoad(NamedTuple):
    queued: int
    oldest_wait: float
    spare: int
    engine_waiting: int = 0
    engine_backlog_seconds: float = 0.0
    engine_running: int = 0
