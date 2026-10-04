# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Opt-in, synchronizing diagnostics; never enable for throughput measurements."""

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)
_ENABLED = os.environ.get("XINFERENCE_XAVIER_PROFILE") == "1"
if _ENABLED:
    # Keep configured actor/file logging; EngineCore may have no handlers.
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = True


@contextmanager
def profile_stage(stage: str, *, device: Any = None, **fields: Any) -> Iterator[None]:
    if not _ENABLED:
        yield
        return

    def synchronize() -> None:
        if device is not None and device.type == "cuda":
            import torch

            torch.cuda.synchronize(device)

    # Drain preceding GPU work outside the timed region. This intentionally
    # serializes execution; profile results are separate from benchmark runs.
    synchronize()
    started = time.perf_counter()
    succeeded = False
    try:
        yield
        synchronize()
        succeeded = True
    finally:
        elapsed = time.perf_counter() - started
        logger.info(
            "Xavier profile: %s",
            json.dumps(
                dict(
                    stage=stage,
                    elapsed_s=elapsed,
                    pid=os.getpid(),
                    succeeded=succeeded,
                    **fields,
                ),
                sort_keys=True,
            ),
        )
