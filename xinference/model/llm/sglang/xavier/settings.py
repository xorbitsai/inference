# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Lightweight settings shared by the plugin and P/D metadata actors."""

import math
import os
from typing import Optional

GPU_CONFIG_ENV = "XINFERENCE_SGLANG_XAVIER_GPU_CONFIG"
TRANSFER_TIMEOUT_ENV = "XINFERENCE_SGLANG_XAVIER_TRANSFER_TIMEOUT"


def transfer_timeout(value: Optional[float] = None) -> float:
    value = float(
        value if value is not None else os.environ.get(TRANSFER_TIMEOUT_ENV, "600")
    )
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{TRANSFER_TIMEOUT_ENV} must be a positive finite number")
    return value
