from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .tokenizer import DeepSeekV4TokenEstimator, TokenBudget

_ESTIMATOR: DeepSeekV4TokenEstimator | None = None


def initialize_tokenization_worker(
    tokenizer_path: str,
    reserve_tokens: int,
    default_output_tokens: int,
) -> None:
    """Initialize one process-local estimator and remove backend credentials."""
    global _ESTIMATOR
    os.environ.pop("XINFERENCE_API_KEY", None)
    os.environ.pop("XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN", None)
    _ESTIMATOR = DeepSeekV4TokenEstimator(
        Path(tokenizer_path),
        reserve_tokens=reserve_tokens,
        default_output_tokens=default_output_tokens,
    )


def estimate_in_worker(payload: dict[str, Any]) -> TokenBudget:
    if _ESTIMATOR is None:
        raise RuntimeError("Tokenization worker is not initialized")
    return _ESTIMATOR.estimate(payload)


def ping_worker(delay_seconds: float = 0.05) -> tuple[int, bool, bool]:
    """Return worker identity without exposing any credential value."""
    if _ESTIMATOR is None:
        raise RuntimeError("Tokenization worker is not initialized")
    time.sleep(delay_seconds)
    return (
        os.getpid(),
        "XINFERENCE_API_KEY" in os.environ,
        "XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN" in os.environ,
    )
