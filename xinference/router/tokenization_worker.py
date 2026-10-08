from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .tokenizer import DeepSeekV4TokenEstimator, TokenBudget
from .tokenizer_asset import (
    DEFAULT_TOKENIZER_ASSET_FILES,
    compute_tokenizer_asset_fingerprint,
    read_tokenizer_asset_revision,
)

_ESTIMATOR: DeepSeekV4TokenEstimator | None = None
_ASSET_FINGERPRINT = ""
_ASSET_REVISION = ""


def _scrub_router_credentials() -> None:
    for name in (
        "XINFERENCE_API_KEY",
        "XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN",
        "XINFERENCE_TOKEN_ROUTER_DATA_PLANE_TOKEN",
    ):
        os.environ.pop(name, None)


def initialize_tokenization_worker(
    tokenizer_path: str,
    reserve_tokens: int,
    default_output_tokens: int,
    required_files: tuple[str, ...] = DEFAULT_TOKENIZER_ASSET_FILES,
) -> None:
    """Initialize an estimator and measure the files it actually loaded."""
    global _ESTIMATOR, _ASSET_FINGERPRINT, _ASSET_REVISION
    _scrub_router_credentials()
    path = Path(tokenizer_path)
    before_fingerprint = compute_tokenizer_asset_fingerprint(path, required_files)
    before_revision = read_tokenizer_asset_revision(path)
    _ESTIMATOR = DeepSeekV4TokenEstimator(
        path,
        reserve_tokens=reserve_tokens,
        default_output_tokens=default_output_tokens,
    )
    after_fingerprint = compute_tokenizer_asset_fingerprint(path, required_files)
    after_revision = read_tokenizer_asset_revision(path)
    if before_fingerprint != after_fingerprint or before_revision != after_revision:
        raise RuntimeError("Tokenizer asset changed while it was being loaded")
    _ASSET_FINGERPRINT = after_fingerprint
    _ASSET_REVISION = after_revision


def estimate_in_worker(payload: dict[str, Any]) -> TokenBudget:
    if _ESTIMATOR is None:
        raise RuntimeError("Tokenization worker is not initialized")
    return _ESTIMATOR.estimate(payload)


def ping_worker(
    delay_seconds: float = 0.05,
) -> tuple[int, bool, bool, str, str]:
    """Return worker identity, credential state, and measured asset metadata."""
    if _ESTIMATOR is None:
        raise RuntimeError("Tokenization worker is not initialized")
    time.sleep(delay_seconds)
    return (
        os.getpid(),
        "XINFERENCE_API_KEY" in os.environ,
        "XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN" in os.environ,
        _ASSET_FINGERPRINT,
        _ASSET_REVISION,
    )
