# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

from typing import Optional
from urllib.parse import urlsplit


def validate_gpu_cache_budget(value, enabled: bool, replicas: int):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("xavier_gpu_cache_bytes must be a non-negative integer")
    if not enabled or replicas <= 1:
        raise ValueError(
            "xavier_gpu_cache_bytes requires Xavier with multiple replicas"
        )
    return value


def get_transport_host(address: str) -> Optional[str]:
    return urlsplit(address if "://" in address else "tcp://" + address).hostname
