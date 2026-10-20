# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Request-scoped prefix-cache isolation, independent of engine dependencies."""

import hashlib
from typing import Any, Optional


def validate_cache_salt(value: Any) -> Optional[str]:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 256
        or any(
            ord(char) < 32 or ord(char) == 127 or 0xD800 <= ord(char) <= 0xDFFF
            for char in value
        )
    ):
        raise ValueError(
            "cache_salt must be a nonempty string of at most 256 characters without control characters"
        )
    return value


def cache_scope(cache_salt: Optional[str]) -> Optional[str]:
    """A stable, non-secret binding for cache keys and P/D handoffs."""
    salt = validate_cache_salt(cache_salt)
    return hashlib.sha256(salt.encode()).hexdigest() if salt is not None else None


def redact_cache_salt(value: Any) -> Any:
    """Copy log arguments without exposing salts in nested generation configs."""
    if isinstance(value, dict):
        return {
            key: "<redacted>" if key == "cache_salt" else redact_cache_salt(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return type(value)(redact_cache_salt(item) for item in value)
    return value
