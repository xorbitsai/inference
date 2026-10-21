# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Storage selection, independent of inference engines and transfer transports."""

import math
from pathlib import Path
from typing import Any, Mapping, Optional


def normalize_cache_backend(
    backend: Optional[str], config: Optional[Mapping[str, Any]] = None
) -> tuple[str, dict]:
    name = "native" if backend is None else backend
    if name not in ("native", "lmcache"):
        raise ValueError("xavier_cache_backend must be native or lmcache")
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("xavier_cache_backend_config must be a mapping")
    options = dict(config or {})
    if name == "native":
        if options:
            raise ValueError("The native Xavier backend takes no backend options")
        return name, options
    unknown = options.keys() - {"cpu_bytes", "disk_path", "disk_bytes", "read_timeout"}
    if unknown:
        raise ValueError(f"Unknown Xavier LMCache options: {sorted(unknown)}")
    options.setdefault("cpu_bytes", 512 * 1024 * 1024)
    options.setdefault("read_timeout", 30.0)
    for key in ("cpu_bytes", "disk_bytes"):
        if type(options.get(key)) is not int or options[key] <= 0:
            raise ValueError(f"Xavier LMCache {key} must be a positive integer")
    path = options.get("disk_path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("Xavier LMCache disk_path must be an absolute path")
    timeout = options["read_timeout"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("Xavier LMCache read_timeout must be positive and finite")
    return name, options
