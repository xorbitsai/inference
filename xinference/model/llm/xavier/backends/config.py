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
    unknown = options.keys() - {
        "cpu_bytes",
        "disk_path",
        "disk_bytes",
        "read_timeout",
        "l2_backend",
        "storage_bytes",
        "mooncake_config",
    }
    if unknown:
        raise ValueError(f"Unknown Xavier LMCache options: {sorted(unknown)}")
    options.setdefault("cpu_bytes", 512 * 1024 * 1024)
    options.setdefault("read_timeout", 30.0)
    l2_backend = options.get("l2_backend", "filesystem")
    if l2_backend not in ("filesystem", "mooncake"):
        raise ValueError("Xavier LMCache l2_backend must be filesystem or mooncake")
    budget = "disk_bytes" if l2_backend == "filesystem" else "storage_bytes"
    for key in ("cpu_bytes", budget):
        if type(options.get(key)) is not int or options[key] <= 0:
            raise ValueError(f"Xavier LMCache {key} must be a positive integer")
    if l2_backend == "filesystem":
        if "storage_bytes" in options or "mooncake_config" in options:
            raise ValueError("Mooncake options require l2_backend=mooncake")
        path = options.get("disk_path")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("Xavier LMCache disk_path must be an absolute path")
    else:
        if "disk_path" in options or "disk_bytes" in options:
            raise ValueError("Mooncake storage does not accept filesystem options")
        setup = options.get("mooncake_config")
        if not isinstance(setup, Mapping) or any(
            not isinstance(k, str) or not k or type(v) not in (str, int)
            for k, v in setup.items()
        ):
            raise ValueError("mooncake_config must map strings to strings or integers")
        required = (
            "local_hostname",
            "metadata_server",
            "master_server_addr",
            "protocol",
        )
        if any(
            not isinstance(setup.get(k), str) or not setup[k].strip() for k in required
        ):
            raise ValueError(f"mooncake_config requires {', '.join(required)}")
        if setup["protocol"] not in ("tcp", "rdma"):
            raise ValueError("Mooncake protocol must be tcp or rdma")
        options["mooncake_config"] = {k: str(v) for k, v in setup.items()}
        options["mooncake_config"].setdefault("global_segment_size", "0")
    timeout = options["read_timeout"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("Xavier LMCache read_timeout must be positive and finite")
    return name, options
