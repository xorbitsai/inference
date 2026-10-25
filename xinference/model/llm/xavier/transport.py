# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import ipaddress
import socket
from typing import Dict, Optional
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
    # Launch config already contains a host, including bare IPv6 literals.
    # Re-parsing those as URLs truncates the address at its first colon.
    try:
        return str(ipaddress.ip_address(address))
    except ValueError:
        pass
    return urlsplit(address if "://" in address else "tcp://" + address).hostname


def resolve_nixl_host(host: str, host_key: str = "VLLM_NIXL_SIDE_CHANNEL_HOST") -> str:
    if host not in ("0.0.0.0", "::"):
        return host
    # A UDP connect selects the default-route interface without
    # sending traffic; hostname resolution covers offline hosts.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            host = sock.getsockname()[0]
    except OSError:
        try:
            host = socket.gethostbyname(socket.gethostname())
        except OSError as exc:
            raise ValueError(
                f"Cannot discover a reachable NIXL host; set {host_key} explicitly"
            ) from exc
    if (
        ipaddress.ip_address(host).is_loopback
        or ipaddress.ip_address(host).is_unspecified
    ):
        raise ValueError(
            f"Discovered NIXL host {host!r} is not remotely reachable; "
            f"set {host_key} explicitly"
        )
    return host


def gpu_pool_options(address: str, env: Dict[str, str]) -> Dict[str, str]:
    import os
    from importlib.metadata import version
    from importlib.util import find_spec

    from packaging.version import Version

    if Version(version("xoscar")) < Version("0.11.1") or find_spec("nixl") is None:
        raise RuntimeError(
            "Xavier GPU transfer requires xoscar[nixl]>=0.11.1 on the worker and in the model environment"
        )
    host = get_transport_host(address)
    if not host:
        raise ValueError("Cannot determine Xavier NIXL worker host")
    host = resolve_nixl_host(host, "a reachable worker address")
    if not host:
        raise ValueError("Cannot determine Xavier NIXL worker host")
    if ":" in host:
        host = f"[{host}]"
    env["UCX_MEMTYPE_CACHE"] = "n"
    env.setdefault("UCX_TLS", os.environ.get("UCX_TLS", "tcp,cuda_copy,cuda_ipc"))
    return {"external_address": f"nixl://{host}:0"}
