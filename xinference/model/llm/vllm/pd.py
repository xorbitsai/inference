# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Native vLLM PD configuration, without importing optional GPU libraries."""

import ipaddress
import os
import socket
from typing import Any, Dict
from uuid import uuid4

from packaging.version import Version


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


def configure_nixl_environment(env: Dict[str, str], worker_address: str) -> None:
    from xoscar.utils import get_next_port

    # Native NIXL's recurrent-state transfer requires dimension-first conv
    # states. Select the layout before vLLM initializes model/cache classes.
    env.setdefault(
        "VLLM_SSM_CONV_STATE_LAYOUT",
        os.environ.get("VLLM_SSM_CONV_STATE_LAYOUT", "DS"),
    )
    # Each model subprocess needs its own listener, including same-host replicas.
    # Reallocate on recovery as the old listener may not have exited yet.
    host_key = "VLLM_NIXL_SIDE_CHANNEL_HOST"
    if host_key not in env:
        host = os.environ.get(host_key, worker_address.rsplit(":", 1)[0].strip("[]"))
        if host_key not in os.environ:
            host = resolve_nixl_host(host)
        env[host_key] = host
    env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(get_next_port())


def configure_nixl_engine(
    config: Dict[str, Any], vllm_version: Version, enable_lora: bool
) -> None:
    if vllm_version < Version("0.21.0"):
        raise ValueError("Native NIXL PD requires vLLM >= 0.21.0")
    if config.get("kv_transfer_config") is not None:
        raise ValueError("NIXL PD manages kv_transfer_config; do not set it separately")
    if enable_lora:
        raise ValueError("Native NIXL PD currently requires a model without LoRA")
    for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
        if config.get(key, 1) != 1:
            raise ValueError("Native NIXL PD phase one requires TP=1, PP=1 and DP=1")
    from vllm.config import KVTransferConfig

    config["kv_transfer_config"] = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
        engine_id=uuid4().hex,
        kv_load_failure_policy="fail",
    )
