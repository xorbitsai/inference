# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Native vLLM PD configuration, without importing optional GPU libraries."""

from typing import Any, Dict
from uuid import uuid4

from packaging.version import Version


def configure_nixl_environment(env: Dict[str, str], worker_address: str) -> None:
    from xoscar.utils import get_next_port

    # Each model subprocess needs its own listener, including same-host replicas.
    # Reallocate on recovery as the old listener may not have exited yet.
    host = worker_address.rsplit(":", 1)[0]
    if host in ("0.0.0.0", "::", "[::]"):
        raise ValueError("NIXL requires a reachable worker host address")
    env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = host
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
