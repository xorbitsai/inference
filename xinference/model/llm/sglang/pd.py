# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Native SGLang NIXL P/D configuration and bootstrap metadata."""

from packaging.version import Version
from xoscar.utils import get_next_port


def configure_nixl(config: dict, replica: dict, version: str, n_worker: int) -> None:
    if Version(version) < Version("0.5.21"):
        raise ValueError("SGLang native NIXL requires SGLang >= 0.5.21")
    if n_worker != 1 or any(
        config.get(key, 1) != 1
        for key in (
            "tp_size",
            "pp_size",
            "dp_size",
            "nnodes",
            "attn_cp_size",
            "dcp_size",
        )
    ):
        raise ValueError("SGLang native NIXL requires TP=PP=DP=1 on one worker")
    if config.get("disaggregation_mode", "null") not in (None, "null") or any(
        config.get(key)
        for key in (
            "enable_lora",
            "lora_paths",
            "enable_dp_attention",
            "speculative_algorithm",
            "speculative_draft_model_path",
            "enable_hierarchical_cache",
            "hicache_storage_backend",
            "_xavier_cache_config",
        )
    ):
        raise ValueError(
            "SGLang native NIXL requires ordinary text P/D without LoRA, speculation or HiCache"
        )
    if replica.get("role") not in ("prefill", "decode"):
        raise ValueError("SGLang native NIXL requires a prefill or decode role")
    replica["port"] = get_next_port()
    config.update(
        host=replica["host"],
        disaggregation_mode=replica["role"],
        disaggregation_transfer_backend="nixl",
        disaggregation_bootstrap_port=replica["port"],
    )


class SGLangNixlHandoff:
    """The native scheduler validates and drains NIXL transfers before decode."""

    def __init__(self, config: dict):
        self.config = config
        self.role = config["role"]

    async def prepare(self, prompt: str, transfer: dict) -> dict:
        handoff = transfer.get("sglang_nixl")
        if (
            not isinstance(handoff, dict)
            or handoff.get("mode") != "nixl"
            or not isinstance(handoff.get("host"), str)
            or not handoff["host"]
            or type(handoff.get("port")) is not int
            or not 0 < handoff["port"] < 65536
            or type(handoff.get("room")) is not int
            or not 0 < handoff["room"] < 2**63
        ):
            raise ValueError("Missing SGLang native NIXL bootstrap metadata")
        if self.role == "prefill" and any(
            handoff[key] != self.config[key] for key in ("host", "port")
        ):
            raise ValueError("SGLang bootstrap does not identify this prefill replica")
        return handoff

    async def accept(self, prompt: str, transfer: dict) -> dict:
        if self.role != "decode":
            raise ValueError("Remote prefill requires a SGLang decode replica")
        return await self.prepare(prompt, transfer)

    async def publish(self, handoff: dict) -> dict:
        return dict(do_remote_prefill=True, sglang_nixl=handoff)

    async def check_hit(self, meta_info: dict, handoff: dict) -> None:
        # Native decode cannot produce output before receiving the KV pages and
        # first-token metadata for this room. Transfer errors abort the request.
        pass

    async def release(self, handoff: dict, failed: bool = False) -> None:
        # abort_request and the native sender/receiver own slot cleanup.
        pass
