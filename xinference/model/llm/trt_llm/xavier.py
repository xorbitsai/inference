# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Configure same-engine TRT-LLM sharing of immutable Xavier CPU pages."""

import json
import os

from ..xavier.pd_contract import build_pd_contract

CONFIG_ENV = "XINFERENCE_TRTLLM_XAVIER_CONFIG"


def configure_xavier(model_path: str, config: dict, cache_config: dict) -> None:
    if cache_config.get("role") or cache_config.get("heterogeneous"):
        raise ValueError(
            "TRT-LLM Xavier currently supports shared caching, without P/D"
        )
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
    ):
        if config.get(name, 1) != 1:
            raise ValueError("TRT-LLM Xavier currently requires TP=PP=CP=1")
    allowed = {
        "backend",
        "dtype",
        "max_seq_len",
        "max_batch_size",
        "max_num_tokens",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
        "kv_cache_config",
        "cuda_graph_config",
        "enable_chunked_prefill",
        "disable_overlap_scheduler",
        "scheduler_config",
        "trust_remote_code",
    }
    if set(config) - allowed or config.get("trust_remote_code"):
        raise ValueError("TRT-LLM Xavier does not allow model or execution overrides")
    if config.get("dtype", "float16") not in ("auto", "float16", "half"):
        raise ValueError("TRT-LLM Xavier currently requires FP16 weights and KV")
    if config.get("enable_chunked_prefill"):
        raise ValueError("TRT-LLM Xavier does not yet support chunked prefill")
    if config.get("disable_overlap_scheduler", True) is not True:
        raise ValueError("TRT-LLM Xavier requires disable_overlap_scheduler=true")
    if config.get("scheduler_config"):
        raise ValueError("TRT-LLM Xavier manages its capacity scheduler policy")
    contract = build_pd_contract(model_path, config.get("max_seq_len"))
    kv = dict(config.get("kv_cache_config") or {})
    if set(kv) - {
        "max_tokens",
        "free_gpu_memory_fraction",
        "tokens_per_block",
        "enable_block_reuse",
    }:
        raise ValueError(
            "TRT-LLM Xavier requires a single GPU KV pool without offload or quantization"
        )
    if (
        kv.get("tokens_per_block", 64) != 64
        or kv.get("enable_block_reuse", True) is not True
    ):
        raise ValueError("TRT-LLM Xavier requires 64-token pages and block reuse")
    kv.update(tokens_per_block=64, enable_block_reuse=True)
    config.update(
        dtype="float16",
        kv_cache_config=kv,
        disable_overlap_scheduler=True,
        enable_chunked_prefill=False,
        scheduler_config={"capacity_scheduler_policy": "GUARANTEED_NO_EVICT"},
        kv_connector_config={
            "connector_module": "xinference.model.llm.trt_llm.connector",
            "connector_scheduler_class": "XavierConnectorScheduler",
            "connector_worker_class": "XavierConnectorWorker",
        },
    )
    # The model process owns its environment; TRT's spawned workers inherit it.
    # KvCacheConnectorConfig in 1.2.1 has no user-config field.
    os.environ[CONFIG_ENV] = json.dumps(
        {**cache_config, "contract": contract.to_dict()}
    )
