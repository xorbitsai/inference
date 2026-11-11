# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Configure TRT-LLM shared caching and Xavier GPU P/D."""

import json
import os
from dataclasses import replace
from pathlib import Path

from ..xavier.pd_contract import build_pd_contract

CONFIG_ENV = "XINFERENCE_TRTLLM_XAVIER_CONFIG"
# TRT's MPI pool forwards only TRTLLM*/TLLM* variables to tensor-parallel ranks.
WORKER_CONFIG_ENV = "TRTLLM_XINFERENCE_XAVIER_CONFIG"
EXECUTION_OPTIONS = {
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


def configure_xavier(
    model_path: str, config: dict, cache_config: dict, *, engine_version: str = "1.2.1"
) -> None:
    if cache_config.get("role") not in (None, "prefill", "decode"):
        raise ValueError("TRT-LLM Xavier role must be prefill, decode or unset")
    if (
        cache_config.get("heterogeneous")
        and cache_config.get("logical_dtype") != "bfloat16"
    ):
        raise ValueError("TRT-LLM cross-engine Xavier requires BF16 replicas")
    if cache_config.get("role") is not None:
        from ..xavier.transport import gpu_pool_options

        if type(cache_config.get("rank")) is not int:
            raise ValueError("TRT-LLM Xavier P/D requires a replica rank")
        gpu_pool_options(cache_config.get("host", ""), dict(os.environ))
    from .compat import supports_tensor_parallel

    ordinary = cache_config.get("role") is None and not cache_config.get(
        "heterogeneous"
    )
    parallel_runtime = supports_tensor_parallel(engine_version)
    tensor_parallel = config.get("tensor_parallel_size", 1)
    if type(tensor_parallel) is not int or tensor_parallel < 1:
        raise ValueError("tensor_parallel_size must be a positive integer")
    if tensor_parallel > 1 and not (ordinary and parallel_runtime):
        raise ValueError(
            "TRT-LLM Xavier P/D requires TP=PP=CP=1; ordinary Xavier replicas "
            "support TP>1 with tensorrt-llm>=1.4.0rc3"
        )
    for name in ("pipeline_parallel_size", "context_parallel_size"):
        if config.get(name, 1) != 1:
            raise ValueError("TRT-LLM Xavier currently requires PP=CP=1")
    if tensor_parallel > 1 and cache_config.get("read_timeout") is not None:
        # Ranks read their own KV shards after the scheduler pins them. Only
        # the native store can pin pages against eviction.
        raise ValueError("TRT-LLM Xavier with TP>1 requires the native cache backend")
    if set(config) - EXECUTION_OPTIONS or config.get("trust_remote_code"):
        raise ValueError("TRT-LLM Xavier does not allow model or execution overrides")
    if config.get("dtype", "bfloat16") not in ("auto", "bfloat16"):
        raise ValueError("TRT-LLM Xavier currently requires BF16 weights and KV")
    if config.get("enable_chunked_prefill"):
        raise ValueError("TRT-LLM Xavier does not yet support chunked prefill")
    if config.get("disable_overlap_scheduler", True) is not True:
        raise ValueError("TRT-LLM Xavier requires disable_overlap_scheduler=true")
    scheduler = dict(config.get("scheduler_config") or {})
    if (
        set(scheduler) - {"capacity_scheduler_policy", "waiting_queue_policy"}
        or scheduler.get("capacity_scheduler_policy", "GUARANTEED_NO_EVICT")
        != "GUARANTEED_NO_EVICT"
    ):
        raise ValueError("TRT-LLM Xavier manages its capacity scheduler policy")
    scheduler["capacity_scheduler_policy"] = "GUARANTEED_NO_EVICT"
    contract = build_pd_contract(model_path, config.get("max_seq_len"))
    checkpoint_config = json.loads((Path(model_path) / "config.json").read_text())
    if (
        checkpoint_config.get("dtype", checkpoint_config.get("torch_dtype"))
        != "bfloat16"
    ):
        raise ValueError("TRT-LLM Xavier requires a BF16 checkpoint")
    contract = replace(contract, logical_dtype="bfloat16")
    kv = dict(config.get("kv_cache_config") or {})
    if (
        set(kv)
        - {
            "max_tokens",
            "free_gpu_memory_fraction",
            "tokens_per_block",
            "enable_block_reuse",
            "enable_partial_reuse",
            "host_cache_size",
        }
        or kv.get("host_cache_size", 0) != 0
    ):
        raise ValueError(
            "TRT-LLM Xavier requires a single GPU KV pool without offload or quantization"
        )
    if (
        kv.get("tokens_per_block", 64) != 64
        or kv.get("enable_block_reuse", True) is not True
        or kv.get("enable_partial_reuse", False) is not False
    ):
        raise ValueError(
            "TRT-LLM Xavier requires 64-token pages and whole-page block reuse"
        )
    # A partially reused local page leaves the computed prefix off a page
    # boundary, where an imported page cannot continue it.
    kv.update(tokens_per_block=64, enable_block_reuse=True, enable_partial_reuse=False)
    if parallel_runtime:
        # TRT 1.4 provisions a host KV tier by default. A connector addresses
        # GPU page slots, which host offload would reassign.
        kv["host_cache_size"] = 0
    elif "host_cache_size" in kv:
        del kv["host_cache_size"]
    if tensor_parallel > 1:
        # Each rank owns num_kv_heads / TP heads of every page. The shared
        # store keeps one page per rank, all described by this shard contract.
        if contract.num_kv_heads % tensor_parallel:
            raise ValueError(
                "TRT-LLM Xavier requires num_key_value_heads divisible by TP"
            )
        contract = replace(
            contract, num_kv_heads=contract.num_kv_heads // tensor_parallel
        )
    config.update(
        dtype="bfloat16",
        kv_cache_config=kv,
        disable_overlap_scheduler=True,
        enable_chunked_prefill=False,
        scheduler_config=scheduler,
        kv_connector_config={
            "connector_module": "xinference.model.llm.trt_llm.connector",
            "connector_scheduler_class": "XavierConnectorScheduler",
            "connector_worker_class": "XavierConnectorWorker",
        },
    )
    cache_config["contract"] = contract.to_dict()
    # The model process owns its environment and hosts TRT's embedded worker.
    # The connector receives only the native engine arguments.
    if tensor_parallel > 1:
        cache_config["tensor_parallel_size"] = tensor_parallel
    worker_config = json.dumps(
        {
            **cache_config,
            "engine_version": engine_version,
            "contract": contract.to_dict(),
            "max_seq_len": config.get("max_seq_len")
            or checkpoint_config["max_position_embeddings"],
        }
    )
    os.environ[CONFIG_ENV] = worker_config
    if tensor_parallel > 1:
        os.environ[WORKER_CONFIG_ENV] = worker_config
    else:
        os.environ.pop(WORKER_CONFIG_ENV, None)
