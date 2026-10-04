# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Translate Xinference's Xavier option into SGLang's dynamic HiCache backend."""

import json
from pathlib import Path

from ...xavier.contract import KVCacheContract, fingerprint_files, fingerprint_metadata


def configure_xavier(model_path: str, model_config: dict, cache_config: dict) -> None:
    if any(
        model_config.get(name, 1) != 1
        for name in (
            "tp_size",
            "pp_size",
            "dp_size",
            "nnodes",
            "attn_cp_size",
            "dcp_size",
        )
    ):
        raise ValueError("SGLang Xavier requires TP=PP=DP=1 on one worker")
    if any(
        model_config.get(name)
        for name in (
            "enable_lora",
            "lora_paths",
            "enable_dp_attention",
            "disable_radix_cache",
            "speculative_algorithm",
            "speculative_draft_model_path",
            "json_model_override_args",
            "skip_tokenizer_init",
            "custom_weight_loader",
            "remote_instance_weight_loader_seed_instance_ip",
        )
    ):
        raise ValueError(
            "SGLang Xavier requires ordinary text attention without model overrides, LoRA or speculation"
        )
    if model_config.get("load_format", "auto") not in ("auto", "safetensors", "pt"):
        raise ValueError("SGLang Xavier requires loading the local model weights")
    if (
        model_config.get("model_impl", "auto") not in ("auto", "sglang")
        or model_config.get("tokenizer_mode", "auto") != "auto"
        or model_config.get("tokenizer_backend", "huggingface") != "huggingface"
        or json.loads(model_config.get("model_loader_extra_config") or "{}")
    ):
        raise ValueError(
            "SGLang Xavier requires the native model and standard tokenizer"
        )
    if model_config.get("quantization") not in (None, "none"):
        raise ValueError("SGLang Xavier requires unquantized weights")
    if model_config.get("dtype") not in (
        None,
        "auto",
        "float16",
        "half",
    ) or model_config.get("kv_cache_dtype", "auto") not in ("auto", "float16"):
        raise ValueError("SGLang Xavier currently requires FP16 KV")
    if model_config.get("hicache_storage_backend") or model_config.get(
        "hicache_storage_backend_extra_config"
    ):
        raise ValueError(
            "SGLang Xavier cannot be combined with another HiCache storage backend"
        )
    if model_config.get("hicache_mem_layout", "layer_first") != "layer_first":
        raise ValueError(
            "SGLang Xavier currently requires layer_first host cache layout"
        )
    if model_config.get("disaggregation_mode", "null") not in (None, "null"):
        raise ValueError(
            "SGLang Xavier PD cannot be combined with native disaggregation"
        )
    path = Path(model_path)
    config = json.loads((path / "config.json").read_text())
    if (
        config.get("model_type") not in ("qwen2", "qwen3", "llama")
        or config.get("use_sliding_window", False)
        or config.get("vision_config")
        or config.get("quantization_config")
        or any(layer != "full_attention" for layer in config.get("layer_types", []))
    ):
        raise ValueError(
            "SGLang Xavier supports full-attention Qwen2, Qwen3 and Llama text models"
        )
    weights = sorted(path.glob("*.safetensors")) or sorted(
        path.glob("pytorch_model*.bin")
    )
    tokenizer = [
        path / name
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "spiece.model",
        )
        if (path / name).is_file()
    ]
    if not weights or not any(
        file.name in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json")
        for file in tokenizer
    ):
        raise ValueError(
            "SGLang Xavier requires local model weights and tokenizer assets"
        )
    model_config.setdefault("page_size", 64)
    contract = KVCacheContract(
        weights_fingerprint=fingerprint_files({file.name: file for file in weights}),
        tokenizer_fingerprint=fingerprint_files(
            {file.name: file for file in tokenizer}
        ),
        attention_fingerprint=fingerprint_metadata(config),
        position_fingerprint=fingerprint_metadata(
            {"config": config, "context_length": model_config.get("context_length")}
        ),
        num_layers=config["num_hidden_layers"],
        num_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
        head_dim=config.get(
            "head_dim", config["hidden_size"] // config["num_attention_heads"]
        ),
        block_size=model_config["page_size"],
        logical_dtype="float16",
    )
    model_config["dtype"] = "float16"
    model_config["enable_hierarchical_cache"] = True
    model_config["hicache_mem_layout"] = "layer_first"
    model_config.setdefault("hicache_write_policy", "write_through")
    model_config.setdefault("hicache_io_backend", "kernel")
    model_config.setdefault("hicache_storage_prefetch_policy", "wait_complete")
    if cache_config.get("role"):
        model_config.setdefault("hicache_host_memory_mode", "buffer_only")
        if (
            model_config["hicache_write_policy"] != "write_through"
            or model_config["hicache_storage_prefetch_policy"] != "wait_complete"
            or model_config["hicache_host_memory_mode"] != "buffer_only"
        ):
            raise ValueError(
                "SGLang Xavier PD requires write_through, wait_complete and buffer_only"
            )
        model_config["enable_cache_report"] = True
    model_config["hicache_storage_backend"] = "dynamic"
    model_config["hicache_storage_backend_extra_config"] = json.dumps(
        {
            "backend_name": "xavier",
            "module_path": "xinference.model.llm.sglang.xavier.storage",
            "class_name": "XavierHiCacheStorage",
            "interface_v1": True,
            "prefetch_threshold": contract.block_size,
            "contract": contract.to_dict(),
            **cache_config,
        }
    )
