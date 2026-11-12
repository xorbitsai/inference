# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Configure SGLang's Xavier HiCache sharing or native GPU P/D adapter."""

import json
import logging
from pathlib import Path

from packaging.version import Version

from .....constants import XINFERENCE_CACHE_DIR
from ...xavier.contract import KVCacheContract, fingerprint_files, fingerprint_metadata

logger = logging.getLogger(__name__)


# Full-attention text models keep the single-pool page contract at TP=1.
_PAGE_MODEL_TYPES = ("qwen2", "qwen3", "llama")
# SGLang offloads these as several host pools with replicated MLA pages.
_POOL_MODEL_TYPES = _PAGE_MODEL_TYPES + ("deepseek_v4", "deepseek_v41")
_SELF_PAGED_MODEL_TYPES = ("deepseek_v4", "deepseek_v41")
# Multi-pool storage needs SGLang's host-pool registration callback.
POOL_SHARING_MIN_SGLANG_VERSION = Version("0.5.23")
# Shards hashed at once when fingerprinting a multi-file checkpoint.
_FINGERPRINT_WORKERS = 8
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
)


def _uses_page_contract(model_config: dict, config: dict, cache_config: dict) -> bool:
    """Whether the original TP=1, unquantized, full-attention path applies.

    P/D roles and cross-engine sharing exchange canonical pages and therefore
    always use it; the checks below then reject anything it cannot describe.
    """
    if (
        cache_config.get("role")
        or cache_config.get("heterogeneous")
        or cache_config.get("shared_prefix")
    ):
        return True
    return (
        model_config.get("tp_size", 1) == 1
        and model_config.get("quantization") in (None, "none")
        and config.get("model_type") in _PAGE_MODEL_TYPES
        and not config.get("use_sliding_window", False)
        and not config.get("vision_config")
        and not config.get("quantization_config")
        and all(
            layer == "full_attention" for layer in (config.get("layer_types") or [])
        )
    )


def configure_xavier(model_path: str, model_config: dict, cache_config: dict) -> None:
    config = json.loads((Path(model_path) / "config.json").read_text())
    if not _uses_page_contract(model_config, config, cache_config):
        _configure_pools(model_path, model_config, cache_config, config)
        return
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
    _check_engine_options(model_config)
    if model_config.get("quantization") not in (None, "none"):
        raise ValueError("SGLang Xavier requires unquantized weights")
    dtype = cache_config.get("logical_dtype", "float16")
    allowed = ("float16", "half") if dtype == "float16" else ("bfloat16",)
    if (
        dtype not in ("float16", "bfloat16")
        or model_config.get("dtype") not in (None, "auto", *allowed)
        or model_config.get("kv_cache_dtype", "auto") not in ("auto", dtype)
    ):
        raise ValueError(f"SGLang Xavier currently requires {dtype} KV")
    path = Path(model_path)
    if (
        config.get("model_type") not in _PAGE_MODEL_TYPES
        or config.get("use_sliding_window", False)
        or config.get("vision_config")
        or config.get("quantization_config")
        or any(layer != "full_attention" for layer in (config.get("layer_types") or []))
    ):
        raise ValueError(
            "SGLang Xavier supports full-attention Qwen2, Qwen3 and Llama text models"
        )
    weights = sorted(path.glob("*.safetensors")) or sorted(
        path.glob("pytorch_model*.bin")
    )
    tokenizer = [path / name for name in _TOKENIZER_FILES if (path / name).is_file()]
    if not weights or not any(
        file.name in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json")
        for file in tokenizer
    ):
        raise ValueError(
            "SGLang Xavier requires local model weights and tokenizer assets"
        )
    model_config.setdefault("page_size", 64)
    if cache_config.get("heterogeneous") or cache_config.get("shared_prefix"):
        from ...xavier.pd_contract import build_pd_contract

        if model_config.get("load_format", "auto") not in ("auto", "safetensors"):
            raise ValueError(
                "Cross-engine Xavier requires standard automatic or safetensors loading"
            )
        if model_config["page_size"] != 64:
            raise ValueError("Cross-engine Xavier PD requires 64-token pages")
        contract = build_pd_contract(
            model_path, model_config.get("context_length"), logical_dtype=dtype
        )
    else:
        num_kv_heads = config.get("num_key_value_heads")
        head_dim = config.get("head_dim")
        contract = KVCacheContract(
            weights_fingerprint=fingerprint_files(
                {file.name: file for file in weights},
                cache_dir=Path(XINFERENCE_CACHE_DIR) / "xavier-fingerprints",
            ),
            tokenizer_fingerprint=fingerprint_files(
                {file.name: file for file in tokenizer}
            ),
            attention_fingerprint=fingerprint_metadata(config),
            position_fingerprint=fingerprint_metadata(
                {"config": config, "context_length": model_config.get("context_length")}
            ),
            num_layers=config["num_hidden_layers"],
            num_kv_heads=(
                config["num_attention_heads"] if num_kv_heads is None else num_kv_heads
            ),
            head_dim=(
                config["hidden_size"] // config["num_attention_heads"]
                if head_dim is None
                else head_dim
            ),
            block_size=model_config["page_size"],
            logical_dtype="float16",
        )
    if dtype == "float16" and config.get("torch_dtype", config.get("dtype")) in (
        "bfloat16",
        "bf16",
    ):
        logger.warning(
            "SGLang Xavier casts this BF16 checkpoint to FP16; weights and KV use FP16"
        )
    model_config["dtype"] = dtype
    cache_config["contract"] = contract.to_dict()
    prefix_cache = cache_config.get("prefix_cache")
    if cache_config.get("role"):
        if not prefix_cache and (
            model_config.get("enable_hierarchical_cache")
            or any(name.startswith("hicache_") for name in model_config)
        ):
            raise ValueError("SGLang Xavier GPU PD cannot be combined with CPU HiCache")
        if cache_config["role"] not in ("prefill", "decode"):
            raise ValueError("SGLang Xavier GPU PD requires a prefill or decode role")
        if model_config.get("disaggregation_decode_enable_radix_cache"):
            raise ValueError("SGLang Xavier GPU PD does not support decode radix cache")
        model_config["disaggregation_decode_enable_radix_cache"] = False
        cache_config["contract"] = contract.to_dict()
        model_config["disaggregation_mode"] = cache_config["role"]
        model_config["disaggregation_transfer_backend"] = "xavier"
        model_config["enable_cache_report"] = True
        if not prefix_cache:
            return
        if cache_config["role"] != "prefill":
            raise ValueError("Only the prefill role can use LMCache history")
    _size_host_pools(model_config, cache_config.get("replicas", 1), False)
    _enable_hicache(
        model_config,
        {
            "prefetch_threshold": contract.block_size,
            "contract": contract.to_dict(),
            **(prefix_cache or cache_config),
        },
    )


def _check_engine_options(model_config: dict) -> None:
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


# SGLang sizes HiCache host pools from free host memory divided among its own
# ranks; replicas launched together would each claim the same memory. Xavier
# splits this share among replicas and leaves room for its shared CPU cache.
_HOST_MEMORY_FRACTION = 0.5


def _size_host_pools(model_config: dict, replicas: int, self_paged: bool) -> None:
    if any(
        model_config.get(name) is not None
        for name in ("hicache_ratio", "hicache_size", "hicache_host_memory_fraction")
    ):
        return
    share = round(_HOST_MEMORY_FRACTION / max(1, replicas), 4)
    if self_paged:
        # DeepSeek-V4 host pools cannot be auto-sized or given a fixed size.
        # Its compressed device pools hold tens of millions of tokens, so the
        # host mirror is scaled down by the same share.
        model_config["hicache_ratio"] = share
    elif replicas > 1:
        model_config["hicache_host_memory_fraction"] = share


def _enable_hicache(model_config: dict, storage_config: dict) -> None:
    model_config["enable_hierarchical_cache"] = True
    model_config["hicache_mem_layout"] = "layer_first"
    model_config.setdefault("hicache_write_policy", "write_through")
    model_config.setdefault("hicache_io_backend", "kernel")
    model_config.setdefault("hicache_storage_prefetch_policy", "wait_complete")
    model_config["hicache_storage_backend"] = "dynamic"
    model_config["hicache_storage_backend_extra_config"] = json.dumps(
        {
            "backend_name": "xavier",
            "module_path": "xinference.model.llm.sglang.xavier.storage",
            "class_name": "XavierHiCacheStorage",
            "interface_v1": True,
            **storage_config,
        }
    )


def _configure_pools(
    model_path: str, model_config: dict, cache_config: dict, config: dict
) -> None:
    """Share SGLang's own host pools: tensor parallel, quantized or MLA models.

    SGLang decides what each pool holds, which rank writes it and which hit
    policy applies; Xavier stores the pages and checks every replica registers
    the same pools for the same model files.
    """
    if any(
        model_config.get(name, 1) != 1
        for name in ("pp_size", "dp_size", "nnodes", "attn_cp_size", "dcp_size")
    ):
        raise ValueError(
            "SGLang Xavier supports tensor parallelism on one worker without PP, DP or CP"
        )
    _check_engine_options(model_config)
    model_type = config.get("model_type")
    if model_type not in _POOL_MODEL_TYPES:
        raise ValueError(
            "SGLang Xavier supports Qwen2, Qwen3, Llama and DeepSeek-V4 models"
        )
    if model_type in _PAGE_MODEL_TYPES and (
        config.get("use_sliding_window", False)
        or config.get("vision_config")
        or any(layer != "full_attention" for layer in (config.get("layer_types") or []))
    ):
        raise ValueError(
            "SGLang Xavier supports full-attention Qwen2, Qwen3 and Llama text models"
        )
    version = cache_config.get("sglang_version")
    if version is not None and Version(version) < POOL_SHARING_MIN_SGLANG_VERSION:
        # Older releases lack the host-pool registration callback that
        # publishes the namespace, and the multi-pool storage interface.
        raise ValueError(
            "SGLang Xavier needs SGLang >= "
            f"{POOL_SHARING_MIN_SGLANG_VERSION} for tensor-parallel, quantized "
            f"or DeepSeek-V4 caches; found {version}"
        )
    path = Path(model_path)
    weights = sorted(path.glob("*.safetensors"))
    tokenizer = {
        name: path / name for name in _TOKENIZER_FILES if (path / name).is_file()
    }
    if not weights or not any(
        name in tokenizer
        for name in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json")
    ):
        raise ValueError(
            "SGLang Xavier requires local safetensors weights and tokenizer assets"
        )
    # SGLang loads only the shards the index's weight_map names, so the same
    # shard bytes under a different index are different weights.
    index = path / "model.safetensors.index.json"
    try:
        weight_map = (
            json.loads(index.read_text())["weight_map"] if index.is_file() else None
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid safetensors index {index.name}") from error
    if weight_map is not None and not isinstance(weight_map, dict):
        raise ValueError(f"Invalid safetensors index {index.name}")
    # Replicas on different workers may hold different local copies of one
    # configured path, so weights are identified by content. Digests are cached
    # per file and reused while the file's identity metadata is unchanged.
    identity = {
        "weights": fingerprint_files(
            {file.name: file for file in weights},
            cache_dir=Path(XINFERENCE_CACHE_DIR) / "xavier-fingerprints",
            workers=_FINGERPRINT_WORKERS,
        ),
        "weight_map": (
            fingerprint_metadata(weight_map) if weight_map is not None else None
        ),
        "tokenizer": fingerprint_files(tokenizer),
        "config": fingerprint_metadata(config),
        "context_length": model_config.get("context_length"),
        "dtype": model_config.get("dtype"),
        "kv_cache_dtype": model_config.get("kv_cache_dtype"),
        "quantization": model_config.get("quantization"),
    }
    if model_type not in _SELF_PAGED_MODEL_TYPES:
        model_config.setdefault("page_size", 64)
    page_size = model_config.get("page_size") or 256
    cache_config["pools"] = True
    cache_config["identity"] = identity
    _size_host_pools(
        model_config,
        cache_config.get("replicas", 1),
        model_type in _SELF_PAGED_MODEL_TYPES,
    )
    _enable_hicache(
        model_config,
        {
            "prefetch_threshold": page_size,
            "pools": True,
            "identity": identity,
            **{
                key: value
                for key, value in cache_config.items()
                if key in ("address", "uid", "read_timeout", "replica")
            },
        },
    )
