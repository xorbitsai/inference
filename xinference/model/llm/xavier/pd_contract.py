# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Common model identity and token boundaries for cross-engine GPU PD."""

import hashlib
import json
from pathlib import Path

from ....constants import XINFERENCE_CACHE_DIR
from .contract import KVCacheContract, fingerprint_files, fingerprint_metadata


def prompt_digest(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def build_pd_contract(
    model_path: str, context_length: int | None = None
) -> KVCacheContract:
    path = Path(model_path)
    config = json.loads((path / "config.json").read_text())
    if (
        config.get("model_type") not in ("qwen2", "qwen3", "llama")
        or config.get("use_sliding_window")
        or config.get("vision_config")
        or config.get("quantization_config")
        or config.get("quantization")
        or config.get("auto_map")
        or any(kind != "full_attention" for kind in (config.get("layer_types") or []))
    ):
        raise ValueError(
            "Cross-engine Xavier PD requires full-attention Qwen2, Qwen3 or Llama text weights"
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
            "Cross-engine Xavier PD requires local weights and tokenizer assets"
        )
    return KVCacheContract(
        weights_fingerprint=fingerprint_files(
            {file.name: file for file in weights},
            cache_dir=Path(XINFERENCE_CACHE_DIR) / "xavier-fingerprints",
        ),
        tokenizer_fingerprint=fingerprint_files(
            {file.name: file for file in tokenizer}
        ),
        attention_fingerprint=fingerprint_metadata(config),
        position_fingerprint=fingerprint_metadata(
            {
                "config": config,
                "context_length": context_length
                or config.get("max_position_embeddings"),
            }
        ),
        num_layers=config["num_hidden_layers"],
        num_kv_heads=(
            config["num_attention_heads"]
            if config.get("num_key_value_heads") is None
            else config["num_key_value_heads"]
        ),
        head_dim=(
            config["hidden_size"] // config["num_attention_heads"]
            if config.get("head_dim") is None
            else config["head_dim"]
        ),
        block_size=64,
        logical_dtype="float16",
    )


async def prepare_pd_request(
    cache_config: dict, tokens: list[int], params: dict
) -> None:
    """Reject incompatible rooms in the API process before entering EngineCore."""
    import asyncio

    import xoscar as xo

    from ..sglang.xavier.settings import transfer_timeout

    if cache_config["role"] == "decode" and len(tokens) < 2:
        raise ValueError("Cross-engine vLLM decode requires at least two prompt tokens")
    handoff = params.get("sglang_xavier")
    mode = "host" if cache_config.get("host_handoff") else "gpu"
    if not isinstance(handoff, dict) or handoff.get("mode") != mode:
        raise ValueError("Missing cross-engine Xavier handoff room")
    digest = prompt_digest(tokens)
    directory = await xo.actor_ref(
        address=cache_config["address"], uid=cache_config["uid"]
    )
    await asyncio.wait_for(
        directory.prepare(
            handoff["room"],
            KVCacheContract.from_dict(cache_config["contract"]).fingerprint,
            digest,
            cache_config["role"],
            rank=cache_config["rank"],
            timeout=transfer_timeout(),
            prompt_tokens=len(tokens),
        ),
        timeout=transfer_timeout(),
    )
    params["xavier_prompt_digest"] = digest
