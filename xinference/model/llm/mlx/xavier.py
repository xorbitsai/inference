# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Canonical Xavier pages for full-attention MLX caches and explicit P/D."""

import asyncio
import importlib.metadata
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import xoscar as xo

from ....constants import XINFERENCE_CACHE_DIR
from ..xavier.contract import (
    KVCacheContract,
    build_prefix_keys,
    fingerprint_files,
    fingerprint_metadata,
)

logger = logging.getLogger(__name__)
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
_TOKENIZER_ASSETS = ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json")


def configure_xavier(model, model_path, model_config, cache_config, n_worker):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, make_prompt_cache
    from packaging.version import Version

    if (
        n_worker != 1
        or model_config.get("max_kv_size")
        or model_config.get("draft_model_path")
        or Version(importlib.metadata.version("mlx-lm")) < Version("0.31.2")
    ):
        raise ValueError(
            "MLX Xavier requires mlx-lm>=0.31.2, one worker and full unquantized KV caches"
        )
    path = Path(model_path)
    config = json.loads((path / "config.json").read_text())
    if (
        config.get("model_type") not in ("qwen2", "qwen3", "llama")
        or any(
            config.get(k)
            for k in (
                "quantization",
                "quantization_config",
                "vision_config",
                "auto_map",
                "use_sliding_window",
            )
        )
        or any(t != "full_attention" for t in config.get("layer_types", []))
    ):
        raise ValueError(
            "MLX Xavier supports unquantized full-attention Qwen2, Qwen3 and Llama text models"
        )
    if not all(type(c) is KVCache for c in make_prompt_cache(model)):
        raise ValueError("MLX Xavier requires ordinary full-attention KVCache layers")
    tokenizer_files = {
        p.name: p for p in path.iterdir() if p.name in _TOKENIZER_FILES and p.is_file()
    }
    if not any(n in tokenizer_files for n in _TOKENIZER_ASSETS):
        raise ValueError("MLX Xavier requires local tokenizer assets")
    tokenizer_config = (
        json.loads(tokenizer_files["tokenizer_config.json"].read_text())
        if "tokenizer_config.json" in tokenizer_files
        else {}
    )
    if tokenizer_config.get("auto_map"):
        raise ValueError("MLX Xavier requires a standard tokenizer")
    weights = {p.name: p for p in path.glob("*.safetensors")}
    if not weights:
        raise ValueError("MLX Xavier requires local safetensors weights")
    model.set_dtype(mx.float16)
    args = asdict(model.args)
    contract = KVCacheContract(
        weights_fingerprint=fingerprint_files(
            weights, cache_dir=Path(XINFERENCE_CACHE_DIR) / "xavier-fingerprints"
        ),
        tokenizer_fingerprint=fingerprint_files(tokenizer_files),
        attention_fingerprint=fingerprint_metadata(
            {"config": config, "effective_args": args, "dtype": "float16"}
        ),
        position_fingerprint=fingerprint_metadata(
            {"config": config, "effective_args": args}
        ),
        num_layers=args["num_hidden_layers"],
        num_kv_heads=args.get("num_key_value_heads", args["num_attention_heads"]),
        head_dim=args.get("head_dim")
        or args["hidden_size"] // args["num_attention_heads"],
        block_size=64,
        logical_dtype="float16",
    )
    return MLXXavierCache(contract, cache_config)


class MLXXavierCache:
    def __init__(self, contract: KVCacheContract, config: dict):
        self.contract, self.config = contract, dict(config)
        self.role = config.get("role")
        self._ref = None
        self._configured = False
        self._capacity_pages = 0
        self._writes: set = set()
        self.imported_tokens = 0

    async def _ensure_configured(self):
        if self._ref is None:
            self._ref = await xo.actor_ref(
                address=self.config["address"], uid=self.config["uid"]
            )
        if not self._configured:
            metadata = await self._ref.configure(self.contract.to_dict())
            self._capacity_pages = min(metadata["capacity_pages"], metadata["max_keys"])
            self._configured = True

    async def initialize(self):
        await asyncio.wait_for(self._ensure_configured(), timeout=30)

    async def _call(self, method, *args):
        async def invoke():
            await self._ensure_configured()
            return await getattr(self._ref, method)(*args)

        return await asyncio.wait_for(invoke(), timeout=30)

    def keys(self, tokens):
        keys = [key.digest for key in build_prefix_keys(self.contract, tokens)]
        remainder = len(tokens) % self.contract.block_size
        if remainder:
            keys.append(
                fingerprint_metadata(
                    {
                        "format": "xavier-partial-page-v1",
                        "contract": self.contract.fingerprint,
                        "previous": keys[-1] if keys else None,
                        "tokens": tokens[-remainder:],
                        "prefix_tokens": len(tokens),
                    }
                )
            )
        return keys

    def encode(self, cache, tokens, start=0):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        c = self.contract
        if len(cache) != c.num_layers:
            raise ValueError("MLX cache layer count differs from its contract")
        length = len(tokens)
        if start % c.block_size or not 0 <= start <= length:
            raise ValueError("MLX page publication must start at a block boundary")
        if start == length:
            return []
        pages = (length - start + c.block_size - 1) // c.block_size
        data = np.zeros(
            (pages, c.num_layers, 2, c.block_size, c.num_kv_heads, c.head_dim),
            dtype="<f2",
        )
        for layer, entry in enumerate(cache):
            if type(entry) is not KVCache or entry.offset < length:
                raise ValueError("MLX cache type or offset differs from its contract")
            for kv, value in enumerate(entry.state):
                if (
                    value.dtype != mx.float16
                    or tuple(value.shape[:2]) != (1, c.num_kv_heads)
                    or value.shape[-1] != c.head_dim
                ):
                    raise ValueError(
                        "MLX KV geometry or dtype differs from its contract"
                    )
                flat = data[:, layer, kv].reshape(
                    pages * c.block_size, c.num_kv_heads, c.head_dim
                )
                flat[: length - start] = np.array(
                    value[0, :, start:length, :].transpose(1, 0, 2), copy=True
                )
                data[:, layer, kv] = flat.reshape(
                    pages, c.block_size, c.num_kv_heads, c.head_dim
                )
        return [page.tobytes() for page in data]

    def decode(self, pages, length):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        c = self.contract
        if len(pages) != (length + c.block_size - 1) // c.block_size or any(
            type(p) is not bytes or len(p) != c.layer_nbytes * c.num_layers
            for p in pages
        ):
            raise ValueError("Incomplete Xavier MLX prefix")
        if not length:
            return [KVCache() for _ in range(c.num_layers)]
        data = np.stack(
            [
                np.frombuffer(p, dtype="<f2").reshape(
                    c.num_layers, 2, c.block_size, c.num_kv_heads, c.head_dim
                )
                for p in pages
            ]
        )
        cache = []
        for layer in range(c.num_layers):
            entry = KVCache()
            entry.state = tuple(
                mx.array(
                    data[:, layer, kv]
                    .reshape(-1, c.num_kv_heads, c.head_dim)[:length]
                    .transpose(1, 0, 2)[None]
                    .copy()
                )
                for kv in (0, 1)
            )
            cache.append(entry)
        mx.eval([entry.state for entry in cache])
        self.imported_tokens += length
        return cache

    async def fetch(self, tokens, transfer=None):
        prefix = tokens[:-1]
        keys = self.keys(prefix)
        if transfer is not None:
            handoff = transfer.get("mlx_xavier", {})
            if (
                self.role != "decode"
                or handoff.get("engine") != "mlx"
                or handoff.get("address") != self.config["address"]
                or handoff.get("uid") != self.config["uid"]
                or handoff.get("namespace") != self.contract.fingerprint
                or handoff.get("keys") != keys
                or handoff.get("tokens") != len(prefix)
            ):
                raise ValueError("Incompatible MLX Xavier PD handoff")
            try:
                pages = await self._call(
                    "get_handoff", self.contract.fingerprint, keys, handoff["ticket"]
                )
                return self.decode(pages, len(prefix)), len(prefix)
            finally:
                await self._release_handoff(handoff["ticket"])
        try:
            pages = await self._call("get", self.contract.fingerprint, keys)
            length = min(len(prefix), len(pages) * self.contract.block_size)
            return (self.decode(pages, length), length) if pages else (None, 0)
        except Exception:
            logger.warning(
                "MLX Xavier prefix fetch failed; treating it as a miss", exc_info=True
            )
            return None, 0

    async def _release_handoff(self, ticket):
        try:
            await asyncio.shield(self._call("release_handoff", ticket))
        except Exception:
            logger.warning("MLX Xavier handoff release failed", exc_info=True)

    def publish(self, cache, tokens, cached_tokens=0):
        async def write():
            try:
                await self.initialize()
                if (
                    len(tokens) + self.contract.block_size - 1
                ) // self.contract.block_size > self._capacity_pages:
                    return
                start = (
                    cached_tokens // self.contract.block_size * self.contract.block_size
                )
                keys = self.keys(tokens)
                pages = self.encode(cache, tokens, start)
                await self._call(
                    "put",
                    self.contract.fingerprint,
                    keys,
                    pages,
                    start // self.contract.block_size,
                )
            except Exception:
                logger.warning("MLX Xavier prefix publication failed", exc_info=True)

        task = asyncio.create_task(write())
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)
        return task

    async def flush(self, writes):
        if writes:
            await asyncio.gather(
                *(asyncio.shield(task) for task in writes), return_exceptions=True
            )

    async def prefill(self, model, tokens, prefix_length=None):
        from mlx_lm.generate import PromptProcessingBatch
        from mlx_lm.models.cache import make_prompt_cache

        if self.role != "prefill" or not tokens:
            raise ValueError(
                "MLX remote prefill requires a prefill replica and a nonempty prompt"
            )
        prefix = tokens[:-1]
        keys = self.keys(prefix)
        ticket = await self._call("prepare_handoff", self.contract.fingerprint, keys)
        try:
            metadata = dict(
                do_remote_prefill=True,
                do_remote_decode=False,
                mlx_xavier=dict(
                    engine="mlx",
                    address=self.config["address"],
                    uid=self.config["uid"],
                    namespace=self.contract.fingerprint,
                    keys=keys,
                    tokens=len(prefix),
                    ticket=ticket,
                ),
            )
            if await self._call(
                "handoff_ready", self.contract.fingerprint, keys, ticket
            ):
                # Pages are already pinned. Avoid uploading a warm prefix to P's
                # GPU and copying it back to CPU solely to hand it to D.
                return metadata
            cache, cached = await self.fetch(tokens)
            cache = cache if cache is not None else make_prompt_cache(model)
            # Use the same batched cache/mask and prompt segment boundaries as
            # ordinary MLX generation. Scalar KVCache prefill uses a different
            # SDPA path and can change greedy results on real FP16 checkpoints.
            batch = PromptProcessingBatch(
                model, [0], [cache], tokens=[prefix[:cached]], prefill_step_size=2048
            )
            start = cached
            while start < len(prefix):
                end = min(start + 2048, len(prefix))
                if prefix_length is not None and start < prefix_length < end:
                    end = prefix_length
                batch.prompt([prefix[start:end]])
                start = end
                await asyncio.sleep(0)
            cache = batch.extract_cache(0)
            start = cached // self.contract.block_size * self.contract.block_size
            if not all(
                await self._call(
                    "put",
                    self.contract.fingerprint,
                    keys,
                    self.encode(cache, prefix, start),
                    start // self.contract.block_size,
                )
            ):
                raise RuntimeError(
                    "MLX Xavier could not publish the complete PD prefix"
                )
            return metadata
        except BaseException:
            await self._release_handoff(ticket)
            raise
