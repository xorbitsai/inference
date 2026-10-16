# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Canonical Xavier pages for full-attention MLX caches and explicit P/D."""

import asyncio
import importlib.metadata
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import xoscar as xo

from ....constants import XINFERENCE_CACHE_DIR
from ..xavier.constants import CROSS_ENGINE_TRANSFER_ACTOR_UID
from ..xavier.contract import (
    KVCacheContract,
    build_prefix_keys,
    fingerprint_files,
    fingerprint_metadata,
)
from ..xavier.pd_contract import build_pd_contract, prompt_digest

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
        or any(t != "full_attention" for t in (config.get("layer_types") or []))
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
    # Loading/configuration run in a worker thread. Materialize casts there so
    # generation never inherits unevaluated operations on its Metal stream.
    mx.eval(model.parameters())
    args = asdict(model.args)
    if cache_config.get("heterogeneous"):
        if cache_config.get("role") not in (
            "prefill",
            "decode",
        ) or not cache_config.get("host_handoff"):
            raise ValueError("Cross-engine MLX Xavier requires a host PD handoff")
        # The shared contract describes checkpoint semantics, not engine-specific
        # defaults. Verify MLX's effective geometry and RoPE before using it.
        for name in (
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "hidden_size",
            "head_dim",
            "rope_theta",
            "rope_scaling",
            "max_position_embeddings",
            "tie_word_embeddings",
        ):
            if config.get(name) is not None and args.get(name) != config[name]:
                raise ValueError(f"Cross-engine MLX effective model override: {name}")
        if args.get("rope_traditional") or model_config.get("lora_modules"):
            raise ValueError("Cross-engine MLX requires standard RoPE without LoRA")
        contract = build_pd_contract(
            str(model_path), model_config.get("context_length")
        )
        if (
            args.get("head_dim") or args["hidden_size"] // args["num_attention_heads"]
        ) != contract.head_dim or (
            args.get("num_key_value_heads") or args["num_attention_heads"]
        ) != contract.num_kv_heads:
            raise ValueError("Cross-engine MLX effective KV geometry differs")
        return MLXXavierCache(contract, cache_config)
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
            if self.config.get("heterogeneous"):
                await self._ref.configure(
                    self.contract.fingerprint, self.contract.to_dict()
                )
            else:
                metadata = await self._ref.configure(self.contract.to_dict())
                self._capacity_pages = min(
                    metadata["capacity_pages"], metadata["max_keys"]
                )
            self._configured = True

    async def initialize(self):
        await asyncio.wait_for(self._ensure_configured(), timeout=30)

    async def _call(self, method, *args, timeout=30):
        async def invoke():
            await self._ensure_configured()
            return await getattr(self._ref, method)(*args)

        return await asyncio.wait_for(invoke(), timeout=timeout)

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
        if self.config.get("heterogeneous"):
            return await self._fetch_host(tokens, transfer)
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

    def _host_room(self, tokens, transfer, role):
        handoff = (transfer or {}).get("sglang_xavier")
        if (
            self.role != role
            or not tokens
            or not isinstance(handoff, dict)
            or handoff.get("mode") != "host"
            or not handoff.get("heterogeneous")
            or handoff.get("address") != self.config["address"]
            or handoff.get("uid") != self.config["uid"]
            or type(handoff.get("room")) is not int
            or not 0 < handoff["room"] < 2**63
        ):
            raise ValueError(f"Incompatible cross-engine MLX {role} host handoff")
        return handoff["room"]

    async def _cleanup_host(self, room, source):
        async def cleanup():
            try:
                if source is not None:
                    await source.abort(room)
            except Exception:
                logger.warning("Cross-engine source cleanup failed", exc_info=True)
            finally:
                try:
                    await self._call("release", room)
                except Exception:
                    logger.warning(
                        "Cross-engine directory release failed", exc_info=True
                    )

        await asyncio.shield(cleanup())

    async def _fetch_host(self, tokens, transfer):
        from ..sglang.xavier.settings import transfer_timeout

        room = self._host_room(tokens, transfer, "decode")
        sender = None
        try:
            await self._call(
                "prepare",
                room,
                self.contract.fingerprint,
                prompt_digest(tokens),
                "decode",
                self.config.get("rank"),
                transfer_timeout(),
                len(tokens),
            )
            deadline = time.monotonic() + transfer_timeout()

            async def rpc(call):
                return await asyncio.wait_for(
                    call, timeout=max(0, deadline - time.monotonic())
                )

            source = await rpc(
                self._call("wait_source", room, timeout=transfer_timeout())
            )
            sender = await rpc(
                xo.actor_ref(
                    address=source["address"],
                    uid=f"{CROSS_ENGINE_TRANSFER_ACTOR_UID}-{source['rank']}",
                )
            )
            pages, index, nbytes = [], 0, 0
            expected = (
                len(tokens) + self.contract.block_size - 1
            ) // self.contract.block_size
            while True:
                chunk = await rpc(sender.wait_chunk(room, index))
                start = 0
                while start < len(chunk["pages"]):
                    result = await rpc(sender.export_host_pages(room, index, start))
                    payload = result["pages"]
                    if (
                        not payload
                        or result["next"] != start + len(payload)
                        or result["next"] > len(chunk["pages"])
                        or any(
                            type(page) is not bytes
                            or len(page)
                            != self.contract.num_layers * self.contract.layer_nbytes
                            for page in payload
                        )
                    ):
                        raise ValueError("Incomplete cross-engine host pages")
                    pages.extend(payload)
                    nbytes += sum(map(len, payload))
                    start = result["next"]
                    if len(pages) > expected:
                        raise ValueError("Cross-engine host page count differs")
                await rpc(sender.release_chunk(chunk["ticket"]))
                index += 1
                if chunk["final"]:
                    break
            if len(pages) != expected:
                raise ValueError("Incomplete cross-engine host prefix")
            length = len(tokens) - 1
            count = (length + self.contract.block_size - 1) // self.contract.block_size
            cache = self.decode(pages[:count], length)
            await self._call("complete", room, 0, length, nbytes)
            await self._call("release", room, "decode")
            return cache, length
        except BaseException:
            await self._cleanup_host(room, sender)
            raise

    def publish(self, cache, tokens, cached_tokens=0):
        if self.config.get("heterogeneous"):
            return

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

    async def _prefill_host(self, model, tokens, prefix_length, transfer, config):
        import mlx.core as mx
        from mlx_lm.generate import PromptProcessingBatch
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        from ..sglang.xavier.settings import transfer_timeout

        room = self._host_room(tokens, transfer, "prefill")
        source = None
        try:
            await self._call(
                "prepare",
                room,
                self.contract.fingerprint,
                prompt_digest(tokens),
                "prefill",
                self.config["rank"],
                transfer_timeout(),
                len(tokens),
            )
            source = await xo.actor_ref(
                address=self.config["source_address"], uid=self.config["source_uid"]
            )
            await asyncio.wait_for(
                source.reserve(room, self.contract.to_dict(), len(tokens)),
                timeout=transfer_timeout(),
            )
            batch = PromptProcessingBatch(
                model, [0], [make_prompt_cache(model)], prefill_step_size=2048
            )
            start = 0
            while start < len(tokens) - 1:
                end = min(start + 2048, len(tokens) - 1)
                if prefix_length is not None and start < prefix_length < end:
                    end = prefix_length
                batch.prompt([tokens[start:end]])
                start = end
                await asyncio.sleep(0)
            # Match GenerationBatch's final prompt step, without generating a
            # second token and extending KV beyond the prompt. SGLang consumes
            # this sampled token together with the full prompt cache.
            logits = model(mx.array([[tokens[-1]]]), cache=batch.prompt_cache)[:, -1, :]
            for processor in make_logits_processors(
                logit_bias=config.get("logit_bias"),
                repetition_penalty=config.get("repetition_penalty"),
                repetition_context_size=config.get("repetition_context_size", 20),
            ):
                logits = processor(mx.array(tokens), logits)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            sampler = make_sampler(
                temp=config.get("temperature", 0),
                top_p=config.get("top_p", 1),
                top_k=int(config.get("top_k") or 0),
            )
            first_token = int(sampler(logprobs).item())
            pages = self.encode(batch.extract_cache(0), tokens)
            await source.publish(
                room, self.contract.to_dict(), pages, first_token, len(tokens)
            )
            # CPU ownership now belongs to the source actor. Native engine slots
            # may only be committed after the consumer completes its import.
            del pages, batch
            await self._call("wait_complete", room, timeout=transfer_timeout())
            await source.release(room)
            await self._call("release", room, "prefill")
            return dict(transfer)
        except BaseException:
            await self._cleanup_host(room, source)
            raise

    async def prefill(
        self, model, tokens, prefix_length=None, transfer=None, generate_config=None
    ):
        from mlx_lm.generate import PromptProcessingBatch
        from mlx_lm.models.cache import make_prompt_cache

        if self.config.get("heterogeneous"):
            return await self._prefill_host(
                model, tokens, prefix_length, transfer, generate_config or {}
            )
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
