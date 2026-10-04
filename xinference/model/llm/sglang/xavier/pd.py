# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Explicit P/D handoff over Xavier's leased HiCache pages.

Only complete prompt pages move to D. SGLang computes the final page locally
to recover the next-token logits, so this CPU path is a correctness baseline.
"""

import asyncio
import time
import uuid

import xoscar as xo


class SGLangXavierHandoff:
    def __init__(self, cache_config: dict, page_size: int, tokenizer):
        self.config = cache_config
        self.page_size = page_size
        self.tokenizer = tokenizer
        self.role = cache_config["role"]

    async def _call(self, method: str, *args):
        async def invoke():
            actor = await xo.actor_ref(
                address=self.config["address"], uid=self.config["uid"]
            )
            return await getattr(actor, method)(*args)

        return await asyncio.wait_for(invoke(), timeout=10)

    def _keys(self, prompt: str, cache_salt: str) -> list[str]:
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.utils import get_storage_hash_str

        tokens = self.tokenizer.encode(prompt)
        # The last prompt token needs a forward pass for sampling logits.
        count = max(0, (len(tokens) - 1) // self.page_size * self.page_size)
        return (
            get_storage_hash_str(
                RadixKey(tokens[:count], cache_salt=cache_salt),
                page_size=self.page_size,
            )
            if count
            else []
        )

    async def prepare(self, prompt: str) -> dict:
        if self.role != "prefill":
            raise ValueError("Remote decode requires a SGLang prefill replica")
        # A request-specific namespace forces P to publish all prompt pages,
        # even if a prior request remains in its local radix cache while L3
        # copies have been evicted. P history reuse is a later optimization.
        salt = "xavier-pd-" + uuid.uuid4().hex
        keys = self._keys(prompt, salt)
        namespace = (await self._call("get_stats"))["namespace"]
        ticket = await self._call("prepare_handoff", namespace, keys)
        return {
            "engine": "sglang",
            "address": self.config["address"],
            "uid": self.config["uid"],
            "namespace": namespace,
            "keys": keys,
            "ticket": ticket,
            "cache_salt": salt,
            "cached_tokens": len(keys) * self.page_size,
        }

    async def publish(self, handoff: dict) -> dict:
        deadline = time.monotonic() + 30
        while True:
            if await self._call(
                "validate_handoff",
                handoff["namespace"],
                handoff["keys"],
                handoff["ticket"],
            ):
                return {
                    "do_remote_prefill": True,
                    "do_remote_decode": False,
                    "sglang_xavier": handoff,
                }
            if time.monotonic() >= deadline:
                raise RuntimeError("SGLang prefill did not publish its Xavier KV pages")
            await asyncio.sleep(0.05)

    async def accept(self, prompt: str, transfer: dict) -> dict:
        if self.role != "decode":
            raise ValueError("Remote prefill requires a SGLang decode replica")
        handoff = transfer.get("sglang_xavier")
        if not isinstance(handoff, dict) or not isinstance(
            handoff.get("cache_salt"), str
        ):
            raise ValueError("SGLang Xavier handoff differs from the decode prompt")
        keys = self._keys(prompt, handoff["cache_salt"])
        if (
            not isinstance(handoff, dict)
            or handoff.get("engine") != "sglang"
            or handoff.get("address") != self.config["address"]
            or handoff.get("uid") != self.config["uid"]
            or handoff.get("keys") != keys
            or handoff.get("cached_tokens") != len(keys) * self.page_size
        ):
            raise ValueError("SGLang Xavier handoff differs from the decode prompt")
        if not await self._call(
            "validate_handoff", handoff.get("namespace"), keys, handoff.get("ticket")
        ):
            raise RuntimeError("SGLang Xavier handoff expired or is unavailable")
        return handoff

    @staticmethod
    def check_hit(meta_info: dict, handoff: dict) -> None:
        if meta_info.get("cached_tokens", 0) < handoff["cached_tokens"]:
            raise RuntimeError("SGLang decode did not restore the Xavier PD prefix")

    async def release(self, handoff: dict) -> None:
        await self._call("release_handoff", handoff["ticket"])
