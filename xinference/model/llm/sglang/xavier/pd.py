# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Validate paired prompts and native GPU handoff completion."""

import asyncio
import hashlib
import json
import time

import xoscar as xo


class SGLangXavierHandoff:
    def __init__(self, cache_config: dict, page_size: int, tokenizer):
        self.config, self.tokenizer = cache_config, tokenizer
        self.role = cache_config["role"]

    async def _call(self, method, *args):
        async def invoke():
            actor = await xo.actor_ref(
                address=self.config["address"], uid=self.config["uid"]
            )
            return await getattr(actor, method)(*args)

        return await asyncio.wait_for(invoke(), timeout=10)

    async def prepare(self, prompt: str, transfer: dict) -> dict:
        handoff = transfer.get("sglang_xavier")
        if not isinstance(handoff, dict) or handoff.get("mode") != "gpu":
            raise ValueError("Missing SGLang Xavier GPU bootstrap metadata")
        tokens = self.tokenizer.encode(prompt)
        prompt_hash = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        namespace = (await self._call("get_stats"))["namespace"]
        await self._call(
            "prepare", handoff.get("room"), namespace, prompt_hash, self.role
        )
        return handoff

    async def accept(self, prompt: str, transfer: dict) -> dict:
        if self.role != "decode":
            raise ValueError("Remote prefill requires a SGLang decode replica")
        return await self.prepare(prompt, transfer)

    async def publish(self, handoff: dict) -> dict:
        deadline = time.monotonic() + 30
        while not await self._call("check", handoff["room"]):
            if time.monotonic() > deadline:
                raise RuntimeError("SGLang Xavier GPU transfer did not complete")
            await asyncio.sleep(0.005)
        await self.release(handoff)
        return dict(
            do_remote_prefill=True, do_remote_decode=False, sglang_xavier=handoff
        )

    async def check_hit(self, meta_info: dict, handoff: dict) -> None:
        if not await self._call("check", handoff["room"]):
            raise RuntimeError(
                "SGLang Xavier decode did not complete its GPU KV transfer"
            )

    async def release(self, handoff: dict, failed: bool = False) -> None:
        await self._call("release", handoff["room"], None if failed else self.role)
