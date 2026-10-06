# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Validate paired prompts and native GPU handoff completion."""

import asyncio
import hashlib
import json
from collections import OrderedDict
from typing import Optional

import xoscar as xo

from .settings import transfer_timeout


class SGLangXavierHandoff:
    def __init__(self, cache_config: dict, tokenizer):
        self.config, self.tokenizer = cache_config, tokenizer
        self.role = cache_config["role"]
        # The directory and namespace live for this model's deployment. Keep
        # request validation/completion as RPCs, without resolving them again.
        self._directory_actor: Optional[xo.ActorRef] = None
        self._namespace: Optional[str] = None
        # Keep fixed-size digests only, scoped to this deployment/tokenizer.
        self._prompt_hashes: OrderedDict[bytes, str] = OrderedDict()
        self._lease_tasks: dict[int, asyncio.Task] = {}

    async def _call(self, method, *args, timeout=10):
        async def invoke():
            if self._directory_actor is None:
                self._directory_actor = await xo.actor_ref(
                    address=self.config["address"], uid=self.config["uid"]
                )
            return await getattr(self._directory_actor, method)(*args)

        return await asyncio.wait_for(invoke(), timeout=timeout)

    async def prepare(self, prompt: str, transfer: dict) -> dict:
        handoff = transfer.get("sglang_xavier")
        if not isinstance(handoff, dict) or handoff.get("mode") != "gpu":
            raise ValueError("Missing SGLang Xavier GPU bootstrap metadata")
        key = hashlib.sha256(prompt.encode()).digest()
        prompt_hash = self._prompt_hashes.pop(key, None)
        if prompt_hash is None:
            tokens = self.tokenizer.encode(prompt)
            prompt_hash = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        self._prompt_hashes[key] = prompt_hash
        if len(self._prompt_hashes) > 256:
            self._prompt_hashes.popitem(last=False)
        if self._namespace is None:
            self._namespace = (await self._call("get_stats"))["namespace"]
            if self._namespace is None:
                raise ValueError("Unregistered SGLang Xavier PD namespace")
        await self._call(
            "prepare",
            handoff.get("room"),
            self._namespace,
            prompt_hash,
            self.role,
            self.config.get("rank"),
            transfer_timeout(),
        )
        return handoff

    async def accept(self, prompt: str, transfer: dict) -> dict:
        if self.role != "decode":
            raise ValueError("Remote prefill requires a SGLang decode replica")
        handoff = await self.prepare(prompt, transfer)
        room = handoff["room"]
        self._lease_tasks[room] = asyncio.create_task(self._renew_completed(room))
        return handoff

    async def _renew_completed(self, room):
        # Completed records are needed by non-streaming decode after generation.
        # Keep them while D is alive; lost cleanup/router processes remain bounded.
        while True:
            await asyncio.sleep(min(60, transfer_timeout() / 3))
            try:
                if not await self._call("renew_completed", room):
                    return
            except Exception:
                # Retry transient RPC errors; completion still fails closed if
                # the directory stayed unavailable until the lease expired.
                continue

    async def publish(self, handoff: dict) -> dict:
        if not await self._call(
            "wait_complete", handoff["room"], timeout=transfer_timeout() + 10
        ):
            raise RuntimeError("SGLang Xavier GPU transfer did not complete")
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
        task = self._lease_tasks.pop(handoff["room"], None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._call("release", handoff["room"], None if failed else self.role)
