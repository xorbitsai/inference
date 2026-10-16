# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Request-owned host pages for a Metal producer and NVIDIA consumer."""

import asyncio
import time

import xoscar as xo

from ....sglang.xavier.settings import transfer_timeout
from ...contract import KVCacheContract


class XavierHostPDSource(xo.StatelessActor):
    def __init__(self, directory, rank: int, capacity_bytes: int = 512 * 1024**2):
        super().__init__()
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("Xavier host PD byte budget must be positive")
        self.directory, self.rank = directory, rank
        self.capacity_bytes = capacity_bytes
        self.contract: KVCacheContract | None = None
        self.rooms: dict[int, dict] = {}
        self.active_bytes = 0
        self.host_bytes = 0
        self._capacity_changed = asyncio.Event()
        self._waiting: set[int] = set()
        self._closed = False

    async def __post_create__(self):
        await self.directory.register_peer(self.rank, self.address)

    async def __pre_destroy__(self):
        self._closed = True
        self.rooms.clear()
        self._waiting.clear()
        self.active_bytes = 0
        self._capacity_changed.set()
        await self.directory.unregister_peer(self.rank, self.address)

    def _expire(self):
        for room, state in list(self.rooms.items()):
            if state["deadline"] <= time.monotonic():
                self.release(room)

    async def reserve(self, room: int, metadata: dict, prompt_tokens: int) -> None:
        """Reserve rounded page capacity before doing any Metal prefill work."""
        self._expire()
        contract = KVCacheContract.from_dict(metadata)
        if self.contract is not None:
            self.contract.require_match(contract)
        page_bytes = contract.num_layers * contract.layer_nbytes
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError("Invalid Xavier host PD prompt length")
        if page_bytes > 64 * 1024**2:
            raise ValueError("Xavier host PD page exceeds 64 MiB")
        nbytes = (
            (prompt_tokens + contract.block_size - 1)
            // contract.block_size
            * page_bytes
        )
        if nbytes > self.capacity_bytes:
            raise ValueError("Xavier host PD prompt exceeds the in-flight byte budget")
        if room in self.rooms or room in self._waiting:
            raise ValueError("Duplicate Xavier host PD source")
        deadline = time.monotonic() + transfer_timeout()
        self._waiting.add(room)
        try:
            await self.directory.configure(contract.fingerprint, contract.to_dict())
            if self.contract is not None:
                self.contract.require_match(contract)
            self.contract = contract
            info = await self.directory.request_info(room)
            if info["prompt_tokens"] != prompt_tokens:
                raise ValueError("Xavier host PD prompt length differs")
            while True:
                self._expire()
                if self._closed:
                    raise RuntimeError("Xavier host PD source was destroyed")
                if room not in self._waiting:
                    raise RuntimeError("Xavier host PD reservation was cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Xavier host PD capacity wait timed out")
                self._capacity_changed.clear()
                if (
                    len(self.rooms) < 4096
                    and self.active_bytes + nbytes <= self.capacity_bytes
                ):
                    break
                expiry = min(s["deadline"] for s in self.rooms.values())
                try:
                    await asyncio.wait_for(
                        self._capacity_changed.wait(),
                        timeout=max(0, min(expiry, deadline) - time.monotonic()),
                    )
                except asyncio.TimeoutError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Xavier host PD capacity wait timed out")
            self.rooms[room] = dict(
                pages=None,
                prompt_tokens=prompt_tokens,
                nbytes=nbytes,
                deadline=deadline,
            )
            self.active_bytes += nbytes
        finally:
            self._waiting.discard(room)

    async def publish(self, room, metadata, pages, first_token, prompt_tokens):
        self._expire()
        state = self.rooms.get(room)
        if state is None:
            raise RuntimeError("Unreserved or expired Xavier host PD source")
        if state["pages"] is not None:
            raise ValueError("Duplicate Xavier host PD source")
        try:
            contract = KVCacheContract.from_dict(metadata)
            assert self.contract is not None
            self.contract.require_match(contract)
            page_bytes = contract.num_layers * contract.layer_nbytes
            if (
                type(prompt_tokens) is not int
                or prompt_tokens != state["prompt_tokens"]
                or type(first_token) is not int
                or first_token < 0
                or len(pages) * page_bytes != state["nbytes"]
                or any(type(p) is not bytes or len(p) != page_bytes for p in pages)
            ):
                raise ValueError("Invalid Xavier host PD pages or first token")
            state.update(pages=tuple(pages), first_token=first_token)
            await self.directory.publish_source(
                room,
                dict(
                    address=self.address,
                    uid=self.uid,
                    rank=self.rank,
                    transport="host",
                ),
            )
        except BaseException:
            self.release(room)
            raise

    def read(self, room: int, start: int = 0) -> dict:
        self._expire()
        state = self.rooms.get(room)
        if state is None:
            raise RuntimeError("Xavier host PD source expired or was cancelled")
        if state["pages"] is None:
            raise RuntimeError("Xavier host PD source has not been published")
        if type(start) is not int or not 0 <= start < len(state["pages"]):
            raise ValueError("Invalid Xavier host PD page cursor")
        assert self.contract is not None
        page_bytes = self.contract.num_layers * self.contract.layer_nbytes
        count = min(16, 64 * 1024**2 // page_bytes)
        pages = state["pages"][start : start + count]
        state["deadline"] = time.monotonic() + transfer_timeout()
        self.host_bytes += len(pages) * page_bytes
        return dict(
            pages=pages,
            next=start + len(pages),
            total=len(state["pages"]),
            first_token=state["first_token"],
            prompt_tokens=state["prompt_tokens"],
        )

    def release(self, room: int) -> None:
        if room in self._waiting:
            self._waiting.discard(room)
            self._capacity_changed.set()
        state = self.rooms.pop(room, None)
        if state is not None:
            self.active_bytes -= state["nbytes"]
            self._capacity_changed.set()

    def abort(self, room: int) -> None:
        self.release(room)

    def get_stats(self) -> dict:
        self._expire()
        return dict(
            active_rooms=len(self.rooms),
            active_bytes=self.active_bytes,
            host_bytes=self.host_bytes,
        )
