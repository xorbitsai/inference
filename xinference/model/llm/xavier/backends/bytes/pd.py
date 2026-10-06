# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Request-owned host pages for a Metal producer and NVIDIA consumer."""

import time

import xoscar as xo

from ....sglang.xavier.settings import transfer_timeout
from ...contract import KVCacheContract


class XavierHostPDSource(xo.Actor):
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

    async def __post_create__(self):
        await self.directory.register_peer(self.rank, self.address)

    async def __pre_destroy__(self):
        self.rooms.clear()
        self.active_bytes = 0
        await self.directory.unregister_peer(self.rank, self.address)

    def _expire(self):
        for room, state in list(self.rooms.items()):
            if state["deadline"] <= time.monotonic():
                self.release(room)

    async def publish(self, room, metadata, pages, first_token, prompt_tokens):
        self._expire()
        contract = KVCacheContract.from_dict(metadata)
        if self.contract is not None:
            self.contract.require_match(contract)
        page_bytes = contract.num_layers * contract.layer_nbytes
        if (
            type(prompt_tokens) is not int
            or prompt_tokens <= 0
            or type(first_token) is not int
            or first_token < 0
            or len(pages)
            != (prompt_tokens + contract.block_size - 1) // contract.block_size
            or any(type(p) is not bytes or len(p) != page_bytes for p in pages)
            or page_bytes > 64 * 1024**2
        ):
            raise ValueError("Invalid Xavier host PD pages or first token")
        nbytes = page_bytes * len(pages)
        if room in self.rooms:
            raise ValueError("Duplicate Xavier host PD source")
        if len(self.rooms) >= 4096 or self.active_bytes + nbytes > self.capacity_bytes:
            raise RuntimeError("Xavier host PD in-flight byte budget exceeded")
        await self.directory.configure(contract.fingerprint)
        info = await self.directory.request_info(room)
        if info["prompt_tokens"] != prompt_tokens:
            raise ValueError("Xavier host PD prompt length differs")
        self.contract = contract
        self.rooms[room] = dict(
            pages=tuple(pages),
            first_token=first_token,
            prompt_tokens=prompt_tokens,
            nbytes=nbytes,
            deadline=time.monotonic() + transfer_timeout(),
        )
        self.active_bytes += nbytes
        try:
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
        state = self.rooms.pop(room, None)
        if state is not None:
            self.active_bytes -= state["nbytes"]

    def abort(self, room: int) -> None:
        self.release(room)

    def get_stats(self) -> dict:
        self._expire()
        return dict(
            active_rooms=len(self.rooms),
            active_bytes=self.active_bytes,
            host_bytes=self.host_bytes,
        )
