# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Worker-local, bounded write-behind for optional P-side prefix history."""

import asyncio
import logging
import re
import time
from collections import Counter, deque
from typing import Optional

import xoscar as xo

from .contract import KVCacheContract

logger = logging.getLogger(__name__)


class XavierHistoryProxy(xo.StatelessActor):
    def __init__(self, target: dict, capacity_bytes: int, read_timeout: float = 30):
        super().__init__()
        self.target = dict(target)
        self.capacity_bytes = capacity_bytes
        self.read_timeout = read_timeout
        self._ref = None
        self._contract: Optional[KVCacheContract] = None
        self._pending: Counter[str] = Counter()
        self._queue: deque = deque()
        self._pending_bytes = 0
        self._writer = None
        self._read_retry_at = 0.0
        self._write_retry_at = 0.0
        self._closed = False
        self._counts = dict(
            enqueued_pages=0,
            written_pages=0,
            dropped_pages=0,
            read_pages=0,
            imported_pages=0,
        )

    async def _target(self):
        if self._ref is None:
            self._ref = await xo.actor_ref(**self.target)
        return self._ref

    async def configure(self, contract: dict) -> dict:
        c = KVCacheContract.from_dict(contract)
        if self._contract is not None:
            self._contract.require_match(c)
        metadata = await asyncio.wait_for(
            (await self._target()).configure(contract), self.read_timeout
        )
        self._contract = c
        self._page_bytes = c.num_layers * c.layer_nbytes
        self._capacity_pages = min(
            metadata["capacity_pages"], self.capacity_bytes // self._page_bytes, 4096
        )
        if not self._capacity_pages:
            raise ValueError("Xavier history queue cannot hold a complete KV page")
        return {**metadata, "capacity_pages": self._capacity_pages}

    def _validate(self, namespace, keys):
        if self._contract is None or namespace != self._contract.fingerprint:
            raise ValueError("Unregistered or incompatible Xavier history namespace")
        if not isinstance(keys, (list, tuple)) or len(keys) > 4096:
            raise ValueError("Invalid Xavier history page keys")
        if any(
            not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None
            for key in keys
        ) or len(set(keys)) != len(keys):
            raise ValueError("Invalid Xavier history page keys")

    def pending_prefix(self, namespace, keys):
        # This only deduplicates in-flight publication. LMCache can evict pages
        # independently, so a completed write is never a local residency hint.
        self._validate(namespace, keys)
        for index, key in enumerate(keys):
            if key not in self._pending:
                return index
        return len(keys)

    async def exists(self, namespace, keys):
        self._validate(namespace, keys)
        if self._closed or time.monotonic() < self._read_retry_at:
            return 0
        try:
            found = await asyncio.wait_for(
                (await self._target()).exists(namespace, keys),
                min(self.read_timeout, 1.0),
            )
        except Exception:
            self._read_retry_at = time.monotonic() + 5
            return 0
        return found

    async def get(self, namespace, keys):
        self._validate(namespace, keys)
        if self._closed or time.monotonic() < self._read_retry_at:
            return []
        try:
            pages = await asyncio.wait_for(
                (await self._target()).get(namespace, keys), self.read_timeout
            )
        except Exception:
            self._read_retry_at = time.monotonic() + 5
            logger.warning(
                "Xavier history lookup failed; computing locally", exc_info=True
            )
            return []
        self._counts["read_pages"] += len(pages)
        return pages

    def put(self, namespace, keys, pages, start=0):
        self._validate(namespace, keys)
        if (
            type(start) is not int
            or not 0 <= start <= len(keys)
            or len(pages) != len(keys) - start
            or any(
                type(page) is not bytes or len(page) != self._page_bytes
                for page in pages
            )
        ):
            raise ValueError("Invalid Xavier history page geometry")
        if not pages:
            return [True] * len(keys)
        nbytes = len(pages) * self._page_bytes
        if (
            self._closed
            or time.monotonic() < self._write_retry_at
            or self._pending_bytes + nbytes > self.capacity_bytes
        ):
            self._counts["dropped_pages"] += len(pages)
            return [False] * len(keys)
        self._queue.append((namespace, list(keys), tuple(pages), start, nbytes))
        self._pending_bytes += nbytes
        self._counts["enqueued_pages"] += len(pages)
        self._pending.update(keys[start:])
        if self._writer is None or self._writer.done():
            self._writer = asyncio.create_task(self._write())
        return [True] * len(keys)

    async def _write(self):
        while self._queue:
            namespace, keys, pages, start, nbytes = self._queue.popleft()
            try:
                if time.monotonic() < self._write_retry_at:
                    self._counts["dropped_pages"] += len(pages)
                    continue
                target = await self._target()
                found = await asyncio.wait_for(
                    target.exists(namespace, keys), self.read_timeout
                )
                if found < start:
                    # The caller's live prefix need not exist in history. Its
                    # suffix would be unreadable until the missing head is sent.
                    self._counts["dropped_pages"] += len(pages)
                    continue
                begin = found
                if begin == len(keys):
                    continue
                result = await asyncio.wait_for(
                    target.put(namespace, keys, list(pages[begin - start :]), begin),
                    self.read_timeout,
                )
                if not all(result):
                    raise RuntimeError("Xavier history store rejected pages")
                self._counts["written_pages"] += len(keys) - begin
            except Exception:
                self._counts["dropped_pages"] += len(pages)
                if time.monotonic() >= self._write_retry_at:
                    logger.warning(
                        "Xavier history write failed; live P/D continues", exc_info=True
                    )
                self._write_retry_at = time.monotonic() + 5
            finally:
                for key in keys[start:]:
                    self._pending[key] -= 1
                    if not self._pending[key]:
                        del self._pending[key]
                self._pending_bytes -= nbytes

    async def flush(self):
        if self._writer is not None:
            await asyncio.shield(self._writer)

    def record_import(self, namespace, pages):
        self._validate(namespace, [])
        if type(pages) is not int or pages <= 0:
            raise ValueError("Imported history page count must be positive")
        self._counts["imported_pages"] += pages

    def get_stats(self):
        return {
            **self._counts,
            "namespace": self._contract.fingerprint if self._contract else None,
            "capacity_bytes": self.capacity_bytes,
            "pending_bytes": self._pending_bytes,
        }

    async def __pre_destroy__(self):
        self._closed = True
        try:
            await asyncio.wait_for(self.flush(), self.read_timeout + 1)
        except Exception:
            pass
        finally:
            if self._writer is not None and not self._writer.done():
                self._writer.cancel()
                await asyncio.gather(self._writer, return_exceptions=True)
            self._writer = None
            self._queue.clear()
            self._pending.clear()
            self._pending_bytes = 0
