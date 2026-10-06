# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Bounded immutable CPU pages, without a Torch or Metal dependency."""

import time
import uuid
from collections import OrderedDict

import xoscar as xo

from ...contract import KVCacheContract


class XavierBytesCacheActor(xo.Actor):
    def __init__(self, capacity_bytes: int = 512 * 1024 * 1024):
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("Xavier cache capacity must be a positive integer")
        self.capacity_bytes = capacity_bytes
        self._contract = None
        self._pages: OrderedDict[str, bytes] = OrderedDict()
        self._handoffs: dict = {}
        self._counts = dict(
            stored_pages=0,
            read_pages=0,
            missed_pages=0,
            evicted_pages=0,
            handoff_reads=0,
        )

    def configure(self, metadata):
        contract = KVCacheContract.from_dict(metadata)
        if self._contract is not None:
            self._contract.require_match(contract)
        else:
            page_bytes = contract.layer_nbytes * contract.num_layers
            if page_bytes > self.capacity_bytes:
                raise ValueError("Xavier cache budget cannot hold one complete page")
            self._contract = contract
            self._page_bytes = page_bytes
            self._capacity = self.capacity_bytes // page_bytes
        return contract.fingerprint

    def _validate(self, namespace, keys):
        if self._contract is None or namespace != self._contract.fingerprint:
            raise ValueError("Unregistered or incompatible Xavier cache namespace")
        if (
            not isinstance(keys, (list, tuple))
            or len(keys) > 4096
            or any(
                not isinstance(key, str)
                or len(key) != 64
                or any(c not in "0123456789abcdef" for c in key)
                for key in keys
            )
            or len(set(keys)) != len(keys)
        ):
            raise ValueError("Invalid Xavier page keys")
        self._expire()

    def _expire(self):
        for ticket, entry in list(self._handoffs.items()):
            if entry["deadline"] <= time.monotonic():
                self.release_handoff(ticket)

    def _pinned(self):
        return {key for entry in self._handoffs.values() for key in entry["keys"]}

    def put(self, namespace, keys, pages, start=0):
        self._validate(namespace, keys)
        if (
            type(start) is not int
            or not 0 <= start <= len(keys)
            or len(keys) - start != len(pages)
            or any(
                type(page) is not bytes or len(page) != self._page_bytes
                for page in pages
            )
        ):
            raise ValueError("Xavier page geometry differs from its contract")
        if len(keys) > self._capacity:
            return [False] * len(keys)
        pinned = self._pinned()
        for key, page in reversed(list(zip(keys[start:], pages))):
            if key not in self._pages:
                if len(self._pages) >= self._capacity:
                    victim = next((k for k in self._pages if k not in pinned), None)
                    if victim is None:
                        continue
                    self._pages.pop(victim)
                    self._counts["evicted_pages"] += 1
                self._pages[key] = page
                self._counts["stored_pages"] += 1
        # Touch the whole chain, including its already cached prefix, so suffix
        # publication also keeps heads newer than tails. Pages remain immutable.
        for key in reversed(keys):
            if key in self._pages:
                self._pages.move_to_end(key)
        return [key in self._pages for key in keys]

    def get(self, namespace, keys):
        self._validate(namespace, keys)
        pages = []
        for key in keys:
            if key not in self._pages:
                self._counts["missed_pages"] += 1
                break
            pages.append(self._pages[key])
        # Keep chain heads newer than tails so eviction preserves usable prefixes.
        for key in reversed(keys[: len(pages)]):
            self._pages.move_to_end(key)
        self._counts["read_pages"] += len(pages)
        return pages

    def prepare_handoff(self, namespace, keys):
        self._validate(namespace, keys)
        pinned = self._pinned().union(keys)
        if len(pinned) > self._capacity:
            raise RuntimeError("Xavier cache budget cannot hold the active PD prefixes")
        # Reserve absent pages before P allocates or acknowledges a handoff.
        while len(set(self._pages).union(pinned)) > self._capacity:
            victim = next(k for k in self._pages if k not in pinned)
            self._pages.pop(victim)
            self._counts["evicted_pages"] += 1
        ticket = uuid.uuid4().hex
        self._handoffs[ticket] = dict(
            keys=list(keys), deadline=time.monotonic() + 300, claimed=False
        )
        return ticket

    def get_handoff(self, namespace, keys, ticket):
        self._validate(namespace, keys)
        entry = self._handoffs.get(ticket)
        if (
            entry is None
            or entry["keys"] != list(keys)
            or entry["claimed"]
            or any(key not in self._pages for key in keys)
        ):
            raise ValueError(
                "Missing, incomplete or already consumed Xavier PD handoff"
            )
        entry["claimed"] = True
        pages = [self._pages[key] for key in keys]
        for key in reversed(keys):
            self._pages.move_to_end(key)
        self._counts["read_pages"] += len(pages)
        self._counts["handoff_reads"] += 1
        return pages

    def handoff_ready(self, namespace, keys, ticket):
        self._validate(namespace, keys)
        entry = self._handoffs.get(ticket)
        if entry is None or entry["keys"] != list(keys) or entry["claimed"]:
            raise ValueError("Missing or already consumed Xavier PD handoff")
        return all(key in self._pages for key in keys)

    def release_handoff(self, ticket):
        self._handoffs.pop(ticket, None)

    def get_stats(self):
        self._expire()
        return {
            **self._counts,
            "namespace": self._contract.fingerprint if self._contract else None,
            "pages": len(self._pages),
            "used_bytes": len(self._pages) * getattr(self, "_page_bytes", 0),
            "capacity_bytes": self.capacity_bytes,
            "active_handoffs": len(self._handoffs),
        }
