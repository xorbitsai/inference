# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Independent lifetimes for explicit MLX prompt-cache checkpoints."""

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Checkpoint:
    cache: Any
    ttl: int
    expires: float
    nbytes: int


class TimedPromptCache:
    def __init__(self, max_size: int = 4, max_bytes: Optional[int] = None):
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.entries: OrderedDict[Tuple[int, ...], Checkpoint] = OrderedDict()
        self._known: OrderedDict[Tuple[int, ...], None] = OrderedDict()

    def fetch(
        self,
        tokens: List[int],
        checkpoints: Optional[List[int]] = None,
        lifetimes: Optional[Dict[int, int]] = None,
    ) -> Tuple[Optional[Any], int]:
        now = time.monotonic()
        for key in list(self.entries):
            if self.entries[key].expires <= now:
                del self.entries[key]
        missing = [
            end for end in checkpoints or [] if tuple(tokens[:end]) not in self._known
        ]
        limit = min(missing) if missing else len(tokens)
        matches = [
            key
            for key in self.entries
            if len(key) < len(tokens)
            and tuple(tokens[: len(key)]) == key
            and len(key) <= limit
        ]
        if not matches:
            return None, 0
        longest = max(matches, key=len)
        # Reading a later checkpoint also reads all its earlier cached prefixes.
        for key in matches:
            entry = self.entries[key]
            entry.ttl = (lifetimes or {}).get(len(key), entry.ttl)
            # A shorter new lease must not revoke an existing longer lease.
            entry.expires = max(entry.expires, now + entry.ttl)
            self.entries.move_to_end(key)
        return copy.deepcopy(self.entries[longest].cache), len(longest)

    def store(self, tokens: List[int], cache: Any, ttl: int) -> bool:
        def nbytes(value):
            if isinstance(value, (list, tuple)):
                return sum(nbytes(item) for item in value)
            return getattr(value, "nbytes", 0)

        key = tuple(tokens)
        self._known[key] = None
        self._known.move_to_end(key)
        while len(self._known) > max(4, self.max_size * 4):
            self._known.popitem(last=False)
        size = sum(nbytes(entry.state) for entry in cache)
        self.entries[key] = Checkpoint(
            copy.deepcopy(cache), ttl, time.monotonic() + ttl, size
        )
        self.entries.move_to_end(key)
        while self.entries and (
            len(self.entries) > self.max_size
            or (
                self.max_bytes is not None
                and sum(entry.nbytes for entry in self.entries.values())
                > self.max_bytes
            )
        ):
            self.entries.popitem(last=False)
        return key in self.entries
