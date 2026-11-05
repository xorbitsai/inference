# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Bounded agent lifecycle hints and soft eviction ordering for free KV blocks."""

import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable

CACHE_SESSION_ARG = "_xinference_cache_session"
CACHE_SESSION_TTL = 300.0
MAX_CACHE_SESSIONS = 4096
MAX_CACHE_SESSIONS_PER_SCOPE = 1024


@dataclass(frozen=True)
class CacheSessionTag:
    """Credential-scoped metadata created at the authenticated API boundary."""

    id: str
    revision: int
    phase: str = "active"
    scope: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def validate_cache_session(value: Any, *, control: bool = False) -> dict:
    if not isinstance(value, dict):
        raise ValueError("cache_session must be an object")
    session_id = value.get("id")
    revision = value.get("revision")
    if (
        not isinstance(session_id, str)
        or not session_id.strip()
        or len(session_id) > 128
    ):
        raise ValueError(
            "cache_session.id must be a non-blank string of at most 128 characters"
        )
    if type(revision) is not int or not 0 <= revision < 2**63:
        raise ValueError(
            "cache_session.revision must be a non-negative integer below 2**63"
        )
    phase = value.get("phase") if control else "active"
    if phase not in ("active", "waiting", "finished"):
        raise ValueError("cache_session.phase must be active, waiting, or finished")
    return {"id": session_id, "revision": revision, "phase": phase}


class CacheSessionRegistry:
    """Expire crashed clients' hints and ignore delayed older transitions."""

    def __init__(
        self,
        capacity: int = MAX_CACHE_SESSIONS,
        ttl: float = CACHE_SESSION_TTL,
        clock: Callable[[], float] = time.monotonic,
        scope_capacity: int = MAX_CACHE_SESSIONS_PER_SCOPE,
    ):
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self.states: OrderedDict[str, tuple[int, str, float]] = OrderedDict()
        self.scope_capacity = min(scope_capacity, capacity)
        self.scopes: Dict[str, OrderedDict[str, None]] = {}
        self.session_scopes: Dict[str, str] = {}

    def _remove(self, session_id: str) -> None:
        self.states.pop(session_id)
        scope = self.session_scopes.pop(session_id)
        sessions = self.scopes[scope]
        sessions.pop(session_id)
        if not sessions:
            del self.scopes[scope]

    def update(self, session: dict) -> bool:
        now = self.clock()
        while self.states and next(iter(self.states.values()))[2] <= now:
            self._remove(next(iter(self.states)))
        session_id, revision, phase = (
            session["id"],
            session["revision"],
            session["phase"],
        )
        previous = self.states.get(session_id)
        if previous is not None:
            if (
                revision < previous[0]
                or revision == previous[0]
                and phase != previous[1]
            ):
                return False
        self.states[session_id] = revision, phase, now + self.ttl
        self.states.move_to_end(session_id)
        scope = session.get("scope", "")
        sessions = self.scopes.setdefault(scope, OrderedDict())
        sessions[session_id] = None
        sessions.move_to_end(session_id)
        self.session_scopes[session_id] = scope
        if len(sessions) > self.scope_capacity:
            self._remove(next(iter(sessions)))
        if len(self.states) > self.capacity:
            self._remove(next(iter(self.states)))
        return True

    def get(self, session_ids: Iterable[str]) -> Dict[str, str]:
        now = self.clock()
        return {
            session_id: state[1]
            for session_id in session_ids
            if (state := self.states.get(session_id)) is not None and state[2] > now
        }


def order_free_session_blocks(
    pool: Any, offers: Iterable[Any], phases: Dict[str, str]
) -> int:
    """Move waiting chains to the tail, ordered so their heads survive longest.

    All blocks remain free. Active requests and in-progress transfers retain
    their existing ownership and reuse fences. Active, finished and expired
    hints leave blocks in their ordinary engine eviction order.
    """
    positions: Dict[int, int] = {}
    for offer in offers:
        if phases.get(offer.session_id) != "waiting":
            continue
        for position, block_id in enumerate(offer.block_ids):
            positions[block_id] = max(positions.get(block_id, -1), position)
    queue = pool.free_block_queue
    moved = 0
    for block_id in sorted(positions, key=positions.__getitem__, reverse=True):
        block = pool.blocks[block_id]
        if (
            block.ref_cnt
            or block.is_null
            or block.prev_free_block is None
            or block.next_free_block is None
        ):
            continue
        queue.remove(block)
        queue.append_n([block])
        moved += 1
    return moved
