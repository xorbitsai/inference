# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Bounded agent lifecycle hints and soft eviction ordering for free KV blocks."""

import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, Optional

CACHE_SESSION_ARG = "_xinference_cache_session"
CACHE_SESSION_TTL = 300.0
MAX_CACHE_SESSIONS = 4096


@dataclass(frozen=True)
class CacheSessionTag:
    """Credential-scoped metadata created at the authenticated API boundary."""

    id: str
    revision: int
    phase: str = "active"

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
    ):
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self.states: OrderedDict[str, tuple[int, str, float]] = OrderedDict()

    def update(self, session: dict) -> bool:
        now = self.clock()
        while self.states and next(iter(self.states.values()))[2] <= now:
            self.states.popitem(last=False)
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
        if len(self.states) > self.capacity:
            self.states.popitem(last=False)
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
) -> tuple[int, int]:
    """Move waiting blocks to the tail and exclusively finished blocks to the head.

    All blocks remain free. Active requests and in-progress transfers retain
    their existing ownership and reuse fences; shared live prefixes win over
    finished owners. Missing/expired hints follow ordinary engine eviction.
    """
    owners: Dict[int, set[Optional[str]]] = {}
    for offer in offers:
        phase = phases.get(offer.session_id)
        for block_id in offer.block_ids:
            owners.setdefault(block_id, set()).add(phase)
    waiting = []
    finished = []
    for block_id, states in owners.items():
        block = pool.blocks[block_id]
        if block.ref_cnt or block.is_null:
            continue
        if "waiting" in states:
            waiting.append(block)
        elif states == {"finished"}:
            finished.append(block)
    queue = pool.free_block_queue
    for block in (*finished, *waiting):
        queue.remove(block)
    queue.prepend_n(finished)
    queue.append_n(waiting)
    return len(waiting), len(finished)
