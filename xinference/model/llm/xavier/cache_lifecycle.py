# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Bounded agent lifecycle hints and soft eviction ordering for free KV blocks."""

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable

CACHE_SESSION_ARG = "_xinference_cache_session"
CACHE_SESSION_TTL = 300.0
MAX_CACHE_SESSIONS = 4096
MAX_CACHE_SESSIONS_PER_SCOPE = 1024
MAX_SESSION_PREFIX_KEYS = 4096

logger = logging.getLogger(__name__)


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


class SessionPrefixPolicy:
    """Engine-independent, bounded soft retention of immutable prefix keys.

    Waiting prefixes are touched tail first so their causal heads survive
    longest. Nothing is pinned: allocation, active reads and P/D leases win.
    Finished or expired sessions immediately lose this preference, without
    deleting pages another session may still use.
    """

    def __init__(self):
        self.registry = CacheSessionRegistry()
        self.prefixes: OrderedDict[str, list[str]] = OrderedDict()
        self.key_count = 0

    def bind(self, session: dict, keys: list[str]) -> bool:
        if (
            not isinstance(keys, (list, tuple))
            or len(keys) > MAX_SESSION_PREFIX_KEYS
            or any(
                not isinstance(key, str)
                or len(key) != 64
                or any(c not in "0123456789abcdef" for c in key)
                for key in keys
            )
        ):
            raise ValueError("Invalid Xavier session prefix keys")
        if not self.registry.update(session):
            return False
        previous = self.prefixes.pop(session["id"], [])
        self.key_count -= len(previous)
        # Batches can arrive independently, including during vLLM offload.
        keys = list(dict.fromkeys([*previous, *keys]))[:MAX_SESSION_PREFIX_KEYS]
        self.prefixes[session["id"]] = keys
        self.key_count += len(keys)
        while self.key_count > MAX_SESSION_PREFIX_KEYS:
            _, dropped = self.prefixes.popitem(last=False)
            self.key_count -= len(dropped)
        return True

    def waiting_keys(self) -> list[str]:
        phases = self.registry.get(self.prefixes)
        positions: Dict[str, int] = {}
        for session_id, keys in list(self.prefixes.items()):
            if session_id not in phases or phases[session_id] == "finished":
                self.key_count -= len(self.prefixes.pop(session_id))
            elif phases[session_id] == "waiting":
                for position, key in enumerate(keys):
                    positions[key] = max(position, positions.get(key, -1))
        return sorted(positions, key=positions.__getitem__, reverse=True)


async def bind_cache_session(config: dict, session: Any, keys: list[str]) -> None:
    """Best-effort metadata RPC, never part of model correctness or KV ownership."""
    if not isinstance(session, CacheSessionTag) or config.get("role"):
        return
    import xoscar as xo

    async def bind():
        ref = await xo.actor_ref(address=config["address"], uid=config["uid"])
        await ref.bind_cache_session(session.to_dict(), keys[:MAX_SESSION_PREFIX_KEYS])

    try:
        await asyncio.wait_for(bind(), timeout=0.5)
    except Exception:
        logger.debug(
            "Agent prefix binding unavailable; continuing inference", exc_info=True
        )
