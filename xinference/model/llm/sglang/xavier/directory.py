# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""P/D rendezvous metadata. KV payloads never pass through this actor."""

import time

import xoscar as xo

from .settings import transfer_timeout


class XavierPDDirectory(xo.StatelessActor):
    def __init__(self):
        super().__init__()
        self.namespace = None
        self.rooms = {}
        self.completed_requests = 0
        self.gpu_bytes = 0
        self.peers = {}

    def register_peer(self, rank, address):
        if rank in self.peers and self.peers[rank] != address:
            raise ValueError(
                "SGLang Xavier GPU replica restarted; relaunch the deployment"
            )
        self.peers[rank] = address

    def unregister_peer(self, rank, address=None):
        if rank not in self.peers or (
            address is not None and self.peers[rank] != address
        ):
            return False
        del self.peers[rank]
        self.rooms = {
            room: state
            for room, state in self.rooms.items()
            if rank not in state["ranks"].values()
            and (state["source"] or {}).get("rank") != rank
        }
        return True

    def configure(self, namespace):
        if self.namespace is not None and namespace != self.namespace:
            raise ValueError("SGLang Xavier GPU KV namespaces differ between replicas")
        self.namespace = namespace

    def _expire(self):
        now = time.monotonic()
        self.rooms = {
            room: state
            for room, state in self.rooms.items()
            if state["completed"] or state["deadline"] > now
        }

    def prepare(self, room, namespace, prompt_hash, role, rank=None, timeout=None):
        self._expire()
        if namespace != self.namespace or role not in ("prefill", "decode"):
            raise ValueError("Unregistered SGLang Xavier PD namespace or role")
        if type(room) is not int or not 0 < room < 2**63:
            raise ValueError("Invalid SGLang Xavier bootstrap room")
        timeout = transfer_timeout(timeout)
        state = self.rooms.get(room)
        if state is None:
            if len(self.rooms) >= 4096:
                raise RuntimeError("Too many active SGLang Xavier PD requests")
            state = self.rooms[room] = dict(
                prompt_hash=prompt_hash,
                roles=[],
                source=None,
                completed=False,
                deadline=time.monotonic() + timeout,
                ranks={},
                finished=[],
            )
        if state["prompt_hash"] != prompt_hash or role in state["roles"]:
            raise ValueError("SGLang Xavier PD prompt mismatch or duplicate role")
        state["deadline"] = max(state["deadline"], time.monotonic() + timeout)
        state["roles"].append(role)
        if rank is not None:
            state["ranks"][role] = rank

    def publish_source(self, room, source):
        state = self.rooms.get(room)
        if state is None or "prefill" not in state["roles"] or state["source"]:
            raise ValueError("Unprepared or duplicate SGLang Xavier producer")
        state["source"] = source

    def source(self, room):
        self._expire()
        state = self.rooms.get(room)
        if state is None:
            raise RuntimeError("SGLang Xavier PD handoff expired or was cancelled")
        return state["source"]

    def complete(self, room, gpu_bytes):
        state = self.rooms.get(room)
        if state is None or set(state["roles"]) != {"prefill", "decode"}:
            raise RuntimeError("SGLang Xavier PD request is unavailable")
        if not state["completed"]:
            self.completed_requests += 1
            self.gpu_bytes += gpu_bytes
        state["completed"] = True

    def check(self, room):
        self._expire()
        state = self.rooms.get(room)
        return bool(state and state["completed"])

    def release(self, room, role=None):
        state = self.rooms.get(room)
        if state is None:
            return
        if role is None:
            self.rooms.pop(room, None)
        else:
            state["finished"].append(role)
            if set(state["finished"]) == {"prefill", "decode"}:
                self.rooms.pop(room, None)

    def get_stats(self):
        self._expire()
        return dict(
            namespace=self.namespace,
            active_handoffs=len(self.rooms),
            completed_requests=self.completed_requests,
            gpu_bytes=self.gpu_bytes,
            peers=self.peers,
        )
