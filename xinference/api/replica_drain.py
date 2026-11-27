# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, Optional

import anyio
from starlette.types import ASGIApp, Receive, Scope, Send


@dataclass
class ModelLookup:
    model_uid: str
    replica_uid: Optional[bytes] = None
    pending: bool = True

    def selected(self, model) -> None:
        uid = getattr(model, "uid", None)
        self.replica_uid = uid if isinstance(uid, bytes) else None
        self.pending = False


@dataclass(eq=False)
class InferenceRequest:
    tracker: "ReplicaDrainRequests"
    lookups: list[ModelLookup] = field(default_factory=list)
    done: bool = False


@dataclass
class DrainWindow:
    requests: set[InferenceRequest]
    collecting: bool = True


_request: ContextVar[Optional[InferenceRequest]] = ContextVar(
    "replica_drain_request", default=None
)


def track_model_lookup(model_uid: str) -> Optional[ModelLookup]:
    request = _request.get()
    if request is None or request.done:
        return None
    lookup = ModelLookup(model_uid)
    request.lookups.append(lookup)
    window = request.tracker.windows.get(model_uid)
    if window is not None and window.collecting:
        window.requests.add(request)
    return lookup


class ReplicaDrainRequests:
    """Track requests through ASGI completion, including streaming responses."""

    def __init__(self):
        self.active: set[InferenceRequest] = set()
        self.windows: dict[str, DrainWindow] = {}
        self._relaunch_admissions = 0

    @contextmanager
    def admit_relaunch(self) -> Iterator[None]:
        if self.windows:
            raise RuntimeError("A model replica drain is in progress")
        self._relaunch_admissions += 1
        try:
            # Keep the reservation through the host's reply, including when
            # the HTTP request is cancelled by its cancel scope.
            with anyio.CancelScope(shield=True):
                yield
        finally:
            self._relaunch_admissions -= 1

    def start(self, model_uid: str) -> DrainWindow:
        if self._relaunch_admissions:
            raise ValueError("An API relaunch is already in progress")
        if model_uid in self.windows:
            raise ValueError("A model replica drain is already in progress")
        window = DrainWindow(
            {
                request
                for request in self.active
                if any(lookup.model_uid == model_uid for lookup in request.lookups)
            }
        )
        self.windows[model_uid] = window
        return window

    def stop(self, model_uid: str) -> None:
        self.windows.pop(model_uid, None)

    @staticmethod
    def drained(window: DrainWindow, model_uid: str, replica_uid: bytes) -> bool:
        # A finished request sends nothing more to its replica, whatever its
        # HTTP outcome. Work it already started (including an aborted stream)
        # is counted by the replica's routing load, which the Supervisor
        # checks before confirming the drain.
        for request in window.requests:
            if request.done:
                continue
            for lookup in request.lookups:
                if lookup.model_uid != model_uid:
                    continue
                if lookup.pending or lookup.replica_uid == replica_uid:
                    return False
                if lookup.replica_uid is None:
                    raise RuntimeError(
                        "An admitted request could not be attributed to a replica"
                    )
        return True


class ReplicaDrainMiddleware:
    def __init__(self, app: ASGIApp, tracker: ReplicaDrainRequests):
        self.app = app
        self.tracker = tracker

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        request = InferenceRequest(self.tracker)
        self.tracker.active.add(request)
        token = _request.set(request)
        try:
            await self.app(scope, receive, send)
        finally:
            request.done = True
            self.tracker.active.discard(request)
            _request.reset(token)
