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
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .streaming_outcome import StreamState


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
    uncertain: bool = False


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
        for request in window.requests:
            for lookup in request.lookups:
                if lookup.model_uid != model_uid:
                    continue
                if lookup.replica_uid is None and not lookup.pending:
                    raise RuntimeError(
                        "An admitted request could not be attributed to a replica"
                    )
                if lookup.replica_uid == replica_uid:
                    if not request.done:
                        return False
                    if request.uncertain:
                        raise RuntimeError(
                            "An admitted request has no confirmed outcome"
                        )
                if lookup.pending and not request.done:
                    return False
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

        async def observe_receive() -> Message:
            message = await receive()
            if message["type"] == "http.disconnect":
                request.uncertain = True
            return message

        async def observe_send(message: Message) -> None:
            if message["type"] == "http.response.start" and message["status"] >= 500:
                request.uncertain = True
            await send(message)

        try:
            await self.app(scope, observe_receive, observe_send)
        except BaseException:
            request.uncertain = True
            raise
        finally:
            reporter = scope.get("state", {}).get("model_stream_outcome")
            if reporter is not None and reporter.outcome.state in (
                StreamState.FAILED,
                StreamState.CANCELLED,
                StreamState.CLIENT_DISCONNECTED,
            ):
                request.uncertain = True
            request.done = True
            self.tracker.active.discard(request)
            _request.reset(token)
