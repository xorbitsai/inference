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

import asyncio
import logging
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class RequestDeadlineMixin:
    """Close connections whose HTTP request is not fully received in time.

    Mitigates Slow HTTP DoS attacks (Slowloris, CVE-2007-6750): uvicorn's
    ``timeout_keep_alive`` only covers idle time *between* requests and is
    reset on every received byte, so a client trickling one byte at a time
    can hold a connection open indefinitely. This mixin enforces an absolute
    deadline of ``request_timeout`` seconds for receiving a complete request
    (headers and body).

    The deadline only covers request receipt — it is disarmed as soon as the
    request is fully received, so long-running or streaming responses (e.g.
    chat completions) and websocket connections are never affected.
    """

    request_timeout: float = 120.0  # overridden by the factory below

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._deadline_handle: Optional[asyncio.TimerHandle] = None
        # Snapshot of self.cycle at arm time, used to distinguish a stale
        # completed cycle from the request currently being received.
        self._deadline_cycle: Any = None
        self._deadline_mid_request: bool = False

    def connection_made(self, transport: Any) -> None:
        super().connection_made(transport)  # type: ignore[misc]
        self._arm_deadline()

    def data_received(self, data: bytes) -> None:
        if self._deadline_handle is None:
            # First bytes of a new request on a keep-alive connection, or
            # remaining body bytes of a pipelined request.
            self._arm_deadline()
        super().data_received(data)  # type: ignore[misc]
        self._maybe_disarm()

    def connection_lost(self, exc: Optional[Exception]) -> None:
        self._disarm_deadline()
        super().connection_lost(exc)  # type: ignore[misc]

    def handle_websocket_upgrade(self, *args: Any) -> None:
        # The transport is handed over to the websocket protocol class and
        # this protocol receives no further callbacks, so a live timer here
        # would later kill the websocket connection. Signature differs
        # between h11 (event arg) and httptools (no args), hence *args.
        self._disarm_deadline()
        return super().handle_websocket_upgrade(*args)  # type: ignore[misc]

    def _arm_deadline(self) -> None:
        if self.request_timeout <= 0 or self._deadline_handle is not None:
            return
        cycle = self.cycle  # type: ignore[attr-defined]
        self._deadline_cycle = cycle
        # True when arming while a request body is already in flight
        # (pipelined request resumed after the previous response finished).
        self._deadline_mid_request = cycle is not None and cycle.more_body
        self._deadline_handle = self.loop.call_later(  # type: ignore[attr-defined]
            self.request_timeout, self._deadline_expired
        )

    def _disarm_deadline(self) -> None:
        if self._deadline_handle is not None:
            self._deadline_handle.cancel()
            self._deadline_handle = None
        self._deadline_cycle = None
        self._deadline_mid_request = False

    def _maybe_disarm(self) -> None:
        if self._deadline_handle is None:
            return
        cycle = self.cycle  # type: ignore[attr-defined]
        if cycle is None or cycle.more_body:
            return  # still receiving (or headers not parsed yet)
        # cycle.more_body is False, which means "request complete" only if
        # it is a new cycle since arming, or we armed mid-request. A stale
        # completed cycle (next request's partial headers trickling in on a
        # keep-alive connection) must NOT disarm the deadline.
        if self._deadline_mid_request or cycle is not self._deadline_cycle:
            self._disarm_deadline()

    def _deadline_expired(self) -> None:
        self._deadline_handle = None
        transport = self.transport  # type: ignore[attr-defined]
        if transport is None or transport.is_closing():
            return
        client = self.client  # type: ignore[attr-defined]
        prefix = "%s:%d - " % client if client else ""
        self.logger.warning(  # type: ignore[attr-defined]
            "%sClosing connection: HTTP request not received within %.0f "
            "seconds (slow-request protection).",
            prefix,
            self.request_timeout,
        )
        transport.close()


def create_hardened_http_protocol(request_timeout: float) -> Union[type, str]:
    """Return a Slowloris-hardened uvicorn HTTP protocol class.

    Returns the string ``"auto"`` (uvicorn's default protocol selection) when
    ``request_timeout`` is non-positive or the installed uvicorn does not
    expose the expected protocol internals.
    """
    if request_timeout <= 0:
        return "auto"
    try:
        # Mirror uvicorn's "auto" selection: httptools if available, else h11.
        try:
            import httptools  # noqa: F401
            from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol as _Base
        except ImportError:
            from uvicorn.protocols.http.h11_impl import (
                H11Protocol as _Base,  # type: ignore[assignment]
            )

        # Guard against future uvicorn refactors of internal attributes.
        for attr in (
            "connection_made",
            "data_received",
            "connection_lost",
            "handle_websocket_upgrade",
        ):
            if not hasattr(_Base, attr):
                raise AttributeError(attr)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "Cannot enable slow-request protection with the installed "
            "uvicorn (%s); falling back to the default HTTP protocol.",
            e,
        )
        return "auto"

    class HardenedHTTPProtocol(RequestDeadlineMixin, _Base):  # type: ignore[valid-type, misc]
        pass

    HardenedHTTPProtocol.request_timeout = float(request_timeout)
    return HardenedHTTPProtocol
