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
"""Side-channel reporting for the outcome of streamed model requests.

Endpoints may encode an exception as an SSE/JSON error event and then finish
their response generator normally.  The response logging route cannot infer
that outcome from chunks without becoming protocol-aware, so endpoints report
it explicitly through request state instead.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Optional

from fastapi import Request


class StreamState(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CLIENT_DISCONNECTED = "client_disconnected"


class FailureOrigin(str, Enum):
    MODEL_GENERATOR = "model_generator"
    UPSTREAM = "upstream"
    PROTOCOL = "protocol"
    CLIENT = "client"
    SERVER = "server"


@dataclass(frozen=True)
class StreamOutcome:
    state: StreamState
    failure_origin: Optional[FailureOrigin] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def terminal(self) -> bool:
        return self.state is not StreamState.PENDING


class StreamingOutcomeReporter:
    """Idempotent outcome holder; the first terminal state wins."""

    def __init__(self) -> None:
        self._outcome = StreamOutcome(StreamState.PENDING)

    @property
    def outcome(self) -> StreamOutcome:
        return self._outcome

    def _finish(
        self,
        state: StreamState,
        *,
        failure_origin: Optional[FailureOrigin] = None,
        error: Optional[BaseException] = None,
    ) -> StreamOutcome:
        if self._outcome.terminal:
            return self._outcome
        self._outcome = StreamOutcome(
            state=state,
            failure_origin=failure_origin,
            error_type=type(error).__name__ if error is not None else None,
            error_message=(
                (str(error) or type(error).__name__) if error is not None else None
            ),
        )
        return self._outcome

    def completed(self) -> StreamOutcome:
        return self._finish(StreamState.COMPLETED)

    def failed(
        self, error: BaseException, failure_origin: FailureOrigin
    ) -> StreamOutcome:
        return self._finish(
            StreamState.FAILED, failure_origin=failure_origin, error=error
        )

    def cancelled(
        self,
        error: Optional[BaseException] = None,
        failure_origin: FailureOrigin = FailureOrigin.SERVER,
    ) -> StreamOutcome:
        return self._finish(
            StreamState.CANCELLED, failure_origin=failure_origin, error=error
        )

    def client_disconnected(
        self, error: Optional[BaseException] = None
    ) -> StreamOutcome:
        return self._finish(
            StreamState.CLIENT_DISCONNECTED,
            failure_origin=FailureOrigin.CLIENT,
            error=error,
        )


def get_stream_outcome_reporter(request: Request) -> StreamingOutcomeReporter:
    reporter = getattr(request.state, "model_stream_outcome", None)
    if reporter is None:
        reporter = StreamingOutcomeReporter()
        request.state.model_stream_outcome = reporter
    return reporter


def report_stream_failure(
    request: Request, error: BaseException, failure_origin: FailureOrigin
) -> StreamOutcome:
    return get_stream_outcome_reporter(request).failed(error, failure_origin)


def report_client_disconnect(
    request: Request, error: Optional[BaseException] = None
) -> StreamOutcome:
    return get_stream_outcome_reporter(request).client_disconnected(error)


async def observe_stream(
    iterator: AsyncIterator[Any],
    reporter: StreamingOutcomeReporter,
    *,
    failure_origin: FailureOrigin,
) -> AsyncIterator[Any]:
    """Observe source failures without parsing, buffering, or changing chunks."""

    source = iterator.__aiter__()
    try:
        async for item in source:
            yield item
    except asyncio.CancelledError:
        # The endpoint or outer response wrapper owns cancellation
        # classification because only that layer can identify a client
        # disconnect reliably.
        raise
    except GeneratorExit:
        # The outer response wrapper classifies consumer-side closure.
        raise
    except BaseException as exc:
        reporter.failed(exc, failure_origin)
        raise
    finally:
        close = getattr(source, "aclose", None)
        if close is not None:
            await close()
