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
"""Helpers for surfacing the real root cause of a failure to the user.

A model launch failure crosses several actor boundaries before it reaches the
REST layer. xoscar preserves the original exception type, message and
traceback across those boundaries, but two things still obscure the cause:
the message is prefixed with ``[address=..., pid=...]``, and the exception the
caller sees is often a shallow wrapper whose real cause sits further down the
``__cause__``/``__context__`` chain. These helpers dig the root cause out and
render it for both status records and HTTP responses.
"""
import os
import re
import traceback
from typing import List, Optional, Sequence, Tuple

from ..constants import XINFERENCE_ENV_DISABLE_ERROR_TRACEBACK

# xoscar's `_AsCauseBase.__str__` prefixes every remote exception message with
# the address and pid of the actor pool it came from. Useful in logs, noise in
# a UI error banner.
_ACTOR_PREFIX_RE = re.compile(r"^\[address=[^\]]*,\s*pid=-?\d+\]\s*")

# A summary lands in a status record and an HTTP `detail`; a traceback lands in
# a collapsible panel. Both are capped so a pathological error cannot bloat the
# response or the actor's in-memory status table.
MAX_SUMMARY_LENGTH = 2000
MAX_TRACEBACK_LENGTH = 20000

_TRUNCATION_MARKER = "... [truncated]"

# Guards against a malformed exception chain (a cycle, or one deep enough that
# walking it is itself a problem).
_MAX_CAUSE_DEPTH = 20


class AggregatedLaunchError(RuntimeError):
    """Several replicas of one model failed, for possibly different reasons.

    Its message is already a rendered summary of every failure, so
    ``format_error_summary`` reports it as-is instead of unwrapping to the
    first replica's cause (which would hide the others). The first failure is
    still attached as ``__cause__`` so the traceback stays reachable.
    """


def strip_actor_prefix(message: str) -> str:
    """Remove xoscar's ``[address=..., pid=...]`` prefix from a message."""
    return _ACTOR_PREFIX_RE.sub("", message)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + _TRUNCATION_MARKER


def root_cause(exc: BaseException) -> BaseException:
    """Walk to the deepest exception in ``exc``'s cause/context chain.

    ``__cause__`` (explicit ``raise ... from ...``) wins over ``__context__``
    (implicit chaining), and ``__context__`` is ignored when the raiser
    suppressed it with ``from None``.
    """
    current = exc
    seen = {id(current)}
    for _ in range(_MAX_CAUSE_DEPTH):
        nxt = current.__cause__
        if nxt is None and not current.__suppress_context__:
            nxt = current.__context__
        if nxt is None or id(nxt) in seen:
            break
        seen.add(id(nxt))
        current = nxt
    return current


def format_error_summary(exc: BaseException) -> str:
    """Render ``exc``'s root cause as a single ``"<Type>: <message>"`` line.

    Note that for an exception that crossed an actor boundary, xoscar
    synthesizes a subclass of the original type whose ``__name__`` is inherited
    from it, so the rendered type name is the real one (``RuntimeError``, not a
    mangled wrapper name).
    """
    # An aggregate already summarises every underlying failure; unwrapping it
    # would report just one of them, and its message already names each
    # replica's exception type, so no type prefix is added.
    if isinstance(exc, AggregatedLaunchError):
        return _truncate(strip_actor_prefix(str(exc)).strip(), MAX_SUMMARY_LENGTH)

    root = root_cause(exc)
    message = strip_actor_prefix(str(root)).strip()
    if not message:
        # Some exceptions carry no message at all (a bare `RuntimeError()`, or
        # a wrapper whose args were dropped). Falling back to the outer
        # exception is more informative than emitting a lone type name.
        message = strip_actor_prefix(str(exc)).strip()
        if not message:
            return type(root).__name__
    return _truncate(f"{type(root).__name__}: {message}", MAX_SUMMARY_LENGTH)


def format_error_traceback(exc: BaseException) -> Optional[str]:
    """Render the full traceback of ``exc``, including its cause chain.

    Returns ``None`` when ``XINFERENCE_DISABLE_ERROR_TRACEBACK`` is set, for
    deployments that would rather not expose filesystem paths over HTTP.

    Formats ``exc`` rather than its root cause on purpose: the outer exception
    carries the frames from every layer, and xoscar attaches the remote
    traceback to it, so this is what shows where the failure actually happened.
    """
    # Read at call time rather than freezing a module-level constant, matching
    # `constants.is_metrics_disabled`: workers are often forked and inherit an
    # already-imported copy of this module.
    if bool(int(os.environ.get(XINFERENCE_ENV_DISABLE_ERROR_TRACEBACK, 0))):
        return None
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return _truncate(text, MAX_TRACEBACK_LENGTH)


def format_replica_errors(
    failures: Sequence[Tuple[str, BaseException]],
) -> str:
    """Render one summary line per failed replica.

    Used when several replicas of the same model fail for different reasons and
    reporting only the first would point at the wrong cause.
    """
    lines: List[str] = [
        f"replica {replica_uid}: {format_error_summary(exc)}"
        for replica_uid, exc in failures
    ]
    return _truncate("\n".join(lines), MAX_SUMMARY_LENGTH)
