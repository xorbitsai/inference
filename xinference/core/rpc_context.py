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
"""Explicit correlation metadata propagation across xoscar actor calls.

The metadata transported here is observability-only.  In particular it never
replaces a model operation's ``request_id``, which may participate in request
cancellation, progress tracking, batching, or backend abort semantics.
"""

import inspect
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any, AsyncIterator, Callable, Dict, Optional, Tuple, TypeVar

RPC_METADATA_KEY = "__xinf_rpc_metadata__"
RPC_METADATA_VERSION = 1
_MAX_METADATA_VALUE_LENGTH = 256

_F = TypeVar("_F", bound=Callable[..., Any])


def _valid_value(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_METADATA_VALUE_LENGTH
        and all(ord(char) >= 32 and ord(char) != 127 for char in value)
    )


@dataclass(frozen=True)
class RpcMetadata:
    """Observability metadata for one actor call."""

    correlation_id: Optional[str] = None
    operation_request_id: Optional[str] = None
    actor_call_id: Optional[str] = None
    parent_call_id: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Any) -> Optional["RpcMetadata"]:
        if (
            not isinstance(payload, dict)
            or payload.get("version") != RPC_METADATA_VERSION
        ):
            return None
        values: Dict[str, Optional[str]] = {}
        for name in (
            "correlation_id",
            "operation_request_id",
            "actor_call_id",
            "parent_call_id",
        ):
            value = payload.get(name)
            if value is not None and not _valid_value(value):
                return None
            values[name] = value
        if not any(values.values()):
            return None
        return cls(**values)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"version": RPC_METADATA_VERSION}
        for name in (
            "correlation_id",
            "operation_request_id",
            "actor_call_id",
            "parent_call_id",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


_RPC_METADATA_CONTEXT: ContextVar[Optional[RpcMetadata]] = ContextVar(
    "xinference_rpc_metadata", default=None
)


def get_current_rpc_metadata() -> Optional[RpcMetadata]:
    return _RPC_METADATA_CONTEXT.get()


def _set_rpc_metadata(metadata: Optional[RpcMetadata]) -> Token[Optional[RpcMetadata]]:
    return _RPC_METADATA_CONTEXT.set(metadata)


def _reset_rpc_metadata(token: Token[Optional[RpcMetadata]]) -> None:
    _RPC_METADATA_CONTEXT.reset(token)


def _string_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value)
    return value if _valid_value(value) else None


def _operation_request_id(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    value = kwargs.get("request_id")
    if value is not None:
        return value
    for arg in reversed(args):
        if isinstance(arg, dict) and arg.get("request_id") is not None:
            return arg["request_id"]
    return None


def pop_rpc_metadata(
    kwargs: Dict[str, Any], operation_request_id: Any = None
) -> Tuple[Optional[RpcMetadata], Token[Optional[RpcMetadata]]]:
    """Consume inbound metadata and bind it to the current actor task.

    The reserved envelope is always removed, including when malformed, so it
    can never leak into an actor implementation or a third-party model backend.
    """

    payload = kwargs.pop(RPC_METADATA_KEY, None)
    metadata = RpcMetadata.from_payload(payload)
    if metadata is None:
        metadata = get_current_rpc_metadata()

    operation_id = _string_id(operation_request_id)
    if metadata is None and operation_id is not None:
        metadata = RpcMetadata(operation_request_id=operation_id)
    elif metadata is not None and operation_id is not None:
        metadata = replace(metadata, operation_request_id=operation_id)
    return metadata, _set_rpc_metadata(metadata)


def rpc_context(func: _F) -> _F:
    """Decorator for actor methods which are not already decorated by log_async."""

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
            metadata, token = pop_rpc_metadata(kwargs, kwargs.get("request_id"))
            try:
                result = await func(*args, **kwargs)
                return wrap_async_iterator(result, metadata)
            finally:
                _reset_rpc_metadata(token)

        return async_wrapped  # type: ignore[return-value]

    @wraps(func)
    def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
        _, token = pop_rpc_metadata(kwargs, kwargs.get("request_id"))
        try:
            return func(*args, **kwargs)
        finally:
            _reset_rpc_metadata(token)

    return sync_wrapped  # type: ignore[return-value]


def build_rpc_metadata(
    *,
    correlation_id: Optional[str] = None,
    operation_request_id: Any = None,
    parent_metadata: Optional[RpcMetadata] = None,
) -> Optional[RpcMetadata]:
    """Build metadata for a child actor call from the current actor context."""

    parent = parent_metadata or get_current_rpc_metadata()
    correlation_id = _string_id(correlation_id) or (
        parent.correlation_id if parent is not None else None
    )
    operation_id = _string_id(operation_request_id) or (
        parent.operation_request_id if parent is not None else None
    )
    if correlation_id is None and operation_id is None:
        return None
    return RpcMetadata(
        correlation_id=correlation_id,
        operation_request_id=operation_id,
        actor_call_id=uuid.uuid4().hex,
        parent_call_id=parent.actor_call_id if parent is not None else None,
    )


def actor_call(
    actor_ref: Any,
    method_name: str,
    *args: Any,
    _rpc_correlation_id: Optional[str] = None,
    _rpc_operation_request_id: Any = None,
    _rpc_parent_metadata: Optional[RpcMetadata] = None,
    **kwargs: Any,
) -> Any:
    """Invoke an actor method with a versioned, internal metadata envelope.

    The helper options use reserved names so ordinary model kwargs such as
    ``correlation_id`` remain untouched and continue to reach the backend.
    """

    if _rpc_operation_request_id is None:
        _rpc_operation_request_id = _operation_request_id(args, kwargs)
    metadata = build_rpc_metadata(
        correlation_id=_rpc_correlation_id,
        operation_request_id=_rpc_operation_request_id,
        parent_metadata=_rpc_parent_metadata,
    )
    if metadata is not None:
        kwargs[RPC_METADATA_KEY] = metadata.to_payload()
    return getattr(actor_ref, method_name)(*args, **kwargs)


def wrap_async_iterator(value: Any, metadata: Optional[RpcMetadata]) -> Any:
    """Keep RPC context active while a deferred async iterator is consumed."""

    if metadata is None or not hasattr(value, "__aiter__"):
        return value

    async def iterate() -> AsyncIterator[Any]:
        iterator = value.__aiter__()
        try:
            while True:
                token = _set_rpc_metadata(metadata)
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    return
                finally:
                    _reset_rpc_metadata(token)
                yield item
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                token = _set_rpc_metadata(metadata)
                try:
                    await close()
                finally:
                    _reset_rpc_metadata(token)

    return iterate()


_CORRELATED_MODEL_METHODS = frozenset(
    {
        "generate",
        "chat",
        "create_embedding",
        "create_audio_embedding",
        "convert_ids_to_tokens",
        "rerank",
        "transcriptions",
        "translations",
        "speech",
        "text_to_image",
        "txt2img",
        "image_to_image",
        "img2img",
        "inpainting",
        "ocr",
        "infer",
        "text_to_video",
        "image_to_video",
        "flf_to_video",
        "world_generate",
        "abort_request",
        "controlnet_module_list",
        "controlnet_control_types",
        "controlnet_detect",
        "controlnet_model_list",
    }
)


def _model_operation_request_id(
    method_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Any:
    operation_request_id = _operation_request_id(args, kwargs)
    if operation_request_id is not None:
        return operation_request_id
    if method_name == "abort_request" and args:
        return args[0]
    return None


class CorrelatedModelRef:
    """API-side proxy that adds correlation metadata only to inference calls."""

    def __init__(self, actor_ref: Any, correlation_id: str):
        self._actor_ref = actor_ref
        self._correlation_id = correlation_id

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._actor_ref, name)
        if name not in _CORRELATED_MODEL_METHODS or not callable(attribute):
            return attribute

        def call(*args: Any, **kwargs: Any) -> Any:
            return actor_call(
                self._actor_ref,
                name,
                *args,
                _rpc_correlation_id=self._correlation_id,
                _rpc_operation_request_id=_model_operation_request_id(
                    name, args, kwargs
                ),
                **kwargs,
            )

        return call


def correlate_model_ref(actor_ref: Any, correlation_id: Optional[str]) -> Any:
    if correlation_id is None:
        return actor_ref
    return CorrelatedModelRef(actor_ref, correlation_id)
