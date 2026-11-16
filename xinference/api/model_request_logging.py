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
"""Opt-in JSON-lines logging for model inference HTTP requests."""

import asyncio
import inspect
import json
import logging
import os
import re
import sys
import threading
import time
import typing
import uuid
from contextvars import ContextVar, Token
from datetime import datetime
from functools import lru_cache
from http import HTTPStatus
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

import anyio
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from sse_starlette.sse import EventSourceResponse
from starlette.datastructures import FormData, UploadFile
from starlette.responses import Response, StreamingResponse

from ..cache_isolation import redact_cache_salt
from ..constants import (
    XINFERENCE_LOG_DIR,
    XINFERENCE_MODEL_REQUEST_LOG_BACKUP_COUNT,
    XINFERENCE_MODEL_REQUEST_LOG_BODY_MAX_BYTES,
    XINFERENCE_MODEL_REQUEST_LOG_ENABLED,
    XINFERENCE_MODEL_REQUEST_LOG_FILE,
    XINFERENCE_MODEL_REQUEST_LOG_MAX_BYTES,
    XINFERENCE_MODEL_REQUEST_LOG_RETENTION_DAYS,
)
from ..deploy.utils import SafeTimedAndSizeRotatingFileHandler, get_process_log_identity
from .streaming_outcome import (
    FailureOrigin,
    StreamingOutcomeReporter,
    StreamOutcome,
    StreamState,
    get_stream_outcome_reporter,
)

logger = logging.getLogger(__name__)
_MODEL_REQUEST_LOGGER_NAME = "xinference.model_request"
_LOGGER_LOCK = threading.Lock()
_LOGGER_CONFIGURED = False
_MAX_REQUEST_ID_LENGTH = 256
_BODY_LOG_EXCLUDED_STATUS_CODES = frozenset({401, 403, 413})
_MODEL_REQUEST_ID_CONTEXT: ContextVar[Optional[str]] = ContextVar(
    "xinference_model_request_id", default=None
)


class _ModelRequestRotatingFileHandler(SafeTimedAndSizeRotatingFileHandler):
    """Keep the active request log private after every reopen or rollover."""

    def _open(self):
        stream = super()._open()
        try:
            os.chmod(self.baseFilename, 0o600)
        except Exception:
            stream.close()
            raise
        return stream

    def handleError(self, record: logging.LogRecord) -> None:
        # Never let logging's default stderr fallback print a request body.
        error = sys.exc_info()[1]
        if error is not None:
            raise error
        raise RuntimeError("Unknown model request log handler failure")


# Management and authentication POST endpoints must not be logged merely because
# they share the same router.
_MODEL_INFERENCE_PATHS = frozenset(
    {
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/messages",
        "/anthropic/v1/messages",
        "/v1/responses",
        "/v1/embeddings",
        "/v1/convert_ids_to_tokens",
        "/v1/rerank",
        "/v1/audio/embeddings",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/audio/speech",
        "/v1/images/generations",
        "/v1/images/variations",
        "/v1/images/inpainting",
        "/v1/images/ocr",
        "/v1/images/edits",
        "/v1/video/generations",
        "/v1/video/generations/image",
        "/v1/video/generations/flf",
        "/v1/worlds/generations",
        "/v1/flexible/infers",
        "/sdapi/v1/txt2img",
        "/sdapi/v1/img2img",
        "/controlnet/detect",
    }
)


def _request_route_path(request: Request) -> str:
    """Return the matched route path without an ASGI ``root_path`` prefix."""

    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    return route_path if isinstance(route_path, str) else request.url.path


def is_model_inference_request(request: Request) -> bool:
    return (
        request.method.upper() == "POST"
        and _request_route_path(request) in _MODEL_INFERENCE_PATHS
    )


def _event_source_supports_close_handler(response: EventSourceResponse) -> bool:
    try:
        return (
            "client_close_handler_callable"
            in inspect.signature(type(response).__init__).parameters
        )
    except (TypeError, ValueError):
        return False


def _observe_event_source_disconnect(
    response: EventSourceResponse, reporter: StreamingOutcomeReporter
) -> None:
    """Report SSE disconnects across supported sse-starlette versions."""

    if _event_source_supports_close_handler(response):
        previous_close_handler = response.client_close_handler_callable

        async def handle_client_disconnect(message: Dict[str, Any]) -> None:
            reporter.client_disconnected()
            if previous_close_handler is not None:
                await previous_close_handler(message)

        response.client_close_handler_callable = handle_client_disconnect
        return

    # sse-starlette 1.6.5-2.1 uses ``listen_for_disconnect`` and 2.2
    # uses ``_listen_for_disconnect``.  Wrap the listener's receive callable
    # instead of adding the newer close-handler attribute: those releases do
    # not consult that attribute.
    listener_name = next(
        (
            name
            for name in ("listen_for_disconnect", "_listen_for_disconnect")
            if hasattr(response, name)
        ),
        None,
    )
    if listener_name is None:
        logger.warning(
            "Cannot observe EventSourceResponse client disconnects with this "
            "sse-starlette version."
        )
        return

    previous_listener = getattr(response, listener_name)

    async def listen_for_disconnect(
        receive: Callable[[], Awaitable[Dict[str, Any]]],
    ) -> None:
        async def receive_and_report() -> Dict[str, Any]:
            message = await receive()
            if message.get("type") == "http.disconnect":
                reporter.client_disconnected()
            return message

        await previous_listener(receive_and_report)

    setattr(response, listener_name, listen_for_disconnect)


def _valid_request_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_REQUEST_ID_LENGTH
        and all(ord(char) >= 32 and ord(char) != 127 for char in value)
    )


def get_model_request_id(request: Request) -> str:
    """Return or establish a safe correlation ID for an inference request."""

    state = getattr(request, "state", None)
    request_id = getattr(state, "model_request_id", None)
    if _valid_request_id(request_id):
        assert isinstance(request_id, str)
        return request_id

    headers = getattr(request, "headers", {})
    request_id = headers.get("request-id") or headers.get("x-request-id")
    if not _valid_request_id(request_id):
        request_id = str(uuid.uuid4())
    assert isinstance(request_id, str)
    if state is not None:
        state.model_request_id = request_id
    return request_id


def get_current_model_request_id() -> Optional[str]:
    """Return the current HTTP request ID within the REST API process."""

    return _MODEL_REQUEST_ID_CONTEXT.get()


def _set_current_model_request_id(request_id: str) -> Token[Optional[str]]:
    return _MODEL_REQUEST_ID_CONTEXT.set(request_id)


def _reset_current_model_request_id(token: Token[Optional[str]]) -> None:
    try:
        _MODEL_REQUEST_ID_CONTEXT.reset(token)
    except ValueError:
        # Stream cleanup may run in a task other than the one that started it.
        pass


def _get_model_request_logger() -> Optional[logging.Logger]:
    """Configure the independent request logger lazily in the REST API process."""

    global _LOGGER_CONFIGURED
    if not XINFERENCE_MODEL_REQUEST_LOG_ENABLED:
        return None

    request_logger = logging.getLogger(_MODEL_REQUEST_LOGGER_NAME)
    if _LOGGER_CONFIGURED:
        return request_logger

    with _LOGGER_LOCK:
        if _LOGGER_CONFIGURED:
            return request_logger
        handler: Optional[logging.Handler] = None
        try:
            filename = XINFERENCE_MODEL_REQUEST_LOG_FILE
            if not os.path.isabs(filename):
                filename = os.path.join(XINFERENCE_LOG_DIR, filename)
            handler = _ModelRequestRotatingFileHandler(
                filename=filename,
                when="midnight",
                backupCount=XINFERENCE_MODEL_REQUEST_LOG_BACKUP_COUNT,
                maxBytes=XINFERENCE_MODEL_REQUEST_LOG_MAX_BYTES,
                retention_days=XINFERENCE_MODEL_REQUEST_LOG_RETENTION_DAYS,
                encoding="utf8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            for existing_handler in request_logger.handlers[:]:
                request_logger.removeHandler(existing_handler)
                existing_handler.close()
            request_logger.addHandler(handler)
            request_logger.setLevel(logging.INFO)
            request_logger.propagate = False
            request_logger.disabled = False
            _LOGGER_CONFIGURED = True
        except Exception:
            if handler is not None:
                request_logger.removeHandler(handler)
                handler.close()
            logger.warning(
                "Failed to initialize the model request log; model serving will continue.",
                exc_info=True,
            )
            return None
    return request_logger


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _write_event(event: Dict[str, Any]) -> None:
    request_logger = _get_model_request_logger()
    if request_logger is None:
        return
    try:
        request_logger.info(
            json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        )
    except Exception:
        # Do not copy the body or an untrusted request ID into fallback logs.
        logger.warning(
            "Failed to write model request log for event %s; model serving will continue.",
            event.get("event", "unknown"),
            exc_info=True,
        )


async def _write_event_async(event: Dict[str, Any]) -> None:
    # JSON serialization and file I/O must not block the API event loop.
    try:
        await asyncio.to_thread(_write_event, event)
    except Exception:
        # Scheduling or executor shutdown must not affect model responses.
        logger.warning(
            "Failed to schedule a model request log write; model serving will continue.",
            exc_info=True,
        )


def _append_form_value(result: Dict[str, Any], key: str, value: Any) -> None:
    if key not in result:
        result[key] = value
        return
    existing = result[key]
    if not isinstance(existing, list):
        result[key] = [existing]
    result[key].append(value)


def _form_to_log_body(form: FormData) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in form.multi_items():
        logged_value: Any
        if isinstance(value, UploadFile):
            logged_value = {"filename": value.filename}
        else:
            logged_value = value
        _append_form_value(result, key, logged_value)
    return result


def _body_size(request: Request) -> Optional[int]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            value = int(content_length)
        except ValueError:
            return None
        return value if value >= 0 else None
    cached_body = getattr(request, "_body", None)
    return len(cached_body) if isinstance(cached_body, bytes) else None


async def _capture_request_body(request: Request) -> Tuple[str, Any]:
    """Capture an already-authorized request without pre-reading large bodies."""

    body_size = _body_size(request)
    max_bytes = XINFERENCE_MODEL_REQUEST_LOG_BODY_MAX_BYTES
    if body_size is None:
        return "request_body_omitted", {"reason": "unknown_size"}
    if max_bytes >= 0 and body_size > max_bytes:
        return "request_body_omitted", {
            "reason": "size_limit",
            "size_bytes": body_size,
            "max_bytes": max_bytes,
        }

    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type == "application/json" or media_type.endswith("+json"):
        try:
            return "request_body", await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Invalid JSON cannot be safely redacted as structured data.
            return "request_body_omitted", {"reason": "invalid_json"}

    if media_type in ("multipart/form-data", "application/x-www-form-urlencoded"):
        return "request_body", _form_to_log_body(await request.form())

    return "request_body_omitted", {"reason": "unsupported_content_type"}


def _api_protocol_for_path(path: str) -> str:
    if path in ("/v1/messages", "/anthropic/v1/messages"):
        return "anthropic"
    if path in ("/v1/chat/completions", "/v1/completions", "/v1/responses"):
        return "openai"
    return "xinference"


def _model_type_for_path(path: str) -> str:
    if path.startswith("/v1/audio/"):
        return "audio"
    if path.startswith(("/v1/images/", "/sdapi/", "/controlnet/")):
        return "image"
    if path.startswith("/v1/video/"):
        return "video"
    if path.startswith("/v1/worlds/"):
        return "world"
    if path in ("/v1/embeddings", "/v1/convert_ids_to_tokens"):
        return "embedding"
    if path == "/v1/rerank":
        return "rerank"
    if path == "/v1/flexible/infers":
        return "flexible"
    return "llm"


def _request_metadata(body: Any) -> Tuple[str, Optional[bool]]:
    if not isinstance(body, dict):
        return "", None
    model_uid = body.get("model_uid") or body.get("model") or ""
    stream = body.get("stream")
    if isinstance(stream, str):
        if stream.lower() in ("true", "1", "yes"):
            stream = True
        elif stream.lower() in ("false", "0", "no"):
            stream = False
        else:
            stream = None
    elif not isinstance(stream, bool):
        stream = None
    return str(model_uid), stream


def _base_event(
    event: str,
    level: str,
    request_id: str,
    endpoint: str,
    model_uid: str,
    model_type: str,
) -> Dict[str, Any]:
    identity = get_process_log_identity()
    return {
        "timestamp": _now_iso(),
        "level": level,
        "event": event,
        "event_type": event,
        "request_id": request_id,
        "role": identity["role"],
        "address": identity["address"],
        "node": identity["node"],
        "module": _MODEL_REQUEST_LOGGER_NAME,
        "pid": os.getpid(),
        "api_protocol": _api_protocol_for_path(endpoint),
        "endpoint": endpoint,
        "model_uid": model_uid,
        "model_type": model_type,
    }


def _error_text(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Request failed"


# Always-on, content-free record of inference requests answered with a 4xx.
# Callers (for example a billing gateway) only see FastAPI's ``{"detail": ...}``
# body, so the runtime log is the only place that can say which check rejected a
# request.  The line must never carry request content: no body, message text,
# ``detail`` string, exception message, or pydantic ``input``/``msg``/``ctx``.
_REJECTION_REASON_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
# pydantic v1 error types are dotted, for example ``type_error.integer``.
_SAFE_ERROR_TYPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,63}")
_MAX_LOGGED_SCHEMA_ERRORS = 10
_UNTAGGED_REJECTION = "untagged"
_REQUEST_VALIDATION_REJECTION = "request_validation"
_REJECTION_STATE_ATTR = "_inference_rejection_reason"
_BODY_MODEL_STATE_ATTR = "_inference_body_model"
_BODY_REQUEST_ID_STATE_ATTR = "_inference_body_request_id"


# FastAPI prefixes request-validation locations with the parameter source.
_REQUEST_LOCATION_SOURCES = frozenset({"body", "query", "path", "header", "cookie"})


def _safe_error_type(value: Any) -> str:
    if isinstance(value, str) and _SAFE_ERROR_TYPE_RE.fullmatch(value):
        return value
    return "*"


def _safe_loc_item(value: Any, field_names: FrozenSet[str]) -> str:
    """Keep array indices and schema-defined names; mask everything else.

    Dictionary keys come from the client and pydantic reports them verbatim,
    so a string survives only if the schema itself defines that name.
    """

    if isinstance(value, bool):
        return "*"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value in field_names:
        return value
    return "*"


def _annotation_args(annotation: Any) -> Tuple[Any, ...]:
    if typing.get_origin(annotation) is typing.Literal:
        return ()
    return typing.get_args(annotation)


def _declared_fields(cls: type) -> List[Tuple[str, Optional[str], Any]]:
    """Return ``(name, alias, annotation)`` for a pydantic model or TypedDict."""

    model_fields = getattr(cls, "model_fields", None)  # pydantic v2
    if isinstance(model_fields, dict):
        return [
            (name, getattr(info, "alias", None), getattr(info, "annotation", None))
            for name, info in model_fields.items()
        ]
    v1_fields = getattr(cls, "__fields__", None)  # pydantic v1
    if isinstance(v1_fields, dict):
        return [
            (name, getattr(info, "alias", None), getattr(info, "outer_type_", None))
            for name, info in v1_fields.items()
        ]
    if typing.is_typeddict(cls) or hasattr(cls, "__required_keys__"):
        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            hints = dict(getattr(cls, "__annotations__", {}))
        return [(name, None, annotation) for name, annotation in hints.items()]
    return []


def _collect_field_names(annotation: Any, names: Set[str], seen: Set[int]) -> None:
    if id(annotation) in seen:
        return
    seen.add(id(annotation))
    for arg in _annotation_args(annotation):
        _collect_field_names(arg, names, seen)
    if not isinstance(annotation, type):
        return
    for name, alias, field_annotation in _declared_fields(annotation):
        names.add(name)
        if isinstance(alias, str):
            names.add(alias)
        _collect_field_names(field_annotation, names, seen)


@lru_cache(maxsize=64)
def schema_field_names(schema: Any) -> FrozenSet[str]:
    """Every field name and alias a request schema defines, at any depth."""

    names: Set[str] = set()
    try:
        _collect_field_names(schema, names, set())
    except Exception:
        logger.debug("Could not collect schema field names", exc_info=True)
    return frozenset(names)


def _route_parameter_names(request: Request) -> FrozenSet[str]:
    # FastAPI's attribute name; split so codespell does not "correct" it.
    route_deps = getattr(request.scope.get("route"), "depend" "ant", None)
    names: Set[str] = set(_REQUEST_LOCATION_SOURCES)
    for group in (
        "path_params",
        "query_params",
        "header_params",
        "cookie_params",
        "body_params",
    ):
        for param in getattr(route_deps, group, None) or ():
            for attr in ("name", "alias"):
                value = getattr(param, attr, None)
                if isinstance(value, str):
                    names.add(value)
    return frozenset(names)


def schema_error_fields(
    errors: Iterable[Any], field_names: FrozenSet[str] = frozenset()
) -> List[str]:
    """Reduce pydantic/FastAPI validation errors to ``loc:type`` strings.

    Only the location and the error type survive.  ``input``, ``msg`` and
    ``ctx`` can echo request content and are dropped.  Location strings that
    ``field_names`` does not define, such as client-chosen dictionary keys,
    become ``*``.
    """

    fields: List[str] = []
    for error in errors:
        if not isinstance(error, dict):
            continue
        loc = error.get("loc")
        loc_items = loc if isinstance(loc, (list, tuple)) else ()
        location = (
            ".".join(_safe_loc_item(item, field_names) for item in loc_items) or "*"
        )
        fields.append(f"{location}:{_safe_error_type(error.get('type'))}")
    return fields


class InferenceRequestRejected(HTTPException):
    """An inference ``HTTPException`` that names the rejecting check.

    The response is identical to ``HTTPException``; ``reason`` is a stable,
    content-free tag written to the rejection log instead of ``detail``.
    """

    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        *,
        reason: str,
        headers: Optional[Dict[str, str]] = None,
        schema_errors: Optional[List[str]] = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.reason = reason
        self.schema_errors = schema_errors


def mark_inference_rejection(request: Request, reason: str) -> None:
    """Tag a 4xx response that a handler returns instead of raising."""

    state = getattr(request, "state", None)
    if state is not None:
        setattr(state, _REJECTION_STATE_ATTR, reason)


def note_inference_request_identity(request: Request, body: Any) -> None:
    """Remember the body's model and request ID for the rejection log.

    A gateway can put its own correlation ID in the JSON ``request_id`` field;
    model replicas log that ID, so the rejection line must use it too.
    """

    state = getattr(request, "state", None)
    if state is None or not isinstance(body, dict):
        return
    for key, attr in (
        ("model", _BODY_MODEL_STATE_ATTR),
        ("request_id", _BODY_REQUEST_ID_STATE_ATTR),
    ):
        value = body.get(key)
        if _valid_request_id(value):
            setattr(state, attr, value)


def _first_bounded_identifier(*candidates: Any) -> str:
    """Return the first candidate within the request-ID bound, else ``""``."""

    for candidate in candidates:
        if _valid_request_id(candidate):
            return candidate
    return ""


def _rejection_cause(exc: BaseException) -> Optional[BaseException]:
    if exc.__cause__ is not None:
        return exc.__cause__
    return None if exc.__suppress_context__ else exc.__context__


def _log_inference_rejection(
    request: Request,
    header_request_id: str,
    endpoint: str,
    status_code: int,
    exc: Optional[BaseException] = None,
) -> None:
    try:
        state = request.state
        reason = getattr(exc, "reason", None) or getattr(
            state, _REJECTION_STATE_ATTR, None
        )
        schema_errors = getattr(exc, "schema_errors", None)
        if isinstance(exc, RequestValidationError):
            reason = _REQUEST_VALIDATION_REJECTION
            schema_errors = schema_error_fields(
                exc.errors(), _route_parameter_names(request)
            )
        if not isinstance(reason, str) or not _REJECTION_REASON_RE.fullmatch(reason):
            reason = _UNTAGGED_REJECTION
        # Apply the same bound at the sink, whichever source supplied the value:
        # _check_model_access stores the client's model string unchecked.
        request_id = _first_bounded_identifier(
            getattr(state, _BODY_REQUEST_ID_STATE_ATTR, None), header_request_id
        )
        model_uid = _first_bounded_identifier(
            getattr(state, "_audit_model_uid", None),
            getattr(state, _BODY_MODEL_STATE_ATTR, None),
        )
        fields = [
            f"request_id={json.dumps(request_id)}",
            f"model_uid={json.dumps(model_uid)}",
            f"endpoint={json.dumps(endpoint)}",
            f"status={status_code}",
            f"reason={reason}",
        ]
        if exc is not None:
            fields.append(f"exception={type(exc).__name__}")
            cause = _rejection_cause(exc)
            if cause is not None:
                fields.append(f"cause={type(cause).__name__}")
        if schema_errors:
            shown = schema_errors[:_MAX_LOGGED_SCHEMA_ERRORS]
            fields.append(f"schema_errors={','.join(shown)}")
            if len(schema_errors) > len(shown):
                fields.append(f"schema_errors_total={len(schema_errors)}")
        logger.warning("Inference request rejected: %s", " ".join(fields))
    except Exception as log_error:
        # Logging must never change the response.  Do not echo request data.
        logger.warning(
            "Failed to log an inference request rejection: %s",
            type(log_error).__name__,
        )


def _is_logged_rejection(status_code: int) -> bool:
    # 499 is the server's own record of a client disconnect, not a rejection.
    return 400 <= status_code < 500 and status_code != 499


class ModelRequestLoggingRoute(APIRoute):
    """APIRoute that logs inference requests after authentication succeeds."""

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            if not is_model_inference_request(request):
                return await original_route_handler(request)

            request_id = get_model_request_id(request)
            started_at = time.perf_counter()
            endpoint = _request_route_path(request)
            model_type = _model_type_for_path(endpoint)
            model_uid = ""
            stream: Optional[bool] = None

            context_token = _set_current_model_request_id(request_id)
            try:
                response = await original_route_handler(request)
            except BaseException as exc:
                _reset_current_model_request_id(context_token)
                status_code = getattr(exc, "status_code", 500)
                if isinstance(exc, asyncio.CancelledError):
                    status_code = 499
                elif isinstance(exc, RequestValidationError):
                    status_code = 422
                elif not isinstance(status_code, int):
                    status_code = 500
                if _is_logged_rejection(status_code):
                    _log_inference_rejection(
                        request, request_id, endpoint, status_code, exc
                    )

                # Authentication/authorization failures and rejected oversized
                # payloads must never persist the body.
                if (
                    XINFERENCE_MODEL_REQUEST_LOG_ENABLED
                    and status_code not in _BODY_LOG_EXCLUDED_STATUS_CODES
                ):
                    model_uid, stream = await self._log_started(request, request_id)
                await self._log_response(
                    request,
                    request_id,
                    endpoint,
                    model_uid,
                    model_type,
                    status_code,
                    started_at,
                    stream=bool(stream),
                    error={
                        "type": type(exc).__name__,
                        "message": str(getattr(exc, "detail", exc)),
                    },
                )
                if isinstance(exc, HTTPException):
                    headers = dict(exc.headers or {})
                    headers.setdefault("X-Request-ID", request_id)
                    exc.headers = headers
                raise

            _reset_current_model_request_id(context_token)
            response.headers.setdefault("X-Request-ID", request_id)
            if _is_logged_rejection(response.status_code):
                _log_inference_rejection(
                    request, request_id, endpoint, response.status_code
                )
            if (
                XINFERENCE_MODEL_REQUEST_LOG_ENABLED
                and response.status_code not in _BODY_LOG_EXCLUDED_STATUS_CODES
            ):
                model_uid, stream = await self._log_started(request, request_id)

            actual_model_uid = (
                getattr(request.state, "_audit_model_uid", "") or model_uid
            )
            actual_model_type = str(
                getattr(request.state, "_audit_model_type", "") or model_type
            ).lower()
            if isinstance(response, (StreamingResponse, EventSourceResponse)):
                if XINFERENCE_MODEL_REQUEST_LOG_ENABLED:
                    reporter = get_stream_outcome_reporter(request)
                    if isinstance(response, EventSourceResponse):
                        _observe_event_source_disconnect(response, reporter)
                    response.body_iterator = self._wrap_stream(
                        response.body_iterator,
                        request,
                        request_id,
                        endpoint,
                        actual_model_uid,
                        actual_model_type,
                        response.status_code,
                        started_at,
                    )
            else:
                await self._log_response(
                    request,
                    request_id,
                    endpoint,
                    actual_model_uid,
                    actual_model_type,
                    response.status_code,
                    started_at,
                    stream=False,
                )
            return response

        return custom_route_handler

    @staticmethod
    async def _log_started(
        request: Request, request_id: str
    ) -> Tuple[str, Optional[bool]]:
        body_field = "request_body_omitted"
        body: Any = {"reason": "capture_failed"}
        try:
            body_field, body = await _capture_request_body(request)
        except Exception:
            logger.warning(
                "Failed to capture a model request body; processing will continue.",
                exc_info=True,
            )
        model_uid, stream = _request_metadata(body)
        event = _base_event(
            "model_request_started",
            "INFO",
            request_id,
            _request_route_path(request),
            model_uid,
            _model_type_for_path(_request_route_path(request)),
        )
        event.update(
            {
                "method": request.method,
                "content_type": request.headers.get("content-type", ""),
                "stream": stream,
                body_field: redact_cache_salt(body),
            }
        )
        await _write_event_async(event)
        return model_uid, stream

    @staticmethod
    async def _log_response(
        request: Request,
        request_id: str,
        endpoint: str,
        model_uid: str,
        model_type: str,
        status_code: int,
        started_at: float,
        *,
        stream: bool,
        stream_completed: Optional[bool] = None,
        error: Optional[Dict[str, str]] = None,
        stream_outcome: Optional[StreamOutcome] = None,
    ) -> None:
        if not XINFERENCE_MODEL_REQUEST_LOG_ENABLED:
            return
        model_uid = getattr(request.state, "_audit_model_uid", "") or model_uid
        model_type = str(
            getattr(request.state, "_audit_model_type", "") or model_type
        ).lower()
        if (
            stream_outcome is not None
            and stream_outcome.state is not StreamState.COMPLETED
        ):
            error = error or {
                "type": stream_outcome.error_type or stream_outcome.state.value,
                "message": stream_outcome.error_message or stream_outcome.state.value,
            }
        success = status_code < 400 and error is None
        event = _base_event(
            "model_request_finished" if success else "model_request_failed",
            "INFO" if success else "WARNING",
            request_id,
            endpoint,
            model_uid,
            model_type,
        )
        event.update(
            {
                "stream": stream,
                "status_code": status_code,
                "success": success,
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 3),
            }
        )
        if stream_completed is not None:
            event["stream_completed"] = stream_completed
        if stream_outcome is not None:
            event["http_success"] = status_code < 400
            event["stream_outcome"] = stream_outcome.state.value
            if stream_outcome.failure_origin is not None:
                event["failure_origin"] = stream_outcome.failure_origin.value
        if not success:
            event["error"] = error or {
                "type": "HTTPError",
                "message": _error_text(status_code),
            }
        await _write_event_async(event)

    @classmethod
    async def _finalize_stream(
        cls,
        source: AsyncIterator[Any],
        reporter: StreamingOutcomeReporter,
        request: Request,
        request_id: str,
        endpoint: str,
        model_uid: str,
        model_type: str,
        status_code: int,
        started_at: float,
        outward_error: Optional[BaseException],
    ) -> Optional[BaseException]:
        cleanup_error: Optional[BaseException] = None
        close = getattr(source, "aclose", None)
        if close is not None:
            context_token = _set_current_model_request_id(request_id)
            try:
                await close()
            except BaseException as exc:
                cleanup_error = exc
                if isinstance(exc, asyncio.CancelledError):
                    reporter.cancelled(exc, FailureOrigin.SERVER)
                else:
                    reporter.failed(exc, FailureOrigin.SERVER)
            finally:
                _reset_current_model_request_id(context_token)

        outcome = reporter.outcome
        error = None
        if outcome.state is not StreamState.COMPLETED:
            effective_error = outward_error or cleanup_error
            error = {
                "type": outcome.error_type
                or (
                    type(effective_error).__name__
                    if effective_error is not None
                    else outcome.state.value
                ),
                "message": outcome.error_message
                or (
                    str(effective_error)
                    if effective_error is not None
                    else outcome.state.value
                ),
            }
        context_token = _set_current_model_request_id(request_id)
        try:
            await cls._log_response(
                request,
                request_id,
                endpoint,
                model_uid,
                model_type,
                status_code,
                started_at,
                stream=True,
                stream_completed=outcome.state is StreamState.COMPLETED,
                error=error,
                stream_outcome=outcome,
            )
        finally:
            _reset_current_model_request_id(context_token)
        return cleanup_error

    @classmethod
    async def _wrap_stream(
        cls,
        iterator: AsyncIterator[Any],
        request: Request,
        request_id: str,
        endpoint: str,
        model_uid: str,
        model_type: str,
        status_code: int,
        started_at: float,
    ) -> AsyncIterator[Any]:
        reporter = get_stream_outcome_reporter(request)
        outward_error: Optional[BaseException] = None
        source = iterator.__aiter__()
        try:
            while True:
                context_token = _set_current_model_request_id(request_id)
                try:
                    item = await source.__anext__()
                except StopAsyncIteration:
                    reporter.completed()
                    break
                finally:
                    _reset_current_model_request_id(context_token)
                yield item
        except asyncio.CancelledError as exc:
            outward_error = exc
            reporter.cancelled(exc)
            raise
        except GeneratorExit as exc:
            outward_error = exc
            reporter.client_disconnected(exc)
            raise
        except BaseException as exc:
            outward_error = exc
            reporter.failed(exc, FailureOrigin.MODEL_GENERATOR)
            raise
        finally:
            # EventSourceResponse cancels its AnyIO task group after a client
            # disconnect.  Finalization must finish outside that cancellation
            # scope or the async log write can be cancelled before it runs.
            with anyio.CancelScope(shield=True):
                cleanup_error = await cls._finalize_stream(
                    source,
                    reporter,
                    request,
                    request_id,
                    endpoint,
                    model_uid,
                    model_type,
                    status_code,
                    started_at,
                    outward_error,
                )

            # Preserve the exception already leaving the generator.  If cleanup
            # was the only failure, propagate it after the terminal event exists.
            if cleanup_error is not None and outward_error is None:
                raise cleanup_error
