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
import json
import logging
import os
import sys
import threading
import time
import uuid
from contextvars import ContextVar, Token
from datetime import datetime
from http import HTTPStatus
from typing import Any, AsyncIterator, Callable, Dict, Optional, Tuple

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from starlette.datastructures import FormData, UploadFile
from starlette.responses import Response, StreamingResponse

from ..constants import (
    XINFERENCE_LOG_DIR,
    XINFERENCE_MODEL_REQUEST_LOG_BACKUP_COUNT,
    XINFERENCE_MODEL_REQUEST_LOG_BODY_MAX_BYTES,
    XINFERENCE_MODEL_REQUEST_LOG_ENABLED,
    XINFERENCE_MODEL_REQUEST_LOG_FILE,
    XINFERENCE_MODEL_REQUEST_LOG_MAX_BYTES,
    XINFERENCE_MODEL_REQUEST_LOG_RETENTION_DAYS,
)
from ..deploy.utils import SafeTimedAndSizeRotatingFileHandler
from .streaming_outcome import (
    FailureOrigin,
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
        request_id = f"xinf-{uuid.uuid4()}"
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
    _MODEL_REQUEST_ID_CONTEXT.reset(token)


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


def _decode_body(request: Request, body: bytes) -> str:
    charset = "utf-8"
    content_type = request.headers.get("content-type", "")
    for item in content_type.split(";")[1:]:
        key, separator, value = item.strip().partition("=")
        if separator and key.lower() == "charset" and value:
            charset = value.strip('"')
            break
    try:
        return body.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return body.decode("utf-8", errors="replace")


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
            body = await request.body()
            return "request_body_raw", _decode_body(request, body)

    if media_type in ("multipart/form-data", "application/x-www-form-urlencoded"):
        return "request_body", _form_to_log_body(await request.form())

    return "request_body_omitted", {"reason": "unsupported_content_type"}


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
    return {
        "timestamp": _now_iso(),
        "level": level,
        "event": event,
        "request_id": request_id,
        "endpoint": endpoint,
        "model_uid": model_uid,
        "model_type": model_type,
    }


def _error_text(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Request failed"


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
                elif not isinstance(status_code, int):
                    status_code = 500

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
            if isinstance(response, StreamingResponse):
                if XINFERENCE_MODEL_REQUEST_LOG_ENABLED:
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
                body_field: body,
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
        context_token = _set_current_model_request_id(request_id)
        reporter = get_stream_outcome_reporter(request)
        outward_error: Optional[BaseException] = None
        try:
            async for item in iterator:
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
            reporter.failed(exc, FailureOrigin.SERVER)
            raise
        else:
            reporter.completed()
        finally:
            outcome = reporter.outcome
            error = None
            if outcome.state is not StreamState.COMPLETED:
                error = {
                    "type": outcome.error_type
                    or (
                        type(outward_error).__name__
                        if outward_error
                        else outcome.state.value
                    ),
                    "message": outcome.error_message
                    or (str(outward_error) if outward_error else outcome.state.value),
                }
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
