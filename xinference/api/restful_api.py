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
import base64
import gc
import inspect
import ipaddress
import json
import logging
import multiprocessing
import os
import pprint
import re
import socket
import struct
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    get_type_hints,
)

import aiohttp
import anyio
import httpx
import xoscar as xo
from aioprometheus import REGISTRY, MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import PlainTextResponse, StreamingResponse
from uvicorn import Config, Server
from xoscar.utils import get_next_port

from ..constants import (
    XINFERENCE_ALLOWED_IPS,
    XINFERENCE_AUTH_DB_PATH,
    XINFERENCE_DEFAULT_CANCEL_BLOCK_DURATION,
    XINFERENCE_DEFAULT_ENDPOINT_PORT,
    XINFERENCE_ENABLE_OTEL,
    XINFERENCE_HTTP_LIMIT_CONCURRENCY,
    XINFERENCE_HTTP_REQUEST_TIMEOUT,
    XINFERENCE_HTTP_TIMEOUT_KEEP_ALIVE,
    XINFERENCE_LAUNCH_HISTORY_DB_PATH,
    XINFERENCE_MONITOR_CONFIG_DB_PATH,
    XINFERENCE_REQUIRE_CACHE_SALT,
    XINFERENCE_SSE_PING_ATTEMPTS_SECONDS,
    XINFERENCE_SYSTEM_SETTINGS_PATH,
    XINFERENCE_TOKEN_ROUTER_ENABLED,
    get_auth_encryption_key,
    get_auth_jwt_secret_key,
    is_auth_advanced,
    is_metrics_disabled,
)
from ..core.event import Event, EventCollectorActor, EventType
from ..core.exceptions import (
    InvalidAudioInputError,
    InvalidModelRequestError,
    ModelNotReadyError,
)
from ..core.http_protocol import create_hardened_http_protocol
from ..core.replica_config import ReplicaConfig
from ..core.rpc_context import actor_call, correlate_model_ref
from ..core.supervisor import SupervisorActor
from ..core.utils import CancelMixin
from ..router.constants import TOKEN_ROUTER_BACKEND_AUTHORIZATION_HEADER
from ..router.credentials import token_router_data_plane_token
from ..types import CreateChatCompletion, PeftModelConfig, max_tokens_field
from .frontend_static import mount_frontend
from .model_request_logging import ModelRequestLoggingRoute, get_model_request_id
from .pdf_ocr import (
    DEFAULT_PDF_OCR_DPI,
    PDF_MAGIC,
    WHOLE_DOCUMENT_OCR_TASKS,
    is_pdf_upload,
    merge_ocr_page_results,
    rasterize_pdf,
    validate_pdf_for_parse,
)
from .protocols import (
    AnthropicProtocolError,
    anthropic_error_response,
    anthropic_request_model,
    anthropic_stream_events,
    openai_to_anthropic,
    parse_anthropic_request,
)
from .protocols.openai_responses import (
    ResponsesProtocolError,
    ResponsesRequest,
    chat_to_response,
    parse_responses_request,
    responses_error,
    responses_error_code,
    responses_stream_events,
)
from .responses import JSONResponse
from .schemas import (
    AutoConfigLLMRequest,
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    RegisterModelRequest,
    RerankRequest,
    SDAPIControlNetDetect,
    SDAPIImg2imgRequst,
    SDAPIInterrupt,
    SDAPIOptionsRequest,
    SDAPIProgress,
    SDAPITxt2imgRequst,
    SpeechRequest,
    TextToImageRequest,
    TextToVideoRequest,
    UpdateModelRequest,
    UpdateModelSpecRequest,
    WorldGenerationRequest,
)
from .streaming_outcome import (
    FailureOrigin,
    get_stream_outcome_reporter,
    observe_stream,
    report_client_disconnect,
    report_stream_failure,
)
from .utils import get_request_route_path, require_model

logger = logging.getLogger(__name__)


def _parse_kwargs(kwargs: Optional[str]) -> Dict[str, Any]:
    if kwargs is None:
        return {}
    try:
        parsed_kwargs = json.loads(kwargs)
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=400, detail="kwargs must be a valid JSON object"
        ) from e
    if not isinstance(parsed_kwargs, dict):
        raise HTTPException(status_code=400, detail="kwargs must be a JSON object")
    return parsed_kwargs


# One base64-encoded 512 MiB media reference plus bounded JSON/config overhead.
_MAX_WORLD_REQUEST_BYTES = ((512 * 1024 * 1024 + 2) // 3) * 4 + 1024 * 1024

_TOKEN_ROUTER_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_TOKEN_ROUTER_REQUEST_HEADERS = {
    "accept",
    "authorization",
    "content-type",
    "traceparent",
    "tracestate",
    "user-agent",
    "x-request-id",
}


class _SSERequestAbortController:
    """Abort and close an interrupted model stream exactly once."""

    def __init__(
        self,
        model: Any,
        request_id: str,
        request: Request,
        kind: str,
        abort_timeout: Optional[float] = None,
    ):
        self._model = model
        self._request_id = request_id
        self._request = request
        self._kind = kind
        self._abort_timeout = abort_timeout
        self._abort_task: Optional[asyncio.Task] = None
        self._body_iterator: Optional[AsyncIterator[Any]] = None
        self._stream_task: Optional[asyncio.Task] = None

    def bind_body_iterator(self, body_iterator: AsyncIterator[Any]) -> None:
        self._body_iterator = body_iterator

    def bind_stream_task(self) -> None:
        self._stream_task = asyncio.current_task()

    async def abort(self) -> None:
        if self._abort_task is None:
            operation = self._model.abort_request(self._request_id)
            if self._abort_timeout is not None:
                operation = asyncio.wait_for(operation, timeout=self._abort_timeout)
            self._abort_task = asyncio.create_task(operation)
        try:
            await asyncio.shield(self._abort_task)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Failed to abort disconnected %s request %s",
                self._kind,
                self._request_id,
            )

    async def handle_disconnect(self, _message: Dict[str, Any]) -> None:
        with anyio.CancelScope(shield=True):
            report_client_disconnect(self._request)
            logger.info(
                "Disconnected from client (via refresh/close) %s during %s.",
                self._request.client,
                self._kind,
            )
            await self.abort()

            # Cancel and join the response's stream task before the disconnect
            # listener returns. EventSourceResponse cancels its AnyIO task
            # group after this callback, which is too late for cleanup code
            # containing cancellation checkpoints.
            stream_task = self._stream_task
            if (
                stream_task is not None
                and stream_task is not asyncio.current_task()
                and not stream_task.done()
            ):
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.debug(
                        "Disconnected %s stream task %s failed during cleanup",
                        self._kind,
                        self._request_id,
                        exc_info=True,
                    )

            # During an ASGI send the generator is suspended at ``yield``, so
            # the response task itself never receives CancelledError. Close it
            # here to run its finally block and release the remote iterator.
            close = getattr(self._body_iterator, "aclose", None)
            if close is not None:
                try:
                    await close()
                except RuntimeError:
                    # The generator may instead be running in __anext__. The
                    # task-group cancellation closes it after this returns.
                    pass
                except Exception:
                    logger.debug(
                        "Failed to close disconnected %s response iterator %s",
                        self._kind,
                        self._request_id,
                        exc_info=True,
                    )


async def _run_non_streaming_request(
    request: Request,
    model: Any,
    request_id: str,
    kind: str,
    operation: Awaitable[Any],
    *,
    abort_controller: Optional[_SSERequestAbortController] = None,
) -> Response:
    """Race a direct vLLM model call against disconnect and join owned tasks."""

    async def wait_for_disconnect() -> None:
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    generation = asyncio.ensure_future(operation)
    listener = asyncio.create_task(wait_for_disconnect())
    if abort_controller is None:
        abort_controller = _SSERequestAbortController(
            model, request_id, request, kind, abort_timeout=5
        )
    interrupted = True
    cancellation: Optional[asyncio.CancelledError] = None

    async def cleanup() -> None:
        with anyio.CancelScope(shield=True):
            try:
                if interrupted:
                    await abort_controller.abort()
            except asyncio.CancelledError:
                logger.exception("Abort cancelled for %s request %s", kind, request_id)
            finally:
                # Non-stream results retain actor-owned serve counts; callers
                # returning a stream must release it if the response is discarded.
                # Cancel/join both tasks even if abort fails or times out.
                for task in (generation, listener):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(generation, listener, return_exceptions=True)

    try:
        try:
            await asyncio.wait(
                (generation, listener), return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError as exc:
            cancellation = exc
            get_stream_outcome_reporter(request).cancelled(exc)
            raise
        # Prefer generation even when both tasks are ready. Its own
        # cancellation is a result, not cancellation of the handler's wait.
        if generation.done():
            interrupted = False
            result = generation.result()
            return (
                result
                if isinstance(result, Response)
                else Response(result, media_type="application/json")
            )
        listener.result()  # Propagate receive failures without claiming disconnect.
        report_client_disconnect(request)
        logger.info(
            "Disconnected from client (via refresh/close) %s during %s.",
            request.client,
            kind,
        )
        return Response(status_code=499)
    finally:
        cleanup_task = asyncio.create_task(cleanup())
        with anyio.CancelScope(shield=True):
            while True:
                try:
                    await asyncio.shield(cleanup_task)
                    break
                except asyncio.CancelledError as exc:
                    # Repeated handler cancellations must not abandon cleanup.
                    if cancellation is None:
                        cancellation = exc
                        get_stream_outcome_reporter(request).cancelled(exc)
                    if cleanup_task.cancelled():
                        break
        if cancellation is not None:
            raise cancellation


def _observe_sse_disconnect(
    response: EventSourceResponse,
    handler: Callable[[Dict[str, Any]], Awaitable[None]],
) -> None:
    """Attach a disconnect handler across supported sse-starlette versions."""

    if hasattr(response, "client_close_handler_callable"):
        previous_handler = response.client_close_handler_callable

        async def chained_handler(message: Dict[str, Any]) -> None:
            await handler(message)
            if previous_handler is not None:
                await previous_handler(message)

        response.client_close_handler_callable = chained_handler
        return

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
            "Cannot abort model requests on SSE disconnects with this sse-starlette version."
        )
        return

    previous_listener = getattr(response, listener_name)

    async def listen_for_disconnect(
        receive: Callable[[], Awaitable[Dict[str, Any]]],
    ) -> None:
        async def receive_and_abort() -> Dict[str, Any]:
            message = await receive()
            if message.get("type") == "http.disconnect":
                await handler(message)
            return message

        await previous_listener(receive_and_abort)

    setattr(response, listener_name, listen_for_disconnect)


async def _close_async_iterator(iterator: Any) -> None:
    from xoscar.api import IteratorWrapper

    close = getattr(iterator, "aclose", None)
    if close is None and isinstance(iterator, IteratorWrapper):
        # This await owns cleanup; __del__ would synchronously join a thread
        # on the API event loop and retry even after an explicit destroy.
        iterator._gc_destroy = False
        close = iterator.destroy
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def _cleanup_model_stream(
    *,
    abort_controller: _SSERequestAbortController,
    interrupted: bool,
    observed_iterator: Any,
    model_iterator: Any,
    model: Any,
    request_id: str,
    kind: str,
) -> None:
    """Finish all stream cleanup even under AnyIO task-group cancellation."""

    from xoscar.api import IteratorWrapper

    with anyio.CancelScope(shield=True):
        if interrupted:
            try:
                await abort_controller.abort()
            except (asyncio.CancelledError, Exception):
                logger.exception(
                    "Failed to abort interrupted %s request %s", kind, request_id
                )

        try:
            if observed_iterator is not None:
                await _close_async_iterator(observed_iterator)
        except (asyncio.CancelledError, Exception):
            logger.debug(
                "Failed to close %s stream %s",
                kind,
                request_id,
                exc_info=True,
            )
        finally:
            if model_iterator is not observed_iterator and isinstance(
                model_iterator, IteratorWrapper
            ):
                try:
                    await _close_async_iterator(model_iterator)
                except (asyncio.CancelledError, Exception):
                    logger.debug(
                        "Failed to destroy %s stream %s",
                        kind,
                        request_id,
                        exc_info=True,
                    )
            if model_iterator is not None and (
                inspect.isasyncgen(model_iterator)
                or inspect.isgenerator(model_iterator)
                or isinstance(model_iterator, IteratorWrapper)
            ):
                try:
                    await _release_model_stream(model, model_iterator)
                except (asyncio.CancelledError, Exception):
                    logger.exception(
                        "Failed to release serve count for %s request %s",
                        kind,
                        request_id,
                    )


async def _release_model_stream(model, iterator) -> None:
    """Release the slot on the actor that actually admitted this iterator."""
    from xoscar.api import IteratorWrapper

    if isinstance(iterator, IteratorWrapper):
        model = iterator._actor_ref
        if model is None:
            model = await xo.actor_ref(
                address=iterator._actor_addr, uid=iterator._actor_uid
            )
    await model.decrease_serve_count()


def _request_credential(request: Request) -> str:
    headers = {key.lower(): value for key, value in request.headers.items()}
    authorization = headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        bearer_token = authorization[7:].strip()
        if bearer_token:
            return bearer_token
    return headers.get("x-api-key", "").strip()


def _token_router_request_headers(
    request: Request,
    request_id: str,
    *,
    forward_external_credential: bool = True,
) -> Dict[str, str]:
    headers = {
        key.lower(): value
        for key, value in request.headers.items()
        if key.lower() in _TOKEN_ROUTER_REQUEST_HEADERS
        and key.lower() != "authorization"
    }
    internal_token = token_router_data_plane_token()
    external_credential = (
        _request_credential(request) if forward_external_credential else ""
    )
    if internal_token:
        headers["authorization"] = f"Bearer {internal_token}"
        if external_credential:
            headers[TOKEN_ROUTER_BACKEND_AUTHORIZATION_HEADER] = (
                f"Bearer {external_credential}"
            )
    elif external_credential:
        headers["authorization"] = f"Bearer {external_credential}"
    headers["content-type"] = "application/json"
    headers["x-request-id"] = request_id
    return headers


def _token_router_response_headers(response: httpx.Response) -> Dict[str, str]:
    excluded = _TOKEN_ROUTER_HOP_BY_HOP_HEADERS | {"content-length"}
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in excluded
    }


def _normalize_token_router_chat_payload(
    raw_body: Dict[str, Any],
) -> Dict[str, Any]:
    """Mirror Chat API normalization before forwarding to the Token Router."""
    normalized = dict(raw_body)
    enable_thinking = raw_body.get("enable_thinking")
    if enable_thinking is None:
        extra_body = raw_body.get("extra_body")
        if isinstance(extra_body, dict):
            enable_thinking = extra_body.get("enable_thinking")
    if isinstance(enable_thinking, bool):
        chat_template_kwargs = raw_body.get("chat_template_kwargs") or {}
        if isinstance(chat_template_kwargs, str):
            try:
                chat_template_kwargs = json.loads(chat_template_kwargs)
            except json.JSONDecodeError:
                chat_template_kwargs = {}
        if not isinstance(chat_template_kwargs, dict):
            chat_template_kwargs = {}
        chat_template_kwargs = dict(chat_template_kwargs)
        chat_template_kwargs["enable_thinking"] = enable_thinking
        chat_template_kwargs["thinking"] = enable_thinking
        normalized["chat_template_kwargs"] = chat_template_kwargs

    max_completion_tokens = raw_body.get("max_completion_tokens")
    if max_completion_tokens is not None:
        normalized["max_tokens"] = max_completion_tokens
    return normalized


_SUPPORTED_ANTHROPIC_VERSIONS = {"2023-06-01"}


_AUDIO_RESPONSE_MEDIA_TYPES = {
    "aac": "audio/aac",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "pcm": "audio/pcm",
    "wav": "audio/wav",
    "wave": "audio/wav",
    "webm": "audio/webm",
}


def _audio_response_media_type(response_format: Optional[str]) -> str:
    normalized_format = (response_format or "mp3").lower().lstrip(".")
    return _AUDIO_RESPONSE_MEDIA_TYPES.get(
        normalized_format, "application/octet-stream"
    )


def _encode_embeddings_base64(embedding: Union[bytes, str]) -> str:
    """Encode dense vectors as base64 little-endian float32, as OpenAI does.

    Sparse embeddings (token -> weight dicts) have no base64 form and are
    returned unchanged.
    """
    result = json.loads(embedding)
    for item in result.get("data") or []:
        vector = item.get("embedding")
        if isinstance(vector, list):
            packed = struct.pack(f"<{len(vector)}f", *vector)
            item["embedding"] = base64.b64encode(packed).decode("ascii")
    return json.dumps(result)


def _validate_replica(value: Any) -> int:
    """Validate and convert the ``replica`` field from a JSON payload.

    Accepts:
      - ``int`` (but not ``bool``, which is an ``int`` subclass in Python)
      - ``str`` representing a valid integer

    Rejects:
      - ``bool`` (``True`` / ``False``)
      - ``float`` (fractional values, ``Infinity``, ``NaN``)
      - ``None``
      - any other non-integral type

    Returns an ``int >= 1``.
    Raises ``HTTPException(400)`` on invalid input.
    """
    if isinstance(value, bool):
        raise HTTPException(
            status_code=400,
            detail="Invalid input. The `replica` field must be an integer, "
            "got a boolean.",
        )
    if isinstance(value, float):
        raise HTTPException(
            status_code=400,
            detail="Invalid input. The `replica` field must be an integer, "
            "got a float.",
        )
    try:
        replica = int(value)
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(
            status_code=400,
            detail="Invalid input. The `replica` field must be a valid integer.",
        )
    if replica < 1:
        raise HTTPException(
            status_code=400,
            detail="Invalid input. The `replica` field must be at least 1.",
        )
    return replica


def _parse_replica_config(payload: dict) -> Optional[List[ReplicaConfig]]:
    """Validate and parse per-replica placement from a launch payload."""
    replica_config_data = payload.get("replica_config")
    if replica_config_data is None:
        return None

    if (
        payload.get("worker_ip") is not None
        or payload.get("gpu_idx") is not None
        or payload.get("n_gpu", "auto") != "auto"
    ):
        raise HTTPException(
            status_code=400,
            detail="Cannot specify worker_ip / n_gpu / gpu_idx together with "
            "replica_config.",
        )

    try:
        return [ReplicaConfig.from_dict(item) for item in replica_config_data]
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid replica_config: {e}",
        )


def _log_setup_required_notice(setup_token: str) -> None:
    """Show the operator's bootstrap secret only while initial setup is pending."""
    logger.warning(
        "\n"
        + "=" * 60
        + "\n"
        + "  FIRST-RUN SETUP REQUIRED\n"
        + "  Setup token: %s\n"
        + "  Create the initial admin account at POST /v1/admin/setup\n"
        + "  (or via the web UI's setup page).\n"
        + "=" * 60,
        setup_token,
    )


class RESTfulAPI(CancelMixin):
    # Add new class attributes
    _allowed_ip_list: Optional[List[ipaddress.IPv4Network]] = None
    _cluster_metrics_task: Optional[asyncio.Task[None]] = None
    QWEN38_REASONING_EFFORTS = {"xhigh", "medium", "low"}
    QWEN38_REASONING_MODEL_NAMES = {"qwen3.8", "qwen3.8-max", "qwen3.8-Flash-Next"}

    @asynccontextmanager
    async def _lifespan(self, _app: FastAPI) -> AsyncIterator[None]:
        # Keep the large startup heap out of full collections while continuing
        # to collect cycles created by requests. Extend any existing freeze
        # with the API startup graph; unfreeze only when this lifespan owns it.
        owns_gc_freeze = gc.get_freeze_count() == 0
        # A runtime may already have frozen a small startup graph. Include the
        # libraries initialized for this API even then; only undo a freeze that
        # this lifespan owns, keeping the pre-existing owner's policy intact.
        gc.collect()
        gc.freeze()
        try:
            if not is_metrics_disabled():
                self._cluster_metrics_task = asyncio.create_task(
                    self._cluster_metrics_update_loop(),
                    name="cluster-metrics-updater",
                )
            yield
        finally:
            if owns_gc_freeze:
                gc.unfreeze()
            if getattr(self, "_benchmark_service", None) is not None:
                await self._benchmark_service.close()
            try:
                task = self._cluster_metrics_task
                self._cluster_metrics_task = None
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.warning(
                            "Cluster metrics updater failed during shutdown",
                            exc_info=True,
                        )
            finally:
                try:
                    await self._close_elasticsearch_client()
                finally:
                    await self._close_token_router_client()

    def __init__(
        self,
        supervisor_address: str,
        host: str,
        port: int,
        *,
        api_key_cache_enabled: bool = True,
    ):
        super().__init__()
        self._supervisor_address = supervisor_address
        self._host = host
        self._port = port
        self._supervisor_ref = None
        self._event_collector_ref = None
        self._advanced_auth_service = None
        self._auth_service: Any = None
        self._uid_to_model_name: dict = {}
        self._benchmark_service: Any = None

        # Authentication has two modes: the database-backed advanced auth
        # (enabled by default via XINFERENCE_AUTH_ADVANCED), and no auth at
        # all (XINFERENCE_AUTH_ADVANCED=0/false/no). When advanced auth is
        # disabled, _advanced_auth_service stays None and every endpoint is
        # served without authentication.
        #
        # These are resolved at construction time (not from module-level
        # constants) so a server started in a forked subprocess honors the
        # XINFERENCE_AUTH_ADVANCED value in its environment rather than a value
        # frozen when the parent first imported constants.
        if is_auth_advanced():
            jwt_secret_key = get_auth_jwt_secret_key()
            encryption_key = get_auth_encryption_key()
            if not jwt_secret_key:
                raise SystemExit(
                    "ERROR: XINFERENCE_AUTH_JWT_SECRET_KEY must be set when using advanced auth."
                )
            if not encryption_key:
                raise SystemExit(
                    "ERROR: XINFERENCE_AUTH_ENCRYPTION_KEY must be set when using advanced auth."
                )
            from .oauth2.advanced.auth_service import AdvancedAuthService

            self._advanced_auth_service = AdvancedAuthService(
                db_path=XINFERENCE_AUTH_DB_PATH,
                jwt_secret_key=jwt_secret_key,
                encryption_key=encryption_key,
                api_key_cache_enabled=api_key_cache_enabled,
            )
            self._auth_service = self._advanced_auth_service
            setup_token = self._advanced_auth_service.setup_token
            if setup_token is not None:
                _log_setup_required_notice(setup_token)

        from ..core.launch_history_store import create_launch_history_store
        from ..core.model_favorites_store import create_model_favorites_store

        self._model_favorites_store = create_model_favorites_store(
            XINFERENCE_AUTH_DB_PATH,
            self._advanced_auth_service.db if self._advanced_auth_service else None,
        )

        self._launch_history_store = create_launch_history_store(
            XINFERENCE_LAUNCH_HISTORY_DB_PATH,
            service="api",
            auth_store=(
                self._advanced_auth_service.db if self._advanced_auth_service else None
            ),
        )

        from ..core.monitor_config_store import MonitorConfigStore

        self._monitor_config_store = MonitorConfigStore(
            XINFERENCE_MONITOR_CONFIG_DB_PATH
        )

        from ..core.system_settings_store import initialize_system_settings

        self._system_settings_store = initialize_system_settings(
            XINFERENCE_SYSTEM_SETTINGS_PATH
        )

        self._router = APIRouter(route_class=ModelRequestLoggingRoute)
        self._elasticsearch_client: Optional[aiohttp.ClientSession] = None
        self._token_router_client: Optional[httpx.AsyncClient] = None
        self._cluster_metrics_task = None
        self._app = FastAPI(lifespan=self._lifespan)
        # Initialize allowed IP list once
        self._init_allowed_ip_list()

    def _init_allowed_ip_list(self):
        """Initialize the allowed IP list from environment variable."""
        if RESTfulAPI._allowed_ip_list is None:
            # ie: export XINFERENCE_ALLOWED_IPS=192.168.1.0/24
            allowed_ips = XINFERENCE_ALLOWED_IPS
            if allowed_ips:
                RESTfulAPI._allowed_ip_list = []
                for ip in allowed_ips.split(","):
                    ip = ip.strip()
                    try:
                        # Try parsing as network/CIDR
                        if "/" in ip:
                            RESTfulAPI._allowed_ip_list.append(ipaddress.ip_network(ip))
                        else:
                            # Parse as single IP
                            RESTfulAPI._allowed_ip_list.append(
                                ipaddress.ip_network(f"{ip}/32")
                            )
                    except ValueError:
                        logger.error(
                            f"Invalid IP address or network: {ip}", exc_info=True
                        )
                        continue

    def _is_ip_allowed(self, ip: str) -> bool:
        """Check if an IP is allowed based on configured rules."""
        if not RESTfulAPI._allowed_ip_list:
            return True

        try:
            client_ip = ipaddress.ip_address(ip)
            return any(
                client_ip in allowed_net for allowed_net in RESTfulAPI._allowed_ip_list
            )
        except ValueError:
            return False

    def is_authenticated(self):
        return self._advanced_auth_service is not None

    def _check_model_access(
        self, request, model_uid: str, model_type: Optional[str] = None
    ):
        if not self._advanced_auth_service:
            return
        token = _request_credential(request)
        if not token:
            return
        entry = getattr(request.state, "authenticated_api_key", None)
        allowed = (
            entry.has_model_access(model_uid, model_type)
            if entry is not None
            else self._advanced_auth_service.validate_model_access(
                token, model_uid, model_type
            )
        )
        if not allowed:
            request.state._audit_model_uid = model_uid
            request.state._audit_model_type = model_type or ""
            request.state.audit_status = "denied"
            raise HTTPException(
                status_code=403,
                detail=f"API key does not have access to model: {model_uid}",
            )
        # Store audit context for deferred recording after request completes
        request.state._audit_model_uid = model_uid
        request.state._audit_model_type = model_type or ""

    def _record_audit(
        self,
        request,
        model_uid: str,
        model_type: str,
        status: str,
        latency_s: float = 0.0,
    ):
        if not self._advanced_auth_service:
            return
        identity = getattr(request.state, "audit_identity", {}) or {}
        if identity.get("auth_type") != "api_key":
            return
        username = identity.get("user", "")
        api_key_name = identity.get("api_key_name", "")
        api_key_prefix = identity.get("api_key_prefix", "")

        model_name = (
            getattr(request.state, "model_name", "")
            or self._uid_to_model_name.get(model_uid, "")
            or ""
        )

        from ..core import metrics as _metrics
        from .oauth2.advanced.audit import record_audit_event

        record_audit_event(
            user=username,
            api_key_name=api_key_name,
            api_key_prefix=api_key_prefix,
            model_id=model_uid,
            model_name=model_name,
            model_type=model_type,
            endpoint=request.url.path,
            status=status,
            latency_ms=round(latency_s * 1000, 1),
            client_ip=request.client.host if request.client else "",
            category="inference",
            auth_type="api_key",
            method=request.method,
            status_code=getattr(request.state, "audit_status_code", 0),
            request_id=str(getattr(request.state, "model_request_id", "") or ""),
        )
        request.state.audit_recorded = True
        _requests_total = getattr(_metrics, "api_key_requests_total", None)
        if _requests_total is not None:
            _requests_total.inc(
                {
                    "user": username,
                    "api_key_name": api_key_name,
                    "model_id": model_uid,
                    "model_name": model_name,
                    "model_type": model_type,
                    "status": status,
                }
            )
        _duration = getattr(_metrics, "api_key_request_duration_seconds", None)
        if _duration is not None:
            _duration.observe(
                {
                    "model_id": model_uid,
                    "model_type": model_type,
                    "model_name": model_name,
                },
                latency_s,
            )

    def _record_admin_audit(
        self,
        request: Request,
        status: str,
        latency_s: float = 0.0,
        status_code: int = 0,
        category: str = "",
    ) -> None:
        if not self._advanced_auth_service:
            return
        identity = getattr(request.state, "audit_identity", {}) or {}
        username = str(identity.get("user", ""))
        api_key_name = str(identity.get("api_key_name", ""))
        api_key_prefix = str(identity.get("api_key_prefix", ""))
        auth_type = str(identity.get("auth_type", ""))
        if not username and not auth_type:
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            if token:
                payload = self._advanced_auth_service.verify_access_token(token)
                username = payload.get("sub", "") if payload else ""
                auth_type = "jwt" if payload else ""

        from .oauth2.advanced.audit import classify_endpoint, record_audit_event

        record_audit_event(
            user=username,
            api_key_name=api_key_name,
            api_key_prefix=api_key_prefix,
            model_id=str(getattr(request.state, "audit_model_id", "") or ""),
            model_name=str(getattr(request.state, "audit_model_name", "") or ""),
            model_type=str(getattr(request.state, "audit_model_type", "") or ""),
            endpoint=request.url.path,
            status=status,
            latency_ms=round(latency_s * 1000, 1),
            client_ip=request.client.host if request.client else "",
            category=category or classify_endpoint(request.url.path),
            auth_type=auth_type,
            method=request.method,
            status_code=status_code,
            request_id=str(getattr(request.state, "model_request_id", "") or ""),
        )
        request.state.audit_recorded = True

    async def _audit_middleware(self, request: Request, call_next):
        from .oauth2.advanced.audit import (
            classify_endpoint,
            should_skip_audit,
            should_skip_completed_audit,
        )

        started = time.perf_counter()
        request_id = get_model_request_id(request)
        try:
            response = await call_next(request)
        except Exception:
            if (
                self._advanced_auth_service
                and not should_skip_audit(request.url.path)
                and not getattr(request.state, "audit_recorded", False)
            ):
                request.state.audit_status_code = 500
                self._record_admin_audit(
                    request,
                    "error",
                    time.perf_counter() - started,
                    status_code=500,
                )
            raise

        response.headers.setdefault("X-Request-ID", request_id)
        if (
            not self._advanced_auth_service
            or should_skip_audit(request.url.path)
            or getattr(request.state, "audit_recorded", False)
        ):
            return response

        status_code = response.status_code
        if should_skip_completed_audit(request.method, request.url.path, status_code):
            return response

        latency_s = time.perf_counter() - started
        request.state.audit_status_code = status_code
        model_uid = getattr(request.state, "_audit_model_uid", "")
        audit_status = getattr(request.state, "audit_status", "")
        if not audit_status:
            if status_code < 400:
                audit_status = "success"
            elif model_uid and status_code == 404:
                audit_status = "model_not_found"
            elif request.method == "POST" and request.url.path == "/token":
                audit_status = "login_failed"
            else:
                audit_status = "error"

        if model_uid:
            model_type = getattr(request.state, "_audit_model_type", "")
            self._record_audit(request, model_uid, model_type, audit_status, latency_s)
        if not getattr(request.state, "audit_recorded", False):
            if model_uid:
                request.state.audit_model_id = model_uid
                request.state.audit_model_type = getattr(
                    request.state, "_audit_model_type", ""
                )
            self._record_admin_audit(
                request,
                audit_status,
                latency_s,
                status_code=status_code,
                category=classify_endpoint(request.url.path),
            )
        return response

    @staticmethod
    def handle_request_limit_error(e: Exception, *, sanitize: bool = False):
        from ..model.llm.context_length import ContextLengthExceeded

        message = str(e)
        context_error = isinstance(e, ContextLengthExceeded)
        if sanitize and (
            isinstance(e, ValueError)
            or (
                type(e).__module__ == "vllm.exceptions"
                and type(e).__name__ == "VLLMValidationError"
            )
        ):
            # Recognize vLLM's native validation type without importing the
            # optional engine. Match known prompt-length forms, but never echo
            # worker text (which can contain credentials as well as actor IDs).
            context_error = context_error or bool(
                re.search(
                    r"The (?:decoder|encoder) prompt \(length \d+\) "
                    r"(?:is longer than|plus the number of requested output tokens "
                    r"\(at least 1\) is longer than) the maximum model length of \d+\.",
                    message,
                )
            )
        if sanitize and isinstance(e, ValueError):
            if re.search(
                r"(?:Invalid|Empty) (?:base64 )?data URI"
                r"(?: for (?:image_url|video_url))?$",
                message,
            ):
                raise HTTPException(status_code=400, detail="Invalid media data URI.")
            if message.endswith(
                (
                    "This vLLM V1 version does not support cache_salt",
                    "cache_salt on legacy vLLM requires Xavier",
                    "cache_salt is not supported by this model engine",
                )
            ):
                raise HTTPException(
                    status_code=400,
                    detail="The model backend does not support cache_salt for this request.",
                )
        if context_error:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Request exceeds the model's configured context length."
                    if sanitize
                    else str(e)
                ),
            )
        if "Rate limit reached" in message or (
            sanitize and isinstance(e, HTTPException) and e.status_code == 429
        ):
            raise HTTPException(
                status_code=429, detail="Rate limit reached" if sanitize else str(e)
            )

    @staticmethod
    def _chat_completion_error_detail() -> str:
        """Return a client-safe error while preserving the original in logs/events."""
        return "The chat completion request failed. Check the server logs for details."

    @staticmethod
    def _count_multimodal_items(messages: List[dict], item_type: str) -> int:
        count = 0
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == f"{item_type}_url" or item_type in item:
                    count += 1
        return count

    @staticmethod
    def _set_trace_model(model_uid: Optional[str]) -> None:
        if not model_uid:
            return

        try:
            from opentelemetry.trace import get_current_span

            span = get_current_span()
            if span is not None and span.is_recording():
                span.set_attribute("xinference.model_uid", model_uid)
        except ImportError:
            return
        except Exception:
            logger.debug("Failed to attach model uid to current trace span.")

    @staticmethod
    def _set_trace_model_type(model_type: Optional[str]) -> None:
        if not model_type:
            return

        try:
            from opentelemetry.trace import get_current_span

            span = get_current_span()
            if span is not None and span.is_recording():
                span.set_attribute("xinference.model_type", model_type)
        except ImportError:
            return
        except Exception:
            logger.debug("Failed to attach model type to current trace span.")

    async def _get_supervisor_ref(self) -> xo.ActorRefType[SupervisorActor]:
        if self._supervisor_ref is None:
            self._supervisor_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=SupervisorActor.default_uid()
            )
        return self._supervisor_ref

    async def _get_event_collector_ref(self) -> xo.ActorRefType[EventCollectorActor]:
        if self._event_collector_ref is None:
            self._event_collector_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=EventCollectorActor.default_uid()
            )
        return self._event_collector_ref

    async def _report_error_event(self, model_uid: Optional[str], content: str) -> None:
        if model_uid is None:
            return
        try:
            event_collector_ref = await self._get_event_collector_ref()
            await event_collector_ref.report_event(
                model_uid,
                Event(
                    event_type=EventType.ERROR,
                    event_ts=int(time.time()),
                    event_content=content,
                ),
            )
        except Exception:
            logger.exception(
                "Report error event failed, model: %s, content: %s", model_uid, content
            )

    def _get_elasticsearch_client(self) -> aiohttp.ClientSession:
        client = getattr(self, "_elasticsearch_client", None)
        if client is None or client.closed:
            client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
            self._elasticsearch_client = client
        return client

    async def _close_elasticsearch_client(self) -> None:
        client = getattr(self, "_elasticsearch_client", None)
        if client is not None:
            await client.close()
            self._elasticsearch_client = None

    def _get_token_router_client(self) -> httpx.AsyncClient:
        client = getattr(self, "_token_router_client", None)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=None, follow_redirects=False)
            self._token_router_client = client
        return client

    async def _close_token_router_client(self) -> None:
        client = getattr(self, "_token_router_client", None)
        if client is not None:
            await client.aclose()
            self._token_router_client = None

    async def _open_token_router_chat_completion(
        self,
        request: Request,
        raw_body: Dict[str, Any],
        runtime: Dict[str, Any],
        *,
        forward_external_credential: bool = True,
    ) -> tuple[httpx.Response, str]:
        request_id = get_model_request_id(request)
        upstream_url = f"{runtime['endpoint']}/v1/chat/completions"
        client = self._get_token_router_client()
        upstream_request = client.build_request(
            "POST",
            upstream_url,
            headers=_token_router_request_headers(
                request,
                request_id,
                forward_external_credential=forward_external_credential,
            ),
            json=raw_body,
        )
        upstream_response = await client.send(upstream_request, stream=True)
        return upstream_response, request_id

    async def _proxy_token_router_chat_completion(
        self,
        request: Request,
        raw_body: Dict[str, Any],
        runtime: Dict[str, Any],
    ) -> Response:
        request_id = get_model_request_id(request)
        endpoint = runtime["endpoint"]
        upstream_url = f"{endpoint}/v1/chat/completions"
        client = self._get_token_router_client()
        try:
            upstream_request = client.build_request(
                "POST",
                upstream_url,
                headers=_token_router_request_headers(request, request_id),
                json=raw_body,
            )
            upstream_response = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            logger.warning(
                "Token Router connection failed: virtual_model_uid=%s "
                "router_uid=%s instance_id=%s error=%s",
                runtime.get("virtual_model_uid"),
                runtime.get("router_uid"),
                runtime.get("instance_id"),
                exc,
            )
            return JSONResponse(
                content={
                    "error": {
                        "message": "Token Router runtime is temporarily unavailable",
                        "type": "router_unavailable",
                    }
                },
                status_code=502,
                headers={"Retry-After": "1", "X-Request-ID": request_id},
            )

        is_stream = bool(raw_body.get("stream", False))
        response_headers = _token_router_response_headers(upstream_response)
        response_headers.setdefault("x-request-id", request_id)

        if is_stream and upstream_response.status_code < 400:
            cleanup_task: asyncio.Task[None] | None = None

            async def release_resources() -> None:
                nonlocal cleanup_task
                if cleanup_task is None:
                    cleanup_task = asyncio.create_task(upstream_response.aclose())
                await asyncio.shield(cleanup_task)

            async def body_stream() -> AsyncIterator[bytes]:
                try:
                    async for chunk in observe_stream(
                        upstream_response.aiter_raw(),
                        get_stream_outcome_reporter(request),
                        failure_origin=FailureOrigin.UPSTREAM,
                    ):
                        if await request.is_disconnected():
                            report_client_disconnect(request)
                            break
                        yield chunk
                except asyncio.CancelledError as exc:
                    report_client_disconnect(request, exc)
                    raise
                except Exception as exc:
                    report_stream_failure(request, exc, FailureOrigin.UPSTREAM)
                    logger.exception(
                        "Token Router stream failed: virtual_model_uid=%s "
                        "router_uid=%s instance_id=%s",
                        runtime.get("virtual_model_uid"),
                        runtime.get("router_uid"),
                        runtime.get("instance_id"),
                    )
                    raise
                finally:
                    await release_resources()

            response_headers.setdefault("cache-control", "no-cache")
            response_headers.setdefault("x-accel-buffering", "no")
            return StreamingResponse(
                body_stream(),
                status_code=upstream_response.status_code,
                headers=response_headers,
                background=BackgroundTask(release_resources),
            )

        try:
            response_body = b"".join(
                [chunk async for chunk in upstream_response.aiter_raw()]
            )
        finally:
            await upstream_response.aclose()
        return Response(
            content=response_body,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    def serve(
        self,
        logging_conf: Optional[dict] = None,
        *,
        server_factory: Optional[Callable[[Config], Server]] = None,
        sockets: Optional[List[socket.socket]] = None,
    ):
        # Optional fastapi-guard security middleware (no-op when disabled).
        # Attached FIRST so it nests INSIDE both CORS and the audited IP
        # middleware (Starlette runs the last-added middleware outermost):
        # the effective chain is audit/IP -> CORS -> guard -> app, so guard's
        # own rejections (403/429) leave with CORS headers (cross-origin
        # browser clients can read them) and still pass through the audit
        # wrapper keeping X-Request-ID, while the XINFERENCE_ALLOWED_IPS
        # allowlist stays first in line on the way in.
        from .guard_integration import attach_guard, uvicorn_forwarded_allow_ips

        attach_guard(self._app)

        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        async def ip_restriction_middleware(request: Request, call_next):
            client_ip = request.client.host
            if not self._is_ip_allowed(client_ip):
                return PlainTextResponse(
                    status_code=403, content=f"Access denied for IP: {client_ip}\n"
                )
            response = await call_next(request)
            return response

        @self._app.middleware("http")
        async def audited_ip_middleware(request: Request, call_next):
            async def check_ip(inner_request):
                return await ip_restriction_middleware(inner_request, call_next)

            return await self._audit_middleware(request, check_ip)

        # Initialise OpenTelemetry tracing & metrics (no-op when disabled)
        if XINFERENCE_ENABLE_OTEL:
            try:
                from ..core.otel import setup_otel

                setup_otel(self._app, register_worker_metrics=False)
            except Exception:
                logger.exception(
                    "Failed to initialise OpenTelemetry — continuing without OTEL."
                )

        @self._app.exception_handler(500)
        async def internal_exception_handler(request: Request, exc: Exception):
            logger.exception("Handling request %s failed: %s", request.url, exc)
            return PlainTextResponse(
                status_code=500, content=f"Internal Server Error: {exc}"
            )

        # Attach API instance for dependency injection (Depends(get_api), etc.)
        self._app.state.api = self
        self._app.state.advanced_auth = self._advanced_auth_service
        self._app.state.monitor_config_store = self._monitor_config_store
        self._app.state.system_settings_store = self._system_settings_store

        # Register all domain routes from routers/ modules
        from .routers import register_all_routes

        register_all_routes(self)

        # Register advanced auth routes if enabled
        if self._advanced_auth_service:
            from .oauth2.advanced.routes import register_advanced_auth_routes

            register_advanced_auth_routes(self)

            from .oauth2.advanced.security_routes import register_security_routes

            register_security_routes(self)

            from ..constants import XINFERENCE_OIDC_ENABLED

            if XINFERENCE_OIDC_ENABLED:
                from .oauth2.advanced.oidc import (
                    register_oidc_routes,
                    validate_oidc_config,
                )

                validate_oidc_config()
                register_oidc_routes(self)

        from ..extensions import APIExtensionContext, configure_api, get_extensions

        if get_extensions():
            configure_api(
                APIExtensionContext(
                    app=self._app,
                    router=self._router,
                    auth_service=self._auth_service,
                    auth_enabled=self.is_authenticated(),
                    get_supervisor_ref=self._get_supervisor_ref,
                )
            )

        if is_metrics_disabled():
            logger.info(
                "Supervisor metrics is disabled due to the environment XINFERENCE_DISABLE_METRICS=1"
            )
            self._app.include_router(self._router)
        else:
            # Only remove MetricsMiddleware's HTTP counters to avoid duplicates
            # on restart; preserve xinference:* custom metrics registered at
            # module level in metrics.py.
            from ..core.metrics import _WORKER_ONLY_METRICS

            for collector in list(REGISTRY.get_all()):
                if not collector.name.startswith("xinference:"):
                    REGISTRY.deregister(collector.name)
                elif collector.name in _WORKER_ONLY_METRICS:
                    REGISTRY.deregister(collector.name)
            self._app.add_middleware(MetricsMiddleware)
            self._app.include_router(self._router)
            self._app.add_route("/metrics", metrics)

        # Check all the routes returns Response.
        # This is to avoid `jsonable_encoder` performance issue:
        # https://github.com/xorbitsai/inference/issues/647
        invalid_routes = []
        try:
            for router in self._router.routes:
                from starlette.routing import WebSocketRoute

                if isinstance(router, WebSocketRoute):
                    continue
                return_annotation = router.endpoint.__annotations__.get("return")
                # Resolve string annotations (e.g. under __future__ annotations)
                if isinstance(return_annotation, str):
                    try:
                        hints = get_type_hints(router.endpoint)
                        return_annotation = hints.get("return")
                    except Exception:
                        pass
                    # Fallback: resolve by name from the endpoint's module globals
                    if isinstance(return_annotation, str):
                        globals = getattr(router.endpoint, "__globals__", {})
                        return_annotation = globals.get(
                            return_annotation, return_annotation
                        )
                if not inspect.isclass(return_annotation) or not issubclass(
                    return_annotation, Response
                ):
                    invalid_routes.append(
                        (router.path, router.endpoint, return_annotation)
                    )
        except Exception:
            pass  # In case that some Python version does not have __annotations__
        if invalid_routes:
            raise Exception(
                f"The return value type of the following routes is not Response: \n"
                f"{pprint.pformat(invalid_routes)}"
            )

        try:
            package_file_path = __import__("xinference").__file__
            assert package_file_path is not None
            lib_location = os.path.abspath(os.path.dirname(package_file_path))
        except ImportError as e:
            raise ImportError(f"Xinference is imported incorrectly: {e}")

        ui_dist_location = os.environ.get(
            "XINFERENCE_FRONTEND_DIST_DIR",
            os.path.join(lib_location, "ui", "web", "dist"),
        )
        if not mount_frontend(self._app, Path(ui_dist_location)):
            warnings.warn(
                f"""
            The Xinference web UI is not built at expected directory: {ui_dist_location}
            The API keeps serving without the web UI. To enable it, build the
            frontend static export from the repository "frontend/" directory with
            "npm ci && npm run build" (this stages the export at the directory
            above), or set XINFERENCE_FRONTEND_DIST_DIR to an export directory,
            and restart. For frontend development, run "npm run dev" instead.
            """
            )

        config = Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_config=logging_conf,
            proxy_headers=True,
            # With the guard enabled, only explicitly trusted proxies may
            # rewrite the client address from X-Forwarded-For; trusting any
            # peer here would let a direct client forge its identity before
            # guard (or the allowlist above) ever sees the request.
            # No-op ("*") when the guard is disabled.
            forwarded_allow_ips=uvicorn_forwarded_allow_ips(),
            # Slow HTTP DoS (Slowloris) protection, see xinference/core/http_protocol.py
            http=create_hardened_http_protocol(XINFERENCE_HTTP_REQUEST_TIMEOUT),
            timeout_keep_alive=XINFERENCE_HTTP_TIMEOUT_KEEP_ALIVE,
            limit_concurrency=XINFERENCE_HTTP_LIMIT_CONCURRENCY,
        )
        server = (server_factory or Server)(config)
        if sockets is None:
            server.run()
        else:
            server.run(sockets=sockets)

    async def _cluster_metrics_update_loop(self):
        """Periodically refresh Supervisor-side Prometheus Gauges (every 15s)."""
        from ..core.metrics import update_cluster_metrics, update_security_gauges

        while True:
            try:
                supervisor_ref = await self._get_supervisor_ref()
                cluster_data = await supervisor_ref.get_cluster_metrics_data()
                models_data = await supervisor_ref.list_models()
                update_cluster_metrics(
                    cluster_data,
                    models_data,
                    supervisor_address=self._supervisor_address,
                )
                if self._advanced_auth_service:
                    update_security_gauges(self._advanced_auth_service)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, xo.ActorNotExist, xo.ServerClosed) as e:
                logger.warning("Failed to update cluster metrics: %s", e)
            except Exception:
                logger.warning("Failed to update cluster metrics", exc_info=True)

            await asyncio.sleep(15)

    async def _get_builtin_prompts(self) -> JSONResponse:
        """
        For internal usage
        """
        try:
            data = await (await self._get_supervisor_ref()).get_builtin_prompts()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_builtin_families(self) -> JSONResponse:
        """
        For internal usage
        """
        try:
            data = await (await self._get_supervisor_ref()).get_builtin_families()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def build_llm_registration_from_config(
        self, request: Request
    ) -> JSONResponse:
        try:
            body = AutoConfigLLMRequest.parse_obj(await request.json())
            from ..model.llm.config_parser import (
                build_llm_registration_from_local_config,
            )

            data = build_llm_registration_from_local_config(
                model_path=body.model_path,
                model_family=body.model_family,
            )
            return JSONResponse(content=data)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def list_models(self) -> JSONResponse:
        try:
            supervisor_ref = await self._get_supervisor_ref()
            physical_models = await supervisor_ref.list_models()
            if XINFERENCE_TOKEN_ROUTER_ENABLED:
                virtual_models = await supervisor_ref.list_virtual_models()
                models = {**physical_models, **virtual_models}
            else:
                models = physical_models

            model_list = []
            for model_id, model_info in models.items():
                self._uid_to_model_name[model_id] = model_info.get(
                    "model_name", model_id
                )
                from .oauth2.advanced.audit import update_model_cache

                update_model_cache(
                    model_id,
                    model_info.get("model_name", model_id),
                    model_info.get("model_type", ""),
                )
                model_list.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "xinference",
                        **model_info,
                    }
                )
            response = {"object": "list", "data": model_list}

            return JSONResponse(content=response)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def anthropic_list_models(self) -> JSONResponse:
        """Anthropic-compatible models endpoint"""
        try:
            # Get running models from xinference
            running_models = await (await self._get_supervisor_ref()).list_models()

            # For backward compatibility with tests, only return running models by default
            model_list = []

            # Add running models to the list
            for model_id, model_info in running_models.items():
                anthropic_model = {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "display_name": model_info.get("model_name", model_id),
                    "type": model_info.get("model_type", "model"),
                    "max_tokens": model_info.get("context_length", 4096),
                }
                model_list.append(anthropic_model)

            return JSONResponse(content=model_list)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def anthropic_get_model(self, model_id: str) -> JSONResponse:
        """Anthropic-compatible model retrieval endpoint"""
        try:
            models = await (await self._get_supervisor_ref()).list_models()

            if model_id not in models:
                raise HTTPException(
                    status_code=404, detail=f"Model not found: {model_id}"
                )
            model_info = models[model_id]

            # Convert to Anthropic format
            anthropic_model = {
                "id": model_id,  # Return the original requested ID
                "object": "model",
                "created": 0,
                "display_name": model_info.get("model_name", model_id),
                "type": model_info.get("model_type", "model"),
                "max_tokens": model_info.get("context_length", 4096),
                **model_info,
            }

            return JSONResponse(content=anthropic_model)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def describe_model(self, model_uid: str) -> JSONResponse:
        try:
            data = await (await self._get_supervisor_ref()).describe_running_model(
                model_uid
            )
            return JSONResponse(content=data)
        except ModelNotReadyError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model is loading, please retry later: {e}",
                headers={"Retry-After": "30"},
            )
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))

        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def launch_model(
        self, request: Request, wait_ready: bool = Query(True)
    ) -> JSONResponse:
        payload = await request.json()
        model_uid = payload.get("model_uid")
        model_name = payload.get("model_name")
        model_engine = payload.get("model_engine")
        model_size_in_billions = payload.get("model_size_in_billions")
        model_format = payload.get("model_format")
        quantization = payload.get("quantization")
        model_type = payload.get("model_type", "LLM")
        replica = _validate_replica(payload.get("replica", 1))
        n_gpu = payload.get("n_gpu", "auto")
        request_limits = payload.get("request_limits", None)
        peft_model_config = payload.get("peft_model_config", None)
        worker_ip = payload.get("worker_ip", None)
        gpu_idx = payload.get("gpu_idx", None)
        download_hub = payload.get("download_hub", None)
        model_path = payload.get("model_path", None)
        enable_virtual_env = payload.get("enable_virtual_env", None)
        virtual_env_packages = payload.get("virtual_env_packages", None)
        virtual_env_find_links = payload.get("virtual_env_find_links", None)
        envs = payload.get("envs", None)
        replica_config = _parse_replica_config(payload)

        exclude_keys = {
            "model_uid",
            "model_name",
            "model_engine",
            "model_size_in_billions",
            "model_format",
            "quantization",
            "model_type",
            "replica",
            "n_gpu",
            "request_limits",
            "peft_model_config",
            "worker_ip",
            "gpu_idx",
            "download_hub",
            "model_path",
            "enable_virtual_env",
            "virtual_env_packages",
            "model_python_executable",
            "virtual_env_find_links",
            "envs",
            "replica_config",
        }

        kwargs = {
            key: value for key, value in payload.items() if key not in exclude_keys
        }

        if not model_name:
            raise HTTPException(
                status_code=400,
                detail="Invalid input. Please specify the `model_name` field.",
            )
        if not model_engine and model_type == "LLM":
            raise HTTPException(
                status_code=400,
                detail="Invalid input. Please specify the `model_engine` field.",
            )

        if isinstance(gpu_idx, int):
            gpu_idx = [gpu_idx]
        if gpu_idx:
            if len(gpu_idx) % replica:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid input. Allocated gpu must be a multiple of replica.",
                )

        if peft_model_config is not None:
            peft_model_config = PeftModelConfig.from_dict(peft_model_config)
        else:
            peft_model_config = None

        try:
            model_uid = await (await self._get_supervisor_ref()).launch_builtin_model(
                model_uid=model_uid,
                model_name=model_name,
                model_engine=model_engine,
                model_size_in_billions=model_size_in_billions,
                model_format=model_format,
                quantization=quantization,
                model_type=model_type,
                replica=replica,
                n_gpu=n_gpu,
                request_limits=request_limits,
                wait_ready=wait_ready,
                peft_model_config=peft_model_config,
                worker_ip=worker_ip,
                gpu_idx=gpu_idx,
                download_hub=download_hub,
                model_path=model_path,
                enable_virtual_env=enable_virtual_env,
                virtual_env_packages=virtual_env_packages,
                virtual_env_find_links=virtual_env_find_links,
                envs=envs,
                replica_config=replica_config,
                **kwargs,
            )
        except HTTPException:
            raise
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            logger.error(str(re), exc_info=True)
            raise HTTPException(status_code=503, detail=str(re))
        except asyncio.CancelledError:
            # Cancelled by the user. str() on a CancelledError is empty, so
            # reporting it verbatim surfaces a blank error in the UI.
            logger.info("Launch of %s was cancelled", model_uid)
            raise HTTPException(status_code=499, detail="Launch cancelled")
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        # Clear negative cache so that get_model for this uid is not blocked
        # by a stale "Model not found" entry.
        from xinference.api.utils import invalidate_model_not_found_cache

        invalidate_model_not_found_cache(model_uid)

        if model_name and model_uid:
            self._uid_to_model_name[model_uid] = model_name
            from .oauth2.advanced.audit import update_model_cache

            update_model_cache(model_uid, model_name, model_type or "")

        return JSONResponse(content={"model_uid": model_uid})

    async def get_instance_info(
        self,
        model_name: Optional[str] = Query(None),
        model_uid: Optional[str] = Query(None),
    ) -> JSONResponse:
        try:
            infos = await (await self._get_supervisor_ref()).get_instance_info(
                model_name, model_uid
            )
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=infos)

    async def get_model_replicas(self, model_uid: str) -> JSONResponse:
        """Get detailed status of all replicas for a model"""
        try:
            replicas = await (await self._get_supervisor_ref()).get_replica_statuses(
                model_uid
            )
            return JSONResponse(content=replicas)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_model_reload_config(
        self, model_uid: str, request: Request
    ) -> JSONResponse:
        try:
            result = await (await self._get_supervisor_ref()).get_model_reload_config(
                model_uid
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        auth = self.__dict__.get("_advanced_auth_service")
        allowed = True
        if auth is not None:
            from fastapi.security import SecurityScopes

            user = await auth(
                request,
                SecurityScopes(scopes=["models:list"]),
                _request_credential(request),
            )
            allowed = "admin" in user.get("permissions", []) and not getattr(
                request.state, "authenticated_api_key", None
            )
        result = dict(result)
        result["can_upgrade_environment"] = allowed
        if not allowed:
            result.pop("python_executable", None)
        return JSONResponse(content=result)

    async def get_model_reload_status(self, model_uid: str) -> JSONResponse:
        try:
            result = await (await self._get_supervisor_ref()).get_model_reload_status(
                model_uid
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(content=result)

    async def reload_model(self, model_uid: str, request: Request) -> JSONResponse:
        import math

        from ..model.llm.weight_cache import validate_reload_patch

        try:
            payload = await request.json()
            if not isinstance(payload, dict) or set(payload) - {
                "model_config",
                "drain_timeout",
                "python_executable",
            }:
                raise ValueError(
                    "Expected model_config and optional drain_timeout or python_executable"
                )
            from ..core.worker_relaunch import validate_relaunch_options

            python_executable = payload.get("python_executable")
            validate_relaunch_options(python_executable)
            auth = self.__dict__.get("_advanced_auth_service")
            if python_executable is not None and auth is not None:
                from fastapi.security import SecurityScopes

                await auth(
                    request,
                    SecurityScopes(scopes=["admin"]),
                    _request_credential(request),
                )
            config = payload.get("model_config", {} if python_executable else None)
            metadata = await (await self._get_supervisor_ref()).get_model_reload_config(
                model_uid
            )
            # Validate the request's shape and types before crossing the actor boundary.
            if not isinstance(config, dict) or (not config and not python_executable):
                raise ValueError("model_config must be a non-empty object")
            if set(config) - set(metadata["parameters"]):
                raise ValueError("Unsupported reload parameters")
            engine = metadata["engine"]
            if config:
                validate_reload_patch(engine, config)
            timeout = payload.get("drain_timeout", 300)
            if (
                type(timeout) not in (int, float)
                or not math.isfinite(timeout)
                or not 0 < timeout <= 3600
            ):
                raise ValueError("drain_timeout must be between 0 and 3600 seconds")
            result = await (await self._get_supervisor_ref()).reload_model(
                model_uid, config, timeout, python_executable=python_executable
            )
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(content=result, status_code=202)

    async def get_launch_model_progress(self, model_uid: str) -> JSONResponse:
        try:
            progress_details = await (
                await self._get_supervisor_ref()
            ).get_launch_builtin_model_progress_details(model_uid)
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=progress_details)

    async def cancel_launch_model(self, model_uid: str) -> JSONResponse:
        try:
            await (await self._get_supervisor_ref()).cancel_launch_builtin_model(
                model_uid
            )
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=None)

    async def get_autostart_config(self, user: Optional[Any] = None) -> JSONResponse:
        try:
            if isinstance(user, dict):
                username = user.get("username", "")
            else:
                username = getattr(user, "username", "") if user else ""
            config = await (await self._get_supervisor_ref()).get_autostart_config(
                username=username
            )
            return JSONResponse(content=config)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_autostart_model_summary(self) -> JSONResponse:
        try:
            summary = await (
                await self._get_supervisor_ref()
            ).get_autostart_model_summary()
            return JSONResponse(content=summary)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def upsert_autostart_model(
        self, request: Request, user: Optional[Any] = None
    ) -> JSONResponse:
        try:
            if isinstance(user, dict):
                username = user.get("username", "")
            else:
                username = getattr(user, "username", "") if user else ""
            config = await (await self._get_supervisor_ref()).upsert_autostart_model(
                await request.json(), username=username
            )
            return JSONResponse(content=config)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def remove_autostart_model(self, model_uid: str) -> JSONResponse:
        try:
            config = await (await self._get_supervisor_ref()).remove_autostart_model(
                model_uid
            )
            return JSONResponse(content=config)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def launch_model_by_version(
        self, request: Request, wait_ready: bool = Query(True)
    ) -> JSONResponse:
        payload = await request.json()
        model_uid = payload.get("model_uid")
        model_engine = payload.get("model_engine")
        model_type = payload.get("model_type")
        model_version = payload.get("model_version")
        replica = _validate_replica(payload.get("replica", 1))
        n_gpu = payload.get("n_gpu", "auto")
        replica_config = _parse_replica_config(payload)

        try:
            model_uid = await (
                await self._get_supervisor_ref()
            ).launch_model_by_version(
                model_uid=model_uid,
                model_engine=model_engine,
                model_type=model_type,
                model_version=model_version,
                replica=replica,
                n_gpu=n_gpu,
                wait_ready=wait_ready,
                replica_config=replica_config,
            )
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content={"model_uid": model_uid})

    async def get_model_versions(
        self, model_type: str, model_name: str
    ) -> JSONResponse:
        try:
            content = await (await self._get_supervisor_ref()).get_model_versions(
                model_type, model_name
            )
            return JSONResponse(content=content)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def terminate_model(self, model_uid: str) -> JSONResponse:
        try:
            assert self._app is not None
            await (await self._get_supervisor_ref()).terminate_model(model_uid)
            self._app.router.routes = [
                route
                for route in self._app.router.routes
                if not (
                    hasattr(route, "path")
                    and isinstance(route.path, str)
                    and route.path == "/" + model_uid
                )
            ]
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        self._uid_to_model_name.pop(model_uid, None)
        from .oauth2.advanced.audit import evict_model_cache

        evict_model_cache(model_uid)
        return JSONResponse(content=None)

    async def add_model_replica(
        self, model_uid: str, payload: Optional[dict[str, Any]] = None
    ) -> JSONResponse:
        """Add one or more replicas to a running model (scale-up).

        ``replica`` controls how many replicas are added. ``model_engine`` and
        ``n_gpu`` may override those launch settings for the new replicas only.
        An optional ``replica_config`` dict applies to every new replica, while
        a list can provide individual placement for each one.
        """
        try:
            payload = payload or {}
            replica = payload.get("replica", 1)
            if isinstance(replica, bool) or not isinstance(replica, int) or replica < 1:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid replica: expected a positive integer.",
                )

            model_engine = payload.get("model_engine")
            if model_engine is not None:
                if not isinstance(model_engine, str) or not model_engine.strip():
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid model_engine: expected a non-empty string.",
                    )
                model_engine = model_engine.strip()

            n_gpu = payload.get("n_gpu")
            if (
                n_gpu is not None
                and n_gpu != "auto"
                and (isinstance(n_gpu, bool) or not isinstance(n_gpu, int) or n_gpu < 0)
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid n_gpu: expected a non-negative integer or 'auto'.",
                )

            replica_config_raw = payload.get("replica_config", None)
            replica_configs: Optional[List[ReplicaConfig]] = None
            if replica_config_raw is not None:
                raw_configs = (
                    replica_config_raw
                    if isinstance(replica_config_raw, list)
                    else [replica_config_raw] * replica
                )
                if len(raw_configs) != replica or any(
                    not isinstance(config, dict) for config in raw_configs
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Invalid replica_config: expected a dict or a list "
                            "with one entry per requested replica."
                        ),
                    )
                try:
                    replica_configs = [
                        ReplicaConfig.from_dict(config) for config in raw_configs
                    ]
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid replica_config: {e}",
                    )
                if any(len(config.devices) != 1 for config in replica_configs):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Each replica_config must contain exactly one device "
                            "entry for scale-up."
                        ),
                    )

            results = await (await self._get_supervisor_ref()).add_model_replicas(
                model_uid=model_uid,
                replica=replica,
                replica_configs=replica_configs,
                model_engine=model_engine,
                n_gpu=n_gpu,
            )
            if replica == 1:
                # Preserve the existing response shape for older callers.
                return JSONResponse(content=results[0])
            return JSONResponse(content={"replica": replica, "replicas": results})
        except HTTPException:
            raise
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            logger.error(str(re), exc_info=True)
            raise HTTPException(status_code=503, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def terminate_model_replica(
        self, model_uid: str, replica_id: int
    ) -> JSONResponse:
        try:
            assert self._app is not None
            remaining_replicas = await (
                await self._get_supervisor_ref()
            ).terminate_model_replica(model_uid, replica_id)
            if remaining_replicas == 0:
                self._app.router.routes = [
                    route
                    for route in self._app.router.routes
                    if not (
                        hasattr(route, "path")
                        and isinstance(route.path, str)
                        and route.path == "/" + model_uid
                    )
                ]
                self._uid_to_model_name.pop(model_uid, None)
                from .oauth2.advanced.audit import evict_model_cache

                evict_model_cache(model_uid)
            return JSONResponse(content={"remaining_replicas": remaining_replicas})
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def _iter_openai_sse_bytes(
        byte_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[Dict[str, Any]]:
        buffer = b""
        async for chunk in byte_stream:
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == b"[DONE]":
                    continue
                try:
                    value = json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning("Ignoring malformed OpenAI SSE payload")
                    continue
                if isinstance(value, dict):
                    yield value
        line = buffer.strip()
        if line.startswith(b"data:"):
            payload = line[5:].strip()
            if payload and payload != b"[DONE]":
                try:
                    value = json.loads(payload)
                except json.JSONDecodeError:
                    return
                if isinstance(value, dict):
                    yield value

    @classmethod
    async def _iter_model_openai_chunks(
        cls, iterator: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        async def _items() -> AsyncIterator[Any]:
            if hasattr(iterator, "__aiter__"):
                async for item in iterator:
                    yield item
            elif isinstance(iterator, (str, bytes, dict)):
                yield iterator
            else:
                for item in iterator:
                    yield item

        async for item in _items():
            if isinstance(item, dict) and "data" not in item:
                yield item
                continue
            payload = item.get("data") if isinstance(item, dict) else item
            if isinstance(payload, bytes):
                payload_bytes = payload
                payload_text = payload.decode("utf-8", errors="replace")
            elif isinstance(payload, str):
                payload_text = payload
                payload_bytes = payload.encode("utf-8")
            else:
                continue

            if any(
                line.lstrip().startswith("data:") for line in payload_text.splitlines()
            ):

                async def _single_payload() -> AsyncIterator[bytes]:
                    yield payload_bytes

                async for value in cls._iter_openai_sse_bytes(_single_payload()):
                    yield value
                continue

            payload_text = payload_text.strip()
            if not payload_text or payload_text == "[DONE]":
                continue
            try:
                value = json.loads(payload_text)
            except json.JSONDecodeError:
                yield {"choices": [{"delta": {"content": payload_text}}]}
                continue
            if isinstance(value, dict):
                yield value

    async def _get_model_last_error(self, replica_model_uid: bytes, e: Exception):
        if not isinstance(e, xo.ServerClosed):
            return e
        try:
            model_status = await (await self._get_supervisor_ref()).get_model_status(
                replica_model_uid.decode("utf-8")
            )
            if model_status is not None and model_status.last_error:
                return Exception(model_status.last_error)
        except Exception as ex:
            return ex
        return e

    @staticmethod
    def _apply_cache_salt(request: Request, source: dict, *targets: dict) -> None:
        from ..cache_isolation import validate_cache_salt
        from ..extensions import resolve_cache_salt

        try:
            provided = resolve_cache_salt(request)
        except Exception:
            # Extension errors may contain credentials; do not log or echo them.
            raise HTTPException(500, "Cache salt provider failed") from None
        try:
            salt = validate_cache_salt(
                provided.value if provided is not None else source.get("cache_salt")
            )
            if XINFERENCE_REQUIRE_CACHE_SALT and salt is None:
                raise ValueError("cache_salt is required")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        for target in (source, *targets):
            target.pop("cache_salt", None)
            if salt is not None:
                target["cache_salt"] = salt

    async def create_completion(self, request: Request) -> Response:
        raw_body = await request.json()
        # Resolve/validate the raw salt after authorization, allowing provider overrides.
        body = CreateCompletionRequest.parse_obj(
            {key: value for key, value in raw_body.items() if key != "cache_salt"}
        )
        exclude = {
            "prompt",
            "model",
            "n",
            "best_of",
            "logit_bias",
            "logit_bias_type",
            "user",
        }
        raw_kwargs = {k: v for k, v in raw_body.items() if k not in exclude}
        kwargs = body.dict(exclude_unset=True, exclude=exclude)
        request_id = str(kwargs.get("request_id") or uuid.uuid4().hex)

        # guided_decoding params
        kwargs.update(self.extract_guided_params(raw_body=raw_body))

        # TODO: Decide if this default value override is necessary #1061
        if body.max_tokens is None:
            kwargs["max_tokens"] = max_tokens_field.default

        if body.logit_bias is not None:
            raise HTTPException(status_code=501, detail="Not implemented")

        model_uid = body.model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("llm")
        self._check_model_access(request, model_uid, "LLM")

        self._apply_cache_salt(request, raw_body, kwargs, raw_kwargs)

        model = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )
        is_vllm_backend = await model.is_vllm_backend()
        model_desc = await (await self._get_supervisor_ref()).describe_model(model_uid)
        RESTfulAPI._apply_multiple_choice_count(
            kwargs,
            getattr(body, "n", None),
            model_desc.get("model_engine"),
            is_vllm_backend,
        )
        model_call_kwargs: Dict[str, Any] = {"raw_params": raw_kwargs}
        if is_vllm_backend:
            kwargs.pop("request_id", None)
            raw_kwargs.pop("request_id", None)
            model_call_kwargs["request_id"] = request_id
        elif await model.is_transformers_batching_enabled():
            model_call_kwargs["request_id"] = request_id

        if body.stream:
            abort_controller = _SSERequestAbortController(
                model, request_id, request, "generate"
            )

            async def stream_results():
                abort_controller.bind_stream_task()
                iterator = None
                observed_iterator = None
                completed = False
                try:
                    try:
                        iterator = await model.generate(
                            body.prompt,
                            kwargs,
                            **model_call_kwargs,
                        )
                    except RuntimeError as re:
                        self.handle_request_limit_error(re)
                    observed_iterator = observe_stream(
                        iterator,
                        get_stream_outcome_reporter(request),
                        failure_origin=FailureOrigin.MODEL_GENERATOR,
                    )
                    async for item in observed_iterator:
                        yield item
                    completed = True
                except asyncio.CancelledError as exc:
                    report_client_disconnect(request, exc)
                    logger.info(
                        f"Disconnected from client (via refresh/close) {request.client} during generate."
                    )
                    await abort_controller.abort()
                    return
                except Exception as ex:
                    report_stream_failure(request, ex, FailureOrigin.MODEL_GENERATOR)
                    ex = await self._get_model_last_error(model.uid, ex)
                    logger.exception("Completion stream got an error: %s", ex)
                    await self._report_error_event(model_uid, str(ex))
                    # https://github.com/openai/openai-python/blob/e0aafc6c1a45334ac889fe3e54957d309c3af93f/src/openai/_streaming.py#L107
                    yield dict(data=json.dumps({"error": str(ex)}))
                    return
                finally:
                    await _cleanup_model_stream(
                        abort_controller=abort_controller,
                        interrupted=not completed,
                        observed_iterator=observed_iterator,
                        model_iterator=iterator,
                        model=model,
                        request_id=request_id,
                        kind="completion",
                    )

            body_iterator = stream_results()
            response = EventSourceResponse(
                body_iterator, ping=XINFERENCE_SSE_PING_ATTEMPTS_SECONDS
            )
            abort_controller.bind_body_iterator(body_iterator)
            _observe_sse_disconnect(response, abort_controller.handle_disconnect)
            return response
        else:
            try:
                if is_vllm_backend:
                    return await _run_non_streaming_request(
                        request,
                        model,
                        request_id,
                        "generate",
                        model.generate(body.prompt, kwargs, **model_call_kwargs),
                    )
                data = await model.generate(
                    body.prompt,
                    kwargs,
                    **model_call_kwargs,
                )
                return Response(data, media_type="application/json")
            except Exception as e:
                e = await self._get_model_last_error(model.uid, e)
                logger.error(e, exc_info=True)
                await self._report_error_event(model_uid, str(e))
                self.handle_request_limit_error(e)
                raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    def _normalize_anthropic_messages(
        raw_system: Any, messages: Optional[List[dict]]
    ) -> List[dict]:
        """Fold the Anthropic top-level ``system`` prompt and any inline
        ``role: system`` messages into a single leading OpenAI-style system
        message.

        Anthropic's Messages API carries the system prompt in a top-level
        ``system`` field, and some clients (notably Claude Code >= 2.1.154)
        additionally place ``role: system`` entries inside the ``messages``
        array. xinference dispatches to an OpenAI-style ``chat`` backend, which
        expects the system prompt as a leading ``{"role": "system"}`` message
        and only accepts ``user``/``assistant`` roles in the array. Without this
        normalization the top-level ``system`` prompt is silently dropped (the
        ``raw_params`` carrying it are discarded before the model is called) and
        inline system messages trip the role validation below.
        """

        def _collect(content: Any, parts: List[str]) -> None:
            if isinstance(content, str):
                if content:
                    parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "text":
                        continue
                    text = block.get("text") or ""
                    # Strip Claude Code's per-request billing header; it carries
                    # a changing hash that would otherwise defeat prefix caching.
                    if text.startswith("x-anthropic-billing-header"):
                        continue
                    parts.append(text)

        system_parts: List[str] = []
        _collect(raw_system, system_parts)

        normalized: List[dict] = []
        for msg in messages or []:
            if msg.get("role") == "system":
                _collect(msg.get("content"), system_parts)
                continue
            normalized.append(msg)

        if system_parts:
            normalized.insert(0, {"role": "system", "content": "\n".join(system_parts)})
        return normalized

    @staticmethod
    def _anthropic_error_type(status_code: int, exc: Optional[Exception] = None) -> str:
        from ..model.llm.context_length import ContextLengthExceeded

        if isinstance(exc, ContextLengthExceeded):
            return "invalid_request_error"
        return {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            413: "request_too_large",
            429: "rate_limit_error",
            504: "timeout_error",
            529: "overloaded_error",
        }.get(status_code, "api_error")

    @classmethod
    def _anthropic_error(
        cls,
        status_code: int,
        message: str,
        request_id: str,
        *,
        error_type: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        exc: Optional[Exception] = None,
    ) -> JSONResponse:
        from ..model.llm.context_length import ContextLengthExceeded

        if isinstance(exc, ContextLengthExceeded):
            status_code = 400
        response_headers = {"request-id": request_id}
        if headers:
            response_headers.update(headers)
        return JSONResponse(
            status_code=status_code,
            headers=response_headers,
            content=anthropic_error_response(
                error_type or cls._anthropic_error_type(status_code),
                message,
                request_id,
            ),
        )

    async def count_anthropic_tokens(self, request: Request) -> Response:
        request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex}"
        version = request.headers.get("anthropic-version")
        if (
            get_request_route_path(request) == "/v1/messages/count_tokens"
            and not version
        ):
            return self._anthropic_error(
                400, "The anthropic-version header is required", request_id
            )
        if version is not None and version not in _SUPPORTED_ANTHROPIC_VERSIONS:
            return self._anthropic_error(
                400, f"Unsupported anthropic-version header: {version}", request_id
            )
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise AnthropicProtocolError(
                    400, "invalid_request_error", "Request body must be an object"
                )
            body = dict(body)
            thinking = body.get("thinking") or {}
            budget = (
                thinking.get("budget_tokens", 0) if isinstance(thinking, dict) else 0
            )
            body["max_tokens"] = max(1, budget + 1) if isinstance(budget, int) else 1
            body["stream"] = False
            self._check_model_access(request, anthropic_request_model(body), "LLM")
            canonical = await asyncio.to_thread(parse_anthropic_request, body)
            self._set_trace_model(canonical.requested_model)
            self._set_trace_model_type("llm")
            supervisor = await self._get_supervisor_ref()
            if (
                await supervisor.resolve_token_router_runtime(canonical.requested_model)
                is not None
            ):
                raise NotImplementedError(
                    "Token counting for Token Router virtual models is not available"
                )
            model = await require_model(
                self._get_supervisor_ref,
                canonical.requested_model,
                self._report_error_event,
            )
            config = canonical.to_openai_body()
            config.pop("model", None)
            config.pop("messages", None)
            # Use the identical backend preparation as generation (including effort).
            prepared = await model.prepare_anthropic_request(canonical.messages, config)
            if isinstance(prepared, (str, bytes)):
                prepared = json.loads(prepared)
            count = prepared["input_tokens"]
            if count == 0:
                # Preparation uses zero when counting is unavailable. Preserve
                # the backend's explicit unsupported/rate-limit error in that case.
                count = await model.count_tokens(
                    prepared["messages"], prepared["generate_config"]
                )
            if isinstance(count, (str, bytes)):
                count = json.loads(count)
            return JSONResponse(
                {"input_tokens": count}, headers={"request-id": request_id}
            )
        except AnthropicProtocolError as exc:
            return self._anthropic_error(
                exc.status_code, exc.message, request_id, error_type=exc.error_type
            )
        except NotImplementedError as exc:
            return self._anthropic_error(501, str(exc), request_id)
        except HTTPException as exc:
            return self._anthropic_error(
                exc.status_code, str(exc.detail), request_id, headers=exc.headers
            )
        except ValueError as exc:
            return self._anthropic_error(400, str(exc), request_id)
        except Exception as exc:
            logger.exception("Anthropic token counting failed")
            status_code = 429 if "Rate limit reached" in str(exc) else 500
            return self._anthropic_error(status_code, str(exc), request_id)

    async def create_message(self, request: Request) -> Response:
        # Keep Anthropic protocol response IDs backward compatible. The route-level
        # correlation ID is exposed separately through ``X-Request-ID``.
        request_id = (
            request.headers.get("request-id")
            or request.headers.get("x-request-id")
            or f"req_{uuid.uuid4().hex}"
        )
        anthropic_version = request.headers.get("anthropic-version")
        if get_request_route_path(request) == "/v1/messages" and not anthropic_version:
            return self._anthropic_error(
                400, "The anthropic-version header is required", request_id
            )
        if (
            anthropic_version is not None
            and anthropic_version not in _SUPPORTED_ANTHROPIC_VERSIONS
        ):
            return self._anthropic_error(
                400,
                f"Unsupported anthropic-version header: {anthropic_version}",
                request_id,
            )

        try:
            body = await request.json()
            self._check_model_access(request, anthropic_request_model(body), "LLM")
            self._apply_cache_salt(request, body)
            canonical = await asyncio.to_thread(parse_anthropic_request, body)
        except HTTPException as exc:
            return self._anthropic_error(
                exc.status_code, str(exc.detail), request_id, headers=exc.headers
            )
        except AnthropicProtocolError as exc:
            return self._anthropic_error(
                exc.status_code,
                exc.message,
                request_id,
                error_type=exc.error_type,
                headers=exc.headers,
            )
        except Exception:
            return self._anthropic_error(
                400, "Request body must be valid JSON", request_id
            )

        model_uid = canonical.requested_model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("llm")

        try:
            supervisor_ref = await self._get_supervisor_ref()
            token_router_runtime = await supervisor_ref.resolve_token_router_runtime(
                model_uid
            )
        except Exception:
            logger.exception("Failed to resolve Anthropic model target: %s", model_uid)
            return self._anthropic_error(
                500, "Failed to resolve the requested model", request_id
            )

        openai_body = canonical.to_openai_body()
        if body.get("cache_salt") is not None:
            openai_body["cache_salt"] = body["cache_salt"]
        if token_router_runtime is not None:
            if canonical.parallel_tool_calls is False:
                return self._anthropic_error(
                    501,
                    "disable_parallel_tool_use is not supported by Token Router",
                    request_id,
                )
            # This worker-local hint is not an OpenAI wire field. Router models
            # retain their upstream engine's automatic prefix caching.
            openai_body.pop("_anthropic_cache_control", None)
            openai_body.pop("_anthropic_content_metrics", None)
            openai_body.pop("_anthropic_requested_features", None)
            if not token_router_runtime.get("available"):
                return self._anthropic_error(
                    503,
                    f"No ready Token Router runtime is available for virtual model {model_uid}",
                    request_id,
                    headers={"Retry-After": "1"},
                )
            try:
                (
                    upstream_response,
                    request_id,
                ) = await self._open_token_router_chat_completion(
                    request,
                    openai_body,
                    token_router_runtime,
                    forward_external_credential=False,
                )
            except httpx.HTTPError:
                logger.exception(
                    "Anthropic Token Router connection failed: virtual_model_uid=%s",
                    model_uid,
                )
                return self._anthropic_error(
                    502,
                    "Token Router runtime is temporarily unavailable",
                    request_id,
                    headers={"Retry-After": "1"},
                )

            if upstream_response.status_code >= 400:
                retry_after = upstream_response.headers.get("retry-after")
                try:
                    raw_error = await upstream_response.aread()
                    error_payload = json.loads(raw_error) if raw_error else {}
                except Exception:
                    error_payload = {}
                finally:
                    await upstream_response.aclose()
                error = error_payload.get("error") or {}
                error_message = (
                    error.get("message") if isinstance(error, dict) else str(error)
                )
                return self._anthropic_error(
                    upstream_response.status_code,
                    error_message or "Token Router request failed",
                    request_id,
                    headers={"Retry-After": retry_after} if retry_after else None,
                )

            router_response_headers = {"request-id": request_id}
            if canonical.cache_control:
                router_response_headers["x-xinference-cache-control"] = "automatic"
            if canonical.stream:
                cleanup_task: asyncio.Task[None] | None = None

                async def release_resources() -> None:
                    nonlocal cleanup_task
                    if cleanup_task is None:
                        cleanup_task = asyncio.create_task(upstream_response.aclose())
                    await asyncio.shield(cleanup_task)

                async def router_chunks() -> AsyncIterator[Dict[str, Any]]:
                    reporter = get_stream_outcome_reporter(request)
                    upstream_chunks = observe_stream(
                        upstream_response.aiter_raw(),
                        reporter,
                        failure_origin=FailureOrigin.UPSTREAM,
                    )
                    protocol_chunks = observe_stream(
                        self._iter_openai_sse_bytes(upstream_chunks),
                        reporter,
                        failure_origin=FailureOrigin.PROTOCOL,
                    )
                    try:
                        async for chunk in protocol_chunks:
                            if await request.is_disconnected():
                                report_client_disconnect(request)
                                break
                            yield chunk
                    except asyncio.CancelledError as exc:
                        report_client_disconnect(request, exc)
                        raise
                    except Exception as exc:
                        report_stream_failure(request, exc, FailureOrigin.PROTOCOL)
                        logger.exception("Anthropic Token Router stream failed")
                        yield {
                            "error": {
                                "type": self._anthropic_error_type(500, exc),
                                "message": str(exc),
                            }
                        }
                    finally:
                        await release_resources()

                anthropic_events = anthropic_stream_events(
                    router_chunks(),
                    model_uid,
                    request_id,
                    include_thinking=canonical.thinking_enabled is True,
                    stop_sequences=canonical.stop or [],
                )
                return EventSourceResponse(
                    observe_stream(
                        anthropic_events,
                        get_stream_outcome_reporter(request),
                        failure_origin=FailureOrigin.PROTOCOL,
                    ),
                    ping=XINFERENCE_SSE_PING_ATTEMPTS_SECONDS,
                    headers=router_response_headers,
                    background=BackgroundTask(release_resources),
                )

            try:
                raw_response = await upstream_response.aread()
                openai_response = json.loads(raw_response)
                return JSONResponse(
                    content=openai_to_anthropic(
                        openai_response,
                        model_uid,
                        include_thinking=canonical.thinking_enabled is True,
                        stop_sequences=canonical.stop or [],
                    ),
                    headers=router_response_headers,
                )
            except (json.JSONDecodeError, AnthropicProtocolError) as exc:
                logger.error("Invalid Token Router response: %s", exc)
                return self._anthropic_error(502, str(exc), request_id)
            finally:
                await upstream_response.aclose()

        try:
            model = await require_model(
                self._get_supervisor_ref, model_uid, self._report_error_event
            )
            is_vllm_backend = (
                hasattr(model, "is_vllm_backend") and await model.is_vllm_backend()
            )
        except HTTPException as exc:
            return self._anthropic_error(
                exc.status_code, str(exc.detail), request_id, headers=exc.headers
            )
        except Exception as exc:
            logger.error(exc, exc_info=True)
            return self._anthropic_error(500, str(exc), request_id)

        kwargs = {
            key: value
            for key, value in openai_body.items()
            if key not in {"model", "messages"}
        }
        raw_kwargs = dict(kwargs)
        messages = canonical.messages
        response_headers = {"request-id": request_id}
        input_tokens = 0
        if hasattr(model, "prepare_anthropic_request"):
            try:
                prepared = await model.prepare_anthropic_request(messages, kwargs)
                if isinstance(prepared, (str, bytes)):
                    prepared = json.loads(prepared)
                messages = prepared["messages"]
                kwargs = prepared["generate_config"]
                raw_kwargs = dict(kwargs)
                input_tokens = prepared["input_tokens"]
                if canonical.cache_control is not None:
                    response_headers["x-xinference-cache-control"] = (
                        "breakpoint"
                        if prepared.get("cache_control_applied")
                        else "automatic"
                    )
            except NotImplementedError as exc:
                return self._anthropic_error(501, str(exc), request_id)
            except ValueError as exc:
                return self._anthropic_error(400, str(exc), request_id)
            except Exception as exc:
                logger.exception("Anthropic backend preparation failed")
                status_code = 429 if "Rate limit reached" in str(exc) else 500
                return self._anthropic_error(status_code, str(exc), request_id)
        else:
            kwargs.pop("_anthropic_content_metrics", None)
            kwargs.pop("_anthropic_requested_features", None)
            raw_kwargs = dict(kwargs)

        model_call_kwargs: Dict[str, Any] = {"raw_params": raw_kwargs}
        # A caller's correlation ID may be reused across concurrent requests.
        model_request_id = uuid.uuid4().hex
        if is_vllm_backend:
            model_call_kwargs["request_id"] = model_request_id
            logger.debug(
                "Anthropic request: request-id=%s engine_request_id=%s model=%s",
                request_id,
                model_request_id,
                model_uid,
            )

        if canonical.stream:
            iterator: Optional[AsyncIterator[Any]] = None
            abort_controller = (
                _SSERequestAbortController(
                    model, model_request_id, request, "chat", abort_timeout=5
                )
                if is_vllm_backend
                else None
            )
            protocol_chunks: Optional[AsyncIterator[Any]] = None
            completed = False
            stream_cleanup_task: Optional[asyncio.Task[None]] = None

            # Startup may acquire a stream that the response never consumes.
            async def release_stream_resources() -> None:
                nonlocal stream_cleanup_task
                if iterator is None or abort_controller is None:
                    return
                with anyio.CancelScope(shield=True):
                    if stream_cleanup_task is None:
                        stream_cleanup_task = asyncio.create_task(
                            _cleanup_model_stream(
                                abort_controller=abort_controller,
                                interrupted=not completed,
                                observed_iterator=(
                                    protocol_chunks
                                    if protocol_chunks is not None
                                    else iterator
                                ),
                                model_iterator=iterator,
                                model=model,
                                request_id=model_request_id,
                                kind="chat",
                            )
                        )
                    await asyncio.shield(stream_cleanup_task)

            async def physical_chunks() -> AsyncIterator[Dict[str, Any]]:
                nonlocal protocol_chunks, completed
                if abort_controller is not None:
                    abort_controller.bind_stream_task()
                reporter = get_stream_outcome_reporter(request)
                try:
                    assert iterator is not None
                    model_chunks = observe_stream(
                        iterator,
                        reporter,
                        failure_origin=FailureOrigin.MODEL_GENERATOR,
                    )
                    chunks = observe_stream(
                        self._iter_model_openai_chunks(model_chunks),
                        reporter,
                        failure_origin=FailureOrigin.PROTOCOL,
                    )
                    protocol_chunks = chunks
                    async for chunk in chunks:
                        if abort_controller is None and await request.is_disconnected():
                            report_client_disconnect(request)
                            break
                        yield chunk
                    completed = True
                except asyncio.CancelledError as exc:
                    report_client_disconnect(request, exc)
                    raise
                except Exception as exc:
                    report_stream_failure(request, exc, FailureOrigin.MODEL_GENERATOR)
                    exc = await self._get_model_last_error(model.uid, exc)
                    logger.exception("Anthropic physical model stream failed")
                    await self._report_error_event(model_uid, str(exc))
                    yield {
                        "error": {
                            "type": self._anthropic_error_type(500, exc),
                            "message": str(exc),
                        }
                    }
                finally:
                    if abort_controller is not None:
                        await release_stream_resources()
                    elif iterator is not None:
                        from xoscar.api import IteratorWrapper

                        if (
                            inspect.isasyncgen(iterator)
                            or inspect.isgenerator(iterator)
                            or isinstance(iterator, IteratorWrapper)
                        ):
                            await _release_model_stream(model, iterator)

            anthropic_events = anthropic_stream_events(
                physical_chunks(),
                model_uid,
                request_id,
                input_tokens=input_tokens,
                include_thinking=canonical.thinking_enabled is True,
                stop_sequences=canonical.stop or [],
            )
            body_iterator = observe_stream(
                anthropic_events,
                get_stream_outcome_reporter(request),
                failure_origin=FailureOrigin.PROTOCOL,
            )
            response = EventSourceResponse(
                body_iterator,
                ping=XINFERENCE_SSE_PING_ATTEMPTS_SECONDS,
                headers=response_headers,
                background=(
                    BackgroundTask(release_stream_resources)
                    if abort_controller is not None
                    else None
                ),
            )
            if abort_controller is not None:
                abort_controller.bind_body_iterator(body_iterator)

                async def handle_disconnect(message: Dict[str, Any]) -> None:
                    try:
                        await abort_controller.handle_disconnect(message)
                    finally:
                        await release_stream_resources()

                _observe_sse_disconnect(response, handle_disconnect)

            async def start_streaming_response() -> Response:
                nonlocal iterator
                iterator = await model.chat(messages, kwargs, **model_call_kwargs)
                return response

            # Commit SSE headers only after model admission/startup succeeds.
            stream_transferred = False
            try:
                if is_vllm_backend:
                    result = await _run_non_streaming_request(
                        request,
                        model,
                        model_request_id,
                        "chat",
                        start_streaming_response(),
                        abort_controller=abort_controller,
                    )
                    stream_transferred = result is response
                    return result
                return await start_streaming_response()
            except NotImplementedError as exc:
                return self._anthropic_error(501, str(exc), request_id)
            except Exception as exc:
                exc = await self._get_model_last_error(model.uid, exc)
                logger.error(exc, exc_info=True)
                await self._report_error_event(model_uid, str(exc))
                status_code = 429 if "Rate limit reached" in str(exc) else 500
                return self._anthropic_error(status_code, str(exc), request_id, exc=exc)
            finally:
                if is_vllm_backend and not stream_transferred:
                    await release_stream_resources()

        async def generate_message_response() -> Response:
            data = await model.chat(messages, kwargs, **model_call_kwargs)
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            openai_response = json.loads(data) if isinstance(data, str) else data
            return JSONResponse(
                content=openai_to_anthropic(
                    openai_response,
                    model_uid,
                    include_thinking=canonical.thinking_enabled is True,
                    stop_sequences=canonical.stop or [],
                ),
                headers=response_headers,
            )

        try:
            if is_vllm_backend:
                return await _run_non_streaming_request(
                    request,
                    model,
                    model_request_id,
                    "chat",
                    generate_message_response(),
                )
            return await generate_message_response()
        except AnthropicProtocolError as exc:
            return self._anthropic_error(
                exc.status_code,
                exc.message,
                request_id,
                error_type=exc.error_type,
            )
        except NotImplementedError as exc:
            return self._anthropic_error(501, str(exc), request_id)
        except Exception as exc:
            exc = await self._get_model_last_error(model.uid, exc)
            logger.error(exc, exc_info=True)
            await self._report_error_event(model_uid, str(exc))
            status_code = 429 if "Rate limit reached" in str(exc) else 500
            return self._anthropic_error(status_code, str(exc), request_id, exc=exc)

    @staticmethod
    def _responses_error(
        status_code: int,
        message: str,
        *,
        param: Optional[str] = None,
        code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> JSONResponse:
        error_type = {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            429: "rate_limit_error",
        }.get(status_code, "server_error")
        return JSONResponse(
            status_code=status_code,
            headers=headers,
            content=responses_error(message, error_type, param, code),
        )

    @classmethod
    def _responses_model_error(cls, exc: Exception) -> JSONResponse:
        message = str(exc)
        code = responses_error_code(message, exc)
        # A 400 keeps clients from retrying a request that can never fit.
        if code == "context_length_exceeded":
            return cls._responses_error(400, message, code=code)
        status_code = 429 if "Rate limit reached" in message else 500
        return cls._responses_error(status_code, message, code=code)

    async def create_response(self, request: Request) -> Response:
        try:
            raw_body = await request.json()
        except Exception:
            return self._responses_error(400, "Request body must be valid JSON")
        try:
            req = parse_responses_request(raw_body)
        except ResponsesProtocolError as exc:
            return self._responses_error(
                exc.status_code, exc.message, param=exc.param, code=exc.code
            )
        except (AttributeError, KeyError, TypeError) as exc:
            return self._responses_error(400, f"Invalid request: {exc}")

        model_uid = req.model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("llm")
        try:
            self._check_model_access(request, model_uid, "LLM")
            self._apply_cache_salt(request, raw_body, req.chat_body)
        except HTTPException as exc:
            return self._responses_error(
                exc.status_code, str(exc.detail), headers=exc.headers
            )

        try:
            supervisor_ref = await self._get_supervisor_ref()
            token_router_runtime = await supervisor_ref.resolve_token_router_runtime(
                model_uid
            )
        except Exception:
            logger.exception("Failed to resolve Responses model target: %s", model_uid)
            return self._responses_error(500, "Failed to resolve the requested model")
        if token_router_runtime is not None:
            return await self._create_response_via_token_router(
                request, req, token_router_runtime
            )

        try:
            model = await require_model(
                self._get_supervisor_ref, model_uid, self._report_error_event
            )
        except HTTPException as exc:
            return self._responses_error(
                exc.status_code, str(exc.detail), headers=exc.headers
            )
        except Exception as exc:
            logger.error(exc, exc_info=True)
            return self._responses_error(500, str(exc))

        messages = req.chat_body["messages"]
        kwargs = {
            key: value
            for key, value in req.chat_body.items()
            if key not in {"model", "messages"}
        }
        raw_kwargs = dict(kwargs)

        if not req.stream:
            try:
                data = await model.chat(messages, kwargs, raw_params=raw_kwargs)
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                chat = json.loads(data) if isinstance(data, str) else data
                return JSONResponse(content=chat_to_response(chat, req))
            except Exception as exc:
                exc = await self._get_model_last_error(model.uid, exc)
                logger.error(exc, exc_info=True)
                await self._report_error_event(model_uid, str(exc))
                return self._responses_model_error(exc)

        iterator = None
        try:
            iterator = await model.chat(messages, kwargs, raw_params=raw_kwargs)
        except Exception as exc:
            exc = await self._get_model_last_error(model.uid, exc)
            logger.error(exc, exc_info=True)
            await self._report_error_event(model_uid, str(exc))
            return self._responses_model_error(exc)

        async def release_serve_count() -> None:
            from xoscar.api import IteratorWrapper

            if (
                inspect.isasyncgen(iterator)
                or inspect.isgenerator(iterator)
                or isinstance(iterator, IteratorWrapper)
            ):
                await _release_model_stream(model, iterator)

        cleanup_task: asyncio.Task[None] | None = None

        # Also run as the response's background task: a client that disconnects
        # before the body starts never enters the generator's finally block.
        async def release_resources() -> None:
            nonlocal cleanup_task
            if cleanup_task is None:
                cleanup_task = asyncio.create_task(release_serve_count())
            await asyncio.shield(cleanup_task)

        async def physical_chunks() -> AsyncIterator[Dict[str, Any]]:
            reporter = get_stream_outcome_reporter(request)
            model_chunks = observe_stream(
                iterator, reporter, failure_origin=FailureOrigin.MODEL_GENERATOR
            )
            protocol_chunks = observe_stream(
                self._iter_model_openai_chunks(model_chunks),
                reporter,
                failure_origin=FailureOrigin.PROTOCOL,
            )
            try:
                async for chunk in protocol_chunks:
                    if await request.is_disconnected():
                        report_client_disconnect(request)
                        break
                    yield chunk
            except asyncio.CancelledError as exc:
                report_client_disconnect(request, exc)
                raise
            except Exception as exc:
                report_stream_failure(request, exc, FailureOrigin.MODEL_GENERATOR)
                exc = await self._get_model_last_error(model.uid, exc)
                logger.exception("Responses physical model stream failed")
                await self._report_error_event(model_uid, str(exc))
                yield {"error": exc}
            finally:
                await release_resources()

        return EventSourceResponse(
            observe_stream(
                responses_stream_events(physical_chunks(), req),
                get_stream_outcome_reporter(request),
                failure_origin=FailureOrigin.PROTOCOL,
            ),
            ping=XINFERENCE_SSE_PING_ATTEMPTS_SECONDS,
            background=BackgroundTask(release_resources),
        )

    async def _create_response_via_token_router(
        self,
        request: Request,
        req: ResponsesRequest,
        runtime: Dict[str, Any],
    ) -> Response:
        if not runtime.get("available"):
            return self._responses_error(
                503,
                f"No ready Token Router runtime is available for virtual model {req.model}",
                headers={"Retry-After": "1"},
            )
        try:
            upstream_response, _ = await self._open_token_router_chat_completion(
                request, req.chat_body, runtime, forward_external_credential=False
            )
        except httpx.HTTPError:
            logger.exception(
                "Responses Token Router connection failed: virtual_model_uid=%s",
                req.model,
            )
            return self._responses_error(
                502,
                "Token Router runtime is temporarily unavailable",
                headers={"Retry-After": "1"},
            )

        if upstream_response.status_code >= 400:
            retry_after = upstream_response.headers.get("retry-after")
            try:
                raw_error = await upstream_response.aread()
                error_payload = json.loads(raw_error) if raw_error else {}
            except Exception:
                error_payload = {}
            finally:
                await upstream_response.aclose()
            error = error_payload.get("error") or {}
            message = error.get("message") if isinstance(error, dict) else str(error)
            return self._responses_error(
                upstream_response.status_code,
                message or "Token Router request failed",
                headers={"Retry-After": retry_after} if retry_after else None,
            )

        if not req.stream:
            try:
                chat = json.loads(await upstream_response.aread())
                return JSONResponse(content=chat_to_response(chat, req))
            except json.JSONDecodeError as exc:
                logger.error("Invalid Token Router response: %s", exc)
                return self._responses_error(502, str(exc))
            finally:
                await upstream_response.aclose()

        cleanup_task: asyncio.Task[None] | None = None

        async def release_resources() -> None:
            nonlocal cleanup_task
            if cleanup_task is None:
                cleanup_task = asyncio.create_task(upstream_response.aclose())
            await asyncio.shield(cleanup_task)

        async def router_chunks() -> AsyncIterator[Dict[str, Any]]:
            reporter = get_stream_outcome_reporter(request)
            upstream_chunks = observe_stream(
                upstream_response.aiter_raw(),
                reporter,
                failure_origin=FailureOrigin.UPSTREAM,
            )
            protocol_chunks = observe_stream(
                self._iter_openai_sse_bytes(upstream_chunks),
                reporter,
                failure_origin=FailureOrigin.PROTOCOL,
            )
            try:
                async for chunk in protocol_chunks:
                    if await request.is_disconnected():
                        report_client_disconnect(request)
                        break
                    yield chunk
            except asyncio.CancelledError as exc:
                report_client_disconnect(request, exc)
                raise
            except Exception as exc:
                report_stream_failure(request, exc, FailureOrigin.PROTOCOL)
                logger.exception("Responses Token Router stream failed")
                yield {"error": exc}
            finally:
                await release_resources()

        return EventSourceResponse(
            observe_stream(
                responses_stream_events(router_chunks(), req),
                get_stream_outcome_reporter(request),
                failure_origin=FailureOrigin.PROTOCOL,
            ),
            ping=XINFERENCE_SSE_PING_ATTEMPTS_SECONDS,
            background=BackgroundTask(release_resources),
        )

    async def create_embedding(self, request: Request) -> Response:
        try:
            payload = await request.json()
            body = CreateEmbeddingRequest.parse_obj(payload)
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        model_uid = body.model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("embedding")
        self._check_model_access(request, model_uid, "embedding")
        exclude = {
            "model",
            "input",
            "user",
            "encoding_format",
        }
        kwargs = {key: value for key, value in payload.items() if key not in exclude}

        model = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        try:
            kwargs["model_uid"] = model_uid
            embedding = await model.create_embedding(body.input, **kwargs)
            if payload.get("encoding_format") == "base64":
                embedding = _encode_embeddings_base64(embedding)
            return Response(embedding, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def convert_ids_to_tokens(self, request: Request) -> Response:
        try:
            payload = await request.json()
            body = CreateEmbeddingRequest.parse_obj(payload)
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        model_uid = body.model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("embedding")
        self._check_model_access(request, model_uid, "embedding")
        exclude = {
            "model",
            "input",
            "user",
        }
        kwargs = {key: value for key, value in payload.items() if key not in exclude}

        model = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        try:
            decoded_texts = await model.convert_ids_to_tokens(body.input, **kwargs)
            return Response(decoded_texts, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def rerank(self, request: Request) -> Response:
        try:
            payload = await request.json()
            body = RerankRequest.parse_obj(payload)
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        model_uid = body.model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("rerank")
        self._check_model_access(request, model_uid, "rerank")
        parsed_kwargs = _parse_kwargs(body.kwargs)

        model = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        try:
            scores = await model.rerank(
                body.documents,
                body.query,
                top_n=body.top_n,
                max_chunks_per_doc=body.max_chunks_per_doc,
                return_documents=body.return_documents,
                return_len=body.return_len,
                **parsed_kwargs,
            )
            return Response(scores, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_audio_embedding(
        self,
        request: Request,
        model: str = Form(...),
        file: UploadFile = File(media_type="application/octet-stream"),
    ) -> Response:
        model_uid = model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("audio")
        self._check_model_access(request, model_uid, "audio")
        model_ref = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        try:
            embedding = await model_ref.create_audio_embedding(
                audio=await file.read(), model_uid=model_uid
            )
            return Response(content=embedding, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_transcriptions(
        self,
        request: Request,
        model: str = Form(...),
        file: UploadFile = File(media_type="application/octet-stream"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(0),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        form = await request.form()
        timestamp_granularities = form.getlist("timestamp_granularities[]") or None
        model_uid = model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("audio")
        self._check_model_access(request, model_uid, "audio")
        parsed_kwargs = _parse_kwargs(kwargs)
        model_ref = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        try:
            transcription = await model_ref.transcriptions(
                audio=await file.read(),
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
                **parsed_kwargs,
            )
            return Response(content=transcription, media_type="application/json")
        except InvalidAudioInputError as e:
            logger.warning("Invalid transcription audio for model %s: %s", model_uid, e)
            raise HTTPException(status_code=400, detail=e.client_message) from e
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_translations(
        self,
        request: Request,
        model: str = Form(...),
        file: UploadFile = File(media_type="application/octet-stream"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(0),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        form = await request.form()
        timestamp_granularities = form.getlist("timestamp_granularities[]") or None
        model_uid = model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("audio")
        self._check_model_access(request, model_uid, "audio")
        parsed_kwargs = _parse_kwargs(kwargs)
        model_ref = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        try:
            translation = await model_ref.translations(
                audio=await file.read(),
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
                **parsed_kwargs,
            )
            return Response(content=translation, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_speech(
        self,
        request: Request,
        prompt_speech: Optional[UploadFile] = File(
            None, media_type="application/octet-stream"
        ),
        prompt_latent: Optional[UploadFile] = File(
            None, media_type="application/octet-stream"
        ),
    ) -> Response:
        try:
            if prompt_speech or prompt_latent:
                f = await request.form()
            else:
                f = await request.json()
            body = SpeechRequest.parse_obj(f)
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        model_uid = body.model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("audio")
        self._check_model_access(request, model_uid, "audio")
        parsed_kwargs = _parse_kwargs(body.kwargs)
        model = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        try:
            if prompt_speech is not None:
                parsed_kwargs["prompt_speech"] = await prompt_speech.read()
            if prompt_latent is not None:
                parsed_kwargs["prompt_latent"] = await prompt_latent.read()
            out = await model.speech(
                input=body.input,
                voice=body.voice,
                response_format=body.response_format,
                speed=body.speed,
                stream=body.stream,
                **parsed_kwargs,
            )
            if body.stream:

                async def stream_results():
                    try:
                        async for item in observe_stream(
                            out,
                            get_stream_outcome_reporter(request),
                            failure_origin=FailureOrigin.MODEL_GENERATOR,
                        ):
                            yield item
                    finally:
                        await _release_model_stream(model, out)

                return EventSourceResponse(
                    media_type="application/octet-stream",
                    content=stream_results(),
                    ping=XINFERENCE_SSE_PING_ATTEMPTS_SECONDS,
                )
            else:
                return Response(
                    media_type=_audio_response_media_type(body.response_format),
                    content=out,
                )
        except (InvalidAudioInputError, InvalidModelRequestError) as e:
            logger.warning("Invalid speech request for model %s: %s", model_uid, e)
            raise HTTPException(status_code=400, detail=e.client_message) from e
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_images(self, request: Request) -> Response:
        try:
            body = TextToImageRequest.parse_obj(await request.json())
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        model_uid = body.model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("image")
        self._check_model_access(request, model_uid, "image")
        kwargs = _parse_kwargs(body.kwargs or None)
        model = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        request_id = None
        try:
            request_id = kwargs.get("request_id")
            self._add_running_task(request_id)
            image_list = await model.text_to_image(
                prompt=body.prompt,
                n=body.n,
                size=body.size,
                response_format=body.response_format,
                **kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def _parse_sdapi_request(schema, request, query=False):
        try:
            return schema.parse_obj(
                request.query_params if query else await request.json()
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    async def sdapi_options(self, request: Request) -> Response:
        body = await self._parse_sdapi_request(SDAPIOptionsRequest, request)
        model_uid = body.sd_model_checkpoint
        if model_uid:
            self._check_model_access(request, model_uid, "image")
        self._set_trace_model(model_uid)
        self._set_trace_model_type("image")

        try:
            if not model_uid:
                raise ValueError("Unknown model")
            await actor_call(
                await self._get_supervisor_ref(),
                "get_model",
                model_uid,
                _rpc_correlation_id=get_model_request_id(request),
            )
            return Response()
        except ModelNotReadyError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model is loading, please retry later: {e}",
                headers={"Retry-After": "30"},
            )
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_sd_models(self, request: Request) -> Response:
        try:
            models = await (await self._get_supervisor_ref()).list_models()
            sd_models = []
            for model_name, info in models.items():
                if info["model_type"] != "image":
                    continue
                try:
                    self._check_model_access(request, model_name, "image")
                except HTTPException as exc:
                    if exc.status_code == 403:
                        continue
                    raise
                sd_models.append({"model_name": model_name, "config": None})
            return JSONResponse(content=sd_models)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_samplers(self, request: Request) -> Response:
        try:
            from ..model.image.stable_diffusion.core import SAMPLING_METHODS

            samplers = [
                {"name": sample_method, "alias": [], "options": {}}
                for sample_method in SAMPLING_METHODS
            ]
            return JSONResponse(content=samplers)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_txt2img(self, request: Request) -> Response:
        body = await self._parse_sdapi_request(SDAPITxt2imgRequst, request)
        model_uid = body.model or body.override_settings.get("sd_model_checkpoint")
        self._check_model_access(request, model_uid, "image")
        self._set_trace_model(model_uid)
        self._set_trace_model_type("image")

        model = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        try:
            kwargs = body.dict(exclude_none=True)
            kwargs.update(json.loads(body.kwargs) if body.kwargs else {})
            image_list = await model.txt2img(
                **kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except asyncio.CancelledError:
            raise HTTPException(
                status_code=409,
                detail=f"The request has been cancelled: {kwargs.get('request_id')}",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_img2img(self, request: Request) -> Response:
        body = await self._parse_sdapi_request(SDAPIImg2imgRequst, request)
        model_uid = body.model or body.override_settings.get("sd_model_checkpoint")
        self._check_model_access(request, model_uid, "image")
        self._set_trace_model(model_uid)
        self._set_trace_model_type("image")

        model = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        try:
            kwargs = body.dict(exclude_none=True)
            kwargs.update(json.loads(body.kwargs) if body.kwargs else {})
            image_list = await model.img2img(
                **kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except asyncio.CancelledError:
            raise HTTPException(
                status_code=409,
                detail=f"The request has been cancelled: {kwargs.get('request_id')}",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_variations(
        self,
        request: Request,
        model: str = Form(...),
        image: List[UploadFile] = File(media_type="application/octet-stream"),
        prompt: Optional[Union[str, List[str]]] = Form(None),
        negative_prompt: Optional[Union[str, List[str]]] = Form(None),
        n: Optional[int] = Form(1),
        response_format: Optional[str] = Form("url"),
        size: Optional[str] = Form(None),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        model_uid = model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("image")
        self._check_model_access(request, model_uid, "image")
        parsed_kwargs = _parse_kwargs(kwargs)
        model_ref = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        request_id = None
        try:
            request_id = parsed_kwargs.get("request_id")
            self._add_running_task(request_id)

            # Handle single image or multiple images
            if len(image) == 1:
                # Single image
                image_data = Image.open(image[0].file)
            else:
                # Multiple images - convert to list of PIL Images
                image_data = [Image.open(img.file) for img in image]

            image_list = await model_ref.image_to_image(
                image=image_data,
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                size=size,
                response_format=response_format,
                **parsed_kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_inpainting(
        self,
        request: Request,
        model: str = Form(...),
        image: UploadFile = File(media_type="application/octet-stream"),
        mask_image: UploadFile = File(media_type="application/octet-stream"),
        prompt: Optional[Union[str, List[str]]] = Form(None),
        negative_prompt: Optional[Union[str, List[str]]] = Form(None),
        n: Optional[int] = Form(1),
        response_format: Optional[str] = Form("url"),
        size: Optional[str] = Form(None),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        model_uid = model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("image")
        self._check_model_access(request, model_uid, "image")
        parsed_kwargs = _parse_kwargs(kwargs)
        model_ref = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        request_id = None
        try:
            request_id = parsed_kwargs.get("request_id")
            self._add_running_task(request_id)
            im = Image.open(image.file)
            mask_im = Image.open(mask_image.file)
            if not size:
                w, h = im.size
                size = f"{w}*{h}"
            image_list = await model_ref.inpainting(
                image=im,
                mask_image=mask_im,
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                size=size,
                response_format=response_format,
                **parsed_kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_ocr(
        self,
        request: Request,
        model: str = Form(...),
        image: UploadFile = File(media_type="application/octet-stream"),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        model_uid = model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("image")
        self._check_model_access(request, model_uid, "image")
        parsed_kwargs = _parse_kwargs(kwargs)
        model_ref = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        request_id = None
        try:
            request_id = parsed_kwargs.get("request_id")
            self._add_running_task(request_id)
            head = image.file.read(len(PDF_MAGIC))
            image.file.seek(0)
            is_pdf = is_pdf_upload(image.content_type, head)
            # Whole-document tasks parse the PDF themselves and need the
            # original bytes plus every page at once, so they bypass the
            # per-page rasterizing path below.
            requested_task = parsed_kwargs.get("task")
            if (
                isinstance(requested_task, str)
                and requested_task in WHOLE_DOCUMENT_OCR_TASKS
            ):
                task = requested_task
                if not is_pdf:
                    raise HTTPException(
                        status_code=400,
                        detail=f"OCR task {task!r} requires a PDF upload",
                    )
                for unsupported in ("pages", "dpi"):
                    if unsupported in parsed_kwargs:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"`{unsupported}` is not supported for OCR task "
                                f"{task!r}, which parses the whole document"
                            ),
                        )
                from ..model.image.ocr.deepdoc import MAX_PARSE_ZOOMIN, parse_zoomin

                data = image.file.read()
                try:
                    # The model validates this too; it is needed here to
                    # budget the raster before the bytes are handed over.
                    zoomin = parse_zoomin(parsed_kwargs.get("zoomin"))
                    await asyncio.to_thread(
                        validate_pdf_for_parse, data, zoomin, MAX_PARSE_ZOOMIN
                    )
                except ValueError as ve:
                    raise HTTPException(status_code=400, detail=str(ve))
                try:
                    result = await model_ref.ocr(
                        image=data,
                        **parsed_kwargs,
                    )
                except ValueError as ve:
                    # Bad `zoomin`/`image_scope` and friends are user input,
                    # not a server fault.
                    raise HTTPException(status_code=400, detail=str(ve))
                return Response(content=result, media_type="application/json")
            if is_pdf:
                pages = parsed_kwargs.pop("pages", None)
                dpi = parsed_kwargs.pop("dpi", DEFAULT_PDF_OCR_DPI)
                try:
                    page_iter = await asyncio.to_thread(
                        rasterize_pdf, image.file.read(), pages=pages, dpi=dpi
                    )
                except ValueError as ve:
                    raise HTTPException(status_code=400, detail=str(ve))
                # Pages are rendered lazily, one at a time, so peak memory
                # stays at a single page regardless of document size.
                page_results = []
                try:
                    while True:
                        item = await asyncio.to_thread(next, page_iter, None)
                        if item is None:
                            break
                        page_number, page_image = item
                        try:
                            result = await model_ref.ocr(
                                image=page_image,
                                **parsed_kwargs,
                            )
                        finally:
                            page_image.close()
                        page_results.append((page_number, json.loads(result)))
                finally:
                    page_iter.close()
                body = merge_ocr_page_results(page_results)
                return Response(content=body, media_type="application/json")
            im = Image.open(image.file)
            result = await model_ref.ocr(
                image=im,
                **parsed_kwargs,
            )
            # ModelActor.ocr always serializes the model's return value to
            # JSON bytes (_call_wrapper_json), and both REST clients parse
            # the body with response.json() — declaring text/plain here
            # breaks aiohttp's content-type check on the async client.
            return Response(content=result, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except HTTPException:
            raise
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_doc_analyze(
        self,
        request: Request,
        model: str = Form(...),
        file: UploadFile = File(media_type="application/octet-stream"),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        if not file.filename or file.size == 0:
            raise HTTPException(status_code=400, detail="File can't be empty")
        model_uid = model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("image")
        self._check_model_access(request, model_uid, "image")
        parsed_kwargs = _parse_kwargs(kwargs)
        model_ref = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        request_id = None
        try:
            request_id = parsed_kwargs.get("request_id")
            self._add_running_task(request_id)
            file_bytes = await file.read()
            if not file_bytes:
                raise HTTPException(status_code=400, detail="File can't be empty")
            file_name = file.filename
            data = await model_ref.docanalyze(
                file_bytes=file_bytes,
                file_name=file_name,
                **parsed_kwargs,
            )
            return Response(content=data, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except HTTPException:
            raise
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_image_edits(
        self,
        request: Request,
        prompt: str = Form(...),
        images: Optional[List[UploadFile]] = File(
            None, media_type="application/octet-stream"
        ),
        mask: Optional[UploadFile] = File(None, media_type="application/octet-stream"),
        model: Optional[str] = Form(None),
        n: Optional[int] = Form(1),
        size: Optional[str] = Form("original"),
        response_format: Optional[str] = Form("url"),
        stream: Optional[bool] = Form(False),
    ) -> Response:
        """OpenAI-compatible image edit endpoint.

        Accepts multiple image files via:
            -F "image=@image1.jpeg" -F "image=@image2.jpeg"
        The first image is used as the primary input; additional images
        are passed as reference images to the model.
        """
        import io

        # If FastAPI didn't bind any files (e.g. client used "image[]" key),
        # fall back to manual form parsing.
        image_files: List[UploadFile] = images or []
        if not image_files:
            form = await request.form()
            image_files = (
                form.getlist("images[]")
                or form.getlist("images")
                or form.getlist("image[]")
                or form.getlist("image")
            )

        if not image_files:
            raise HTTPException(
                status_code=400, detail="At least one image file is required"
            )

        if response_format not in ("url", "b64_json"):
            raise HTTPException(
                status_code=400, detail="response_format must be 'url' or 'b64_json'"
            )

        # Resolve model when the caller didn't specify one.
        if not model:
            try:
                models = await (await self._get_supervisor_ref()).list_models()
                image_models = [
                    name
                    for name, info in models.items()
                    if info.get("model_type") == "image"
                    and (
                        "image2image" in info.get("model_ability", [])
                        or "inpainting" in info.get("model_ability", [])
                    )
                ]
                if not image_models:
                    raise HTTPException(
                        status_code=400, detail="No available image models found"
                    )
                model = image_models[0]
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get available models: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500, detail="Failed to get available models"
                )

        assert model is not None
        model_uid = model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("image")
        self._check_model_access(request, model_uid, "image")
        model_ref = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        request_id = None
        try:
            self._add_running_task(request_id)

            description = await (await self._get_supervisor_ref()).describe_model(
                model_uid
            )
            preserve_alpha = description.get("model_name", "").lower() in {
                "qwen-image-2.1",
                "qwen-image-2.1-turbo",
                "ming-image-0.1-design",
                "ming-image-0.1-design-layer",
            }

            # Preserve transparent references for models with RGBA inputs.
            pil_images: list[Image.Image] = []
            for img_file in image_files:
                image_content = await img_file.read()
                pil_image = Image.open(io.BytesIO(image_content))

                if preserve_alpha:
                    pil_image = pil_image.convert("RGBA")
                elif pil_image.mode == "RGBA":
                    background = Image.new("RGB", pil_image.size, (255, 255, 255))
                    background.paste(pil_image, mask=pil_image.split()[3])
                    pil_image = background
                elif pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                pil_images.append(pil_image)

            primary_image = pil_images[0]
            reference_images = pil_images[1:] if len(pil_images) > 1 else []

            # Normalise the size parameter.
            if size and size != "original":
                model_size = size.replace("x", "*")
            else:
                model_size = ""

            model_params: dict[str, Any] = {
                "prompt": prompt,
                "n": n or 1,
                "size": model_size,
                "response_format": response_format,
                "denoising_strength": 0.75,
                "negative_prompt": " ",
            }
            if reference_images:
                model_params["reference_images"] = reference_images

            # DEBUG: Log the number of files and model params before processing
            logger.debug(
                f"Processing image edit with {len(image_files)} image(s), "
                f"model: {model_uid}, stream: {stream}, mask: {'yes' if mask else 'no'}, "
                f"model_params: {model_params}"
            )

            if stream:
                return EventSourceResponse(
                    self._stream_image_edit(
                        request,
                        model_ref,
                        primary_image,
                        reference_images,
                        mask,
                        model_params,
                    )
                )

            if mask:
                mask_content = await mask.read()
                mask_image = Image.open(io.BytesIO(mask_content))
                result = await model_ref.inpainting(
                    image=primary_image,
                    mask_image=mask_image,
                    **model_params,
                )
            else:
                result = await model_ref.image_to_image(
                    image=primary_image,
                    **model_params,
                )

            return Response(content=result, media_type="application/json")

        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id or 'unknown'}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def _stream_image_edit(
        self,
        request: Request,
        model_ref,
        primary_image: Image.Image,
        reference_images: list,
        mask: Optional[UploadFile],
        model_params: dict,
    ):
        """Stream image editing progress and results."""
        import io
        import json
        from datetime import datetime

        try:
            yield {
                "event": "start",
                "data": json.dumps(
                    {
                        "type": "image_edit_started",
                        "timestamp": datetime.now().isoformat(),
                        "prompt": model_params.get("prompt", ""),
                        "image_count": 1 + len(reference_images),
                    }
                ),
            }

            if mask:
                mask_content = await mask.read()
                mask_image = Image.open(io.BytesIO(mask_content))
                yield {
                    "event": "processing",
                    "data": json.dumps(
                        {
                            "type": "mask_loaded",
                            "timestamp": datetime.now().isoformat(),
                            "mask_size": mask_image.size,
                        }
                    ),
                }
                result = await model_ref.inpainting(
                    image=primary_image,
                    mask_image=mask_image,
                    **model_params,
                )
            else:
                yield {
                    "event": "processing",
                    "data": json.dumps(
                        {
                            "type": "starting_generation",
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                }
                result = await model_ref.image_to_image(
                    image=primary_image,
                    **model_params,
                )

            result_data = json.loads(result)
            yield {
                "event": "complete",
                "data": json.dumps(result_data),
            }

        except Exception as e:
            report_stream_failure(request, e, FailureOrigin.MODEL_GENERATOR)
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "type": "image_edit_error",
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                    }
                ),
            }

    async def create_flexible_infer(self, request: Request) -> Response:
        payload = await request.json()

        model_uid = payload.get("model")
        self._set_trace_model(model_uid)
        self._set_trace_model_type("flexible")
        self._check_model_access(request, model_uid, None)
        args = payload.get("args")

        exclude = {"model", "args"}
        kwargs = {key: value for key, value in payload.items() if key not in exclude}

        model = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        try:
            result = await model.infer(*args, **kwargs)
            return Response(result, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_videos(self, request: Request) -> Response:
        try:
            body = TextToVideoRequest.parse_obj(await request.json())
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        model_uid = body.model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("video")
        self._check_model_access(request, model_uid, "video")
        kwargs = _parse_kwargs(body.kwargs or None)
        model = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        request_id = None
        try:
            request_id = kwargs.get("request_id")
            self._add_running_task(request_id)
            video_list = await model.text_to_video(
                prompt=body.prompt,
                n=body.n,
                **kwargs,
            )
            return Response(content=video_list, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id or 'unknown'}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except InvalidModelRequestError as e:
            logger.warning("Invalid video request for model %s: %s", model_uid, e)
            raise HTTPException(status_code=400, detail=e.client_message) from e
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_videos_from_images(
        self,
        request: Request,
        model: str = Form(...),
        image: UploadFile = File(media_type="application/octet-stream"),
        video: Optional[UploadFile] = File(None, media_type="application/octet-stream"),
        prompt: Optional[Union[str, List[str]]] = Form(None),
        negative_prompt: Optional[Union[str, List[str]]] = Form(None),
        n: Optional[int] = Form(1),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        model_uid = model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("video")
        self._check_model_access(request, model_uid, "video")
        parsed_kwargs = _parse_kwargs(kwargs)
        model_ref = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        request_id = None
        try:
            if video is not None:
                parsed_kwargs["video"] = await video.read()
            request_id = parsed_kwargs.get("request_id")
            self._add_running_task(request_id)
            video_list = await model_ref.image_to_video(
                image=Image.open(image.file),
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                **parsed_kwargs,
            )
            return Response(content=video_list, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id or 'unknown'}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except InvalidModelRequestError as e:
            logger.warning("Invalid video request for model %s: %s", model_uid, e)
            raise HTTPException(status_code=400, detail=e.client_message) from e
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_videos_from_first_last_frame(
        self,
        request: Request,
        model: str = Form(...),
        first_frame: UploadFile = File(media_type="application/octet-stream"),
        last_frame: UploadFile = File(media_type="application/octet-stream"),
        prompt: Optional[Union[str, List[str]]] = Form(None),
        negative_prompt: Optional[Union[str, List[str]]] = Form(None),
        n: Optional[int] = Form(1),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        model_uid = model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("video")
        self._check_model_access(request, model_uid, "video")
        parsed_kwargs = _parse_kwargs(kwargs)
        model_ref = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )

        request_id = None
        try:
            request_id = parsed_kwargs.get("request_id")
            self._add_running_task(request_id)
            video_list = await model_ref.flf_to_video(
                first_frame=Image.open(first_frame.file),
                last_frame=Image.open(last_frame.file),
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                **parsed_kwargs,
            )
            return Response(content=video_list, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id or 'unknown'}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except InvalidModelRequestError as e:
            logger.warning("Invalid video request for model %s: %s", model_uid, e)
            raise HTTPException(status_code=400, detail=e.client_message) from e
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_world(self, request: Request) -> Response:
        try:
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > _MAX_WORLD_REQUEST_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="World generation request is too large",
                        )
                except ValueError:
                    raise HTTPException(
                        status_code=400, detail="Invalid Content-Length header"
                    ) from None

            raw_body = bytearray()
            async for chunk in request.stream():
                if len(raw_body) + len(chunk) > _MAX_WORLD_REQUEST_BYTES:
                    raise HTTPException(
                        status_code=413, detail="World generation request is too large"
                    )
                raw_body.extend(chunk)
            body = WorldGenerationRequest.parse_obj(json.loads(raw_body))
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")
        model_uid = body.model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("world")
        self._check_model_access(request, model_uid, "world")
        if body.image is not None and body.video is not None:
            raise HTTPException(
                status_code=400,
                detail="Only one of image and video may be provided",
            )
        for media_type, reference in (("image", body.image), ("video", body.video)):
            if reference is None:
                continue
            header, separator, _ = reference.partition(",")
            if (
                not separator
                or not header.startswith(f"data:{media_type}/")
                or ";base64" not in header
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"World generation {media_type} must be a base64 data URL. "
                        "Use the Xinference client to encode local files."
                    ),
                )

        model_ref = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )
        generation_config = dict(body.generation_config)
        model_kwargs = dict(body.extra_body)
        kwargs_request_id = model_kwargs.pop("request_id", None)
        config_request_id = generation_config.pop("request_id", None)
        request_id = kwargs_request_id or config_request_id or uuid.uuid4().hex
        try:
            self._add_running_task(request_id)
            result = await model_ref.world_generate(
                prompt=body.prompt,
                image=body.image,
                video=body.video,
                generation_config=generation_config,
                model_kwargs=model_kwargs,
                request_id=request_id,
            )
            return Response(content=result, media_type="application/json")
        except asyncio.CancelledError:
            if request_id:
                try:
                    await asyncio.shield(model_ref.abort_request(request_id))
                except Exception:
                    logger.exception(
                        "Failed to stop cancelled world request %s", request_id
                    )
            err_str = f"The request has been cancelled: {request_id or 'unknown'}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except ValueError as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    def _coerce_chat_template_kwargs(value: Any) -> dict:
        if not value:
            return {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {}
        if isinstance(value, dict):
            return dict(value)
        return {}

    @classmethod
    def _is_qwen38_description(cls, desc: dict) -> bool:
        return (
            desc.get("model_name") in cls.QWEN38_REASONING_MODEL_NAMES
            or desc.get("model_family") in cls.QWEN38_REASONING_MODEL_NAMES
        )

    @classmethod
    def _apply_qwen38_reasoning_effort(
        cls, raw_body: dict, raw_kwargs: dict, kwargs: dict
    ) -> None:
        has_top_level_effort = "reasoning_effort" in raw_body
        top_level_effort = raw_body.get("reasoning_effort")
        chat_template_kwargs = cls._coerce_chat_template_kwargs(
            raw_kwargs.get("chat_template_kwargs")
        )
        has_template_effort = "reasoning_effort" in chat_template_kwargs
        template_effort = chat_template_kwargs.get("reasoning_effort")

        if not has_top_level_effort and not has_template_effort:
            return

        if has_template_effort and template_effort not in cls.QWEN38_REASONING_EFFORTS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid chat_template_kwargs.reasoning_effort for qwen3.8. "
                    "Supported values are xhigh, medium, and low."
                ),
            )

        if not has_top_level_effort:
            return

        if top_level_effort not in cls.QWEN38_REASONING_EFFORTS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid reasoning_effort for qwen3.8. Supported values are "
                    "xhigh, medium, and low. Use chat_template_kwargs.enable_thinking "
                    "to disable thinking."
                ),
            )

        if has_template_effort and template_effort != top_level_effort:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Conflicting reasoning_effort values for qwen3.8. Provide either "
                    "top-level reasoning_effort or matching "
                    "chat_template_kwargs.reasoning_effort."
                ),
            )

        chat_template_kwargs["reasoning_effort"] = top_level_effort
        raw_kwargs.pop("reasoning_effort", None)
        kwargs.pop("reasoning_effort", None)
        raw_kwargs["chat_template_kwargs"] = chat_template_kwargs
        kwargs["chat_template_kwargs"] = chat_template_kwargs

    async def create_chat_completion(self, request: Request) -> Response:
        raw_body = await request.json()
        # Resolve/validate the raw salt after authorization, allowing provider overrides.
        body = CreateChatCompletion.parse_obj(
            {key: value for key, value in raw_body.items() if key != "cache_salt"}
        )
        raw_body = _normalize_token_router_chat_payload(raw_body)
        exclude = {
            "prompt",
            "model",
            "n",
            "messages",
            "logit_bias",
            "logit_bias_type",
            "user",
            "max_completion_tokens",
        }

        raw_kwargs = {k: v for k, v in raw_body.items() if k not in exclude}
        kwargs = body.dict(exclude_unset=True, exclude=exclude)
        request_id = str(kwargs.get("request_id") or uuid.uuid4().hex)

        enable_thinking = raw_body.get("enable_thinking")
        if enable_thinking is None:
            extra_body = raw_body.get("extra_body")
            if isinstance(extra_body, dict):
                enable_thinking = extra_body.get("enable_thinking")
        if isinstance(enable_thinking, bool):
            raw_kwargs.pop("enable_thinking", None)
            chat_template_kwargs = raw_kwargs.get("chat_template_kwargs") or {}
            if isinstance(chat_template_kwargs, str):
                try:
                    chat_template_kwargs = json.loads(chat_template_kwargs)
                except json.JSONDecodeError:
                    chat_template_kwargs = {}
            if not isinstance(chat_template_kwargs, dict):
                chat_template_kwargs = {}
            chat_template_kwargs = dict(chat_template_kwargs)
            chat_template_kwargs["enable_thinking"] = enable_thinking
            chat_template_kwargs["thinking"] = enable_thinking
            raw_kwargs["chat_template_kwargs"] = chat_template_kwargs
            kwargs["chat_template_kwargs"] = chat_template_kwargs

        # guided_decoding params
        kwargs.update(self.extract_guided_params(raw_body=raw_body))

        # TODO: Decide if this default value override is necessary #1061
        if body.max_tokens is None:
            kwargs["max_tokens"] = max_tokens_field.default

        if body.max_completion_tokens is not None:
            kwargs["max_tokens"] = body.max_completion_tokens

        if body.logit_bias is not None:
            raise HTTPException(status_code=501, detail="Not implemented")

        messages = body.messages and list(body.messages) or None

        if not messages:
            raise HTTPException(
                status_code=400, detail="Invalid input. Please specify the prompt."
            )

        has_tool_message = messages[-1].get("role") == "tool"
        model_uid = body.model
        self._set_trace_model(model_uid)
        self._set_trace_model_type("llm")
        self._check_model_access(request, model_uid, "LLM")
        self._apply_cache_salt(request, raw_body, kwargs, raw_kwargs)

        supervisor_ref = await self._get_supervisor_ref()
        token_router_runtime = None
        if XINFERENCE_TOKEN_ROUTER_ENABLED:
            token_router_runtime = await supervisor_ref.resolve_token_router_runtime(
                model_uid
            )
        if token_router_runtime is not None:
            if not token_router_runtime.get("available"):
                return JSONResponse(
                    content={
                        "error": {
                            "message": (
                                "No ready Token Router runtime is available for "
                                f"virtual model {model_uid}"
                            ),
                            "type": "router_unavailable",
                        }
                    },
                    status_code=503,
                    headers={"Retry-After": "1"},
                )
            return await self._proxy_token_router_chat_completion(
                request, raw_body, token_router_runtime
            )

        model = await require_model(
            self._get_supervisor_ref, model_uid, self._report_error_event
        )
        is_vllm_backend = await model.is_vllm_backend()
        model_call_kwargs: Dict[str, Any] = {"raw_params": raw_kwargs}
        if is_vllm_backend:
            kwargs.pop("request_id", None)
            raw_kwargs.pop("request_id", None)
            model_call_kwargs["request_id"] = request_id
        elif await model.is_transformers_batching_enabled():
            model_call_kwargs["request_id"] = request_id

        try:
            desc = await (await self._get_supervisor_ref()).describe_model(model_uid)
        except ModelNotReadyError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model is loading, please retry later: {e}",
                headers={"Retry-After": "30"},
            )
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        from ..model.llm.utils import (
            DEEPSEEK_TOOL_CALL_FAMILY,
            GEMMA_TOOL_CALL_FAMILY,
            GLM4_TOOL_CALL_FAMILY,
            GLM5_TOOL_CALL_FAMILY,
            KIMI_K3_TOOL_CALL_FAMILY,
            LLAMA3_TOOL_CALL_FAMILY,
            MINICPM5_TOOL_CALL_FAMILY,
            QWEN_TOOL_CALL_FAMILY,
        )

        if self._is_qwen38_description(desc):
            self._apply_qwen38_reasoning_effort(raw_body, raw_kwargs, kwargs)

        model_family = desc.get("model_family", "")
        RESTfulAPI._apply_multiple_choice_count(
            kwargs,
            getattr(body, "n", None),
            desc.get("model_engine"),
            is_vllm_backend,
        )

        total_call_family = (
            DEEPSEEK_TOOL_CALL_FAMILY
            | GEMMA_TOOL_CALL_FAMILY
            | GLM4_TOOL_CALL_FAMILY
            | LLAMA3_TOOL_CALL_FAMILY
            | QWEN_TOOL_CALL_FAMILY
            | GLM5_TOOL_CALL_FAMILY
            | KIMI_K3_TOOL_CALL_FAMILY
            | MINICPM5_TOOL_CALL_FAMILY
        )
        if model_family not in total_call_family:
            if body.tools:
                raise HTTPException(
                    status_code=400,
                    detail=f"Only {total_call_family} support tool calls",
                )
            if has_tool_message:
                raise HTTPException(
                    status_code=400,
                    detail=f"Only {total_call_family} support tool messages",
                )

        # Reject misplaced ``system`` messages before entering the worker for
        # models whose chat template requires system-first ordering (Qwen3
        # family: Ornith-1.0-35B / qwen3.5 / qwen3.6 / Nex-N2). Placed before
        # the stream/non-stream split so BOTH paths return a clean 400 instead
        # of the worker raising mid-render (non-stream 500 / stream 200+SSE).
        if desc.get("strict_system_first"):
            from ..model.llm.utils import MessageRoleOrderError, check_system_role_order

            try:
                check_system_role_order(messages)
            except MessageRoleOrderError as ve:
                raise HTTPException(status_code=400, detail=str(ve))

        limits = None
        if is_vllm_backend and any(
            self._count_multimodal_items(messages, item_type)
            for item_type in ("image", "video", "audio")
        ):
            try:
                limits = await model.get_limit_mm_per_prompt()
            except Exception:
                logger.debug(
                    "Cannot read multimodal limits; defer validation to the worker.",
                    exc_info=True,
                )
        for item_type, limit in (limits if isinstance(limits, dict) else {}).items():
            if type(limit) is not int or limit < 0:
                continue
            count = self._count_multimodal_items(messages, item_type)
            if count > limit:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"At most {limit} {item_type}(s) may be provided in one prompt; "
                        f"received {count}."
                    ),
                )

        if "skip_special_tokens" in raw_kwargs and is_vllm_backend:
            kwargs["skip_special_tokens"] = raw_kwargs["skip_special_tokens"]
        if body.stream:
            abort_controller = _SSERequestAbortController(
                model, request_id, request, "chat"
            )

            async def stream_results():
                abort_controller.bind_stream_task()
                iterator = None
                observed_iterator = None
                completed = False
                try:
                    iterator = await model.chat(
                        messages,
                        kwargs,
                        **model_call_kwargs,
                    )
                    observed_iterator = observe_stream(
                        iterator,
                        get_stream_outcome_reporter(request),
                        failure_origin=FailureOrigin.MODEL_GENERATOR,
                    )
                    async for item in observed_iterator:
                        yield item
                    completed = True
                    yield "[DONE]"
                # Note that asyncio.CancelledError does not inherit from Exception.
                # When the user uses ctrl+c to cancel the streaming chat, asyncio.CancelledError would be triggered.
                # See https://github.com/sysid/sse-starlette/blob/main/examples/example.py#L48
                except asyncio.CancelledError as exc:
                    report_client_disconnect(request, exc)
                    logger.info(
                        f"Disconnected from client (via refresh/close) {request.client} during chat."
                    )
                    await abort_controller.abort()
                    # See https://github.com/sysid/sse-starlette/blob/main/examples/error_handling.py#L13
                    # Use return to stop the generator from continuing.
                    # TODO: Cannot yield here. Yield here would leads to error for the next streaming request.
                    return
                except Exception as ex:
                    report_stream_failure(request, ex, FailureOrigin.MODEL_GENERATOR)
                    ex = await self._get_model_last_error(model.uid, ex)
                    logger.exception("Chat completion stream got an error: %s", ex)
                    await self._report_error_event(model_uid, str(ex))
                    try:
                        self.handle_request_limit_error(ex, sanitize=True)
                    except HTTPException as safe_error:
                        detail = safe_error.detail
                    else:
                        detail = self._chat_completion_error_detail()
                    # https://github.com/openai/openai-python/blob/e0aafc6c1a45334ac889fe3e54957d309c3af93f/src/openai/_streaming.py#L107
                    yield dict(data=json.dumps({"error": detail}))
                    return
                finally:
                    await _cleanup_model_stream(
                        abort_controller=abort_controller,
                        interrupted=not completed,
                        observed_iterator=observed_iterator,
                        model_iterator=iterator,
                        model=model,
                        request_id=request_id,
                        kind="chat",
                    )

            body_iterator = stream_results()
            response = EventSourceResponse(
                body_iterator, ping=XINFERENCE_SSE_PING_ATTEMPTS_SECONDS
            )
            abort_controller.bind_body_iterator(body_iterator)
            _observe_sse_disconnect(response, abort_controller.handle_disconnect)
            return response
        else:
            try:
                if is_vllm_backend:
                    return await _run_non_streaming_request(
                        request,
                        model,
                        request_id,
                        "chat",
                        model.chat(messages, kwargs, **model_call_kwargs),
                    )
                data = await model.chat(
                    messages,
                    kwargs,
                    **model_call_kwargs,
                )
                return Response(content=data, media_type="application/json")
            except Exception as e:
                e = await self._get_model_last_error(model.uid, e)
                logger.error(e, exc_info=True)
                await self._report_error_event(model_uid, str(e))
                self.handle_request_limit_error(e, sanitize=True)
                raise HTTPException(
                    status_code=500, detail=self._chat_completion_error_detail()
                )

    @staticmethod
    def _apply_multiple_choice_count(
        kwargs: Dict[str, Any],
        n: Optional[int],
        model_engine: Optional[str],
        is_vllm_backend: bool,
    ) -> None:
        if n is None or n == 1:
            return
        if n < 1:
            raise HTTPException(status_code=400, detail="n must be at least 1")
        if not (is_vllm_backend or model_engine in {"llama.cpp", "llama_cpp"}):
            raise HTTPException(
                status_code=400,
                detail="n > 1 is not supported by this model backend",
            )
        kwargs["n"] = n

    async def recommend_model(self, request: Request) -> JSONResponse:
        from .._compat import ValidationError
        from ..core.model_recommendation import (
            ModelRecommendationRequest,
            RecommendationModelNotFound,
        )

        try:
            body = ModelRecommendationRequest.parse_obj(await request.json())
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            result = await (await self._get_supervisor_ref()).recommend_model(
                body.dict(exclude_unset=True)
            )
            return JSONResponse(content=result)
        except RecommendationModelNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Model recommendation failed", exc_info=True)
            raise HTTPException(
                status_code=500, detail="Model recommendation failed"
            ) from exc

    async def query_engines_by_model_name(
        self, request: Request, model_name: str, model_type: Optional[str] = None
    ) -> JSONResponse:
        try:
            model_type = model_type or request.path_params.get("model_type", "LLM")
            enable_virtual_env = request.query_params.get("enable_virtual_env")
            if enable_virtual_env is not None:
                enable_virtual_env = enable_virtual_env.lower() in ("1", "true", "yes")
            content = await (
                await self._get_supervisor_ref()
            ).query_engines_by_model_name(
                model_name,
                model_type=model_type,
                enable_virtual_env=enable_virtual_env,
            )
            return JSONResponse(content=content)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def register_model(self, model_type: str, request: Request) -> JSONResponse:
        body = RegisterModelRequest.parse_obj(await request.json())
        model = body.model
        worker_ip = body.worker_ip
        persist = body.persist

        try:
            await (await self._get_supervisor_ref()).register_model(
                model_type, model, persist, worker_ip
            )
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=None)

    async def unregister_model(self, model_type: str, model_name: str) -> JSONResponse:
        try:
            await (await self._get_supervisor_ref()).unregister_model(
                model_type, model_name
            )
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=None)

    async def update_model_type(self, request: Request) -> JSONResponse:
        return await self._update_model_catalog(request, single_model=False)

    async def update_model_spec(self, request: Request) -> JSONResponse:
        return await self._update_model_catalog(request, single_model=True)

    async def _update_model_catalog(
        self, request: Request, *, single_model: bool
    ) -> JSONResponse:
        from ..core.exceptions import (
            ModelCatalogUnavailableError,
            ModelCatalogUpdateError,
        )
        from ..extensions import resolve_model_catalog_credentials

        # Parse client input before invoking any credential provider.
        try:
            schema = UpdateModelSpecRequest if single_model else UpdateModelRequest
            body = schema.parse_obj(await request.json())
        except (ValueError, TypeError):
            raise HTTPException(400, "Invalid request body") from None
        model_type = body.model_type
        model_name = (
            body.model_name if isinstance(body, UpdateModelSpecRequest) else None
        )

        try:
            credentials = resolve_model_catalog_credentials(request, model_type)
        except StarletteHTTPException:
            raise
        except Exception as e:
            logger.error(
                "Model catalog credential provider failed (%s)", type(e).__name__
            )
            raise HTTPException(
                500, "Failed to resolve model catalog credentials"
            ) from None

        try:
            supervisor_ref = await self._get_supervisor_ref()
            options: Dict[str, Any] = {}
            if credentials is not None:
                options["credentials"] = credentials
            if model_name is not None:
                options["model_name"] = model_name
            await supervisor_ref.update_model_type(model_type, **options)
        except ModelCatalogUnavailableError:
            raise HTTPException(
                503, "No workers available to update model catalog"
            ) from None
        except ModelCatalogUpdateError:
            raise HTTPException(502, "Failed to update model catalog") from None
        except Exception as e:
            logger.error("Model catalog supervisor unavailable (%s)", type(e).__name__)
            raise HTTPException(503, "Model catalog supervisor unavailable") from None

        response_data = {
            "data": {
                "model_type": model_type,
                "message": f"Successfully updated model type: {model_type}",
            }
        }

        if model_name is not None:
            response_data["data"].update(
                model_name=model_name,
                message=f"Successfully updated model: {model_name}",
            )

        return JSONResponse(content=response_data)

    async def list_model_registrations(
        self, model_type: str, detailed: bool = Query(False)
    ) -> JSONResponse:
        try:
            data = await (await self._get_supervisor_ref()).list_model_registrations(
                model_type, detailed=detailed
            )
            # Remove duplicate model names.
            model_names = set()
            final_data = []
            for item in data:
                if item["model_name"] not in model_names:
                    model_names.add(item["model_name"])
                    final_data.append(item)
            return JSONResponse(content=final_data)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_model_registrations(
        self, model_type: str, model_name: str
    ) -> JSONResponse:
        try:
            data = await (await self._get_supervisor_ref()).get_model_registration(
                model_type, model_name
            )
            return JSONResponse(content=data)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_model_events(self, model_uid: str) -> JSONResponse:
        try:
            event_collector_ref = await self._get_event_collector_ref()
            events = await event_collector_ref.get_model_events(model_uid)
            return JSONResponse(content=events)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def abort_request(
        self, request: Request, model_uid: str, request_id: str
    ) -> JSONResponse:
        try:
            payload = await request.json()
            block_duration = payload.get(
                "block_duration", XINFERENCE_DEFAULT_CANCEL_BLOCK_DURATION
            )
            logger.info(
                "Abort request with model uid: %s, request id: %s, block duration: %s",
                model_uid,
                request_id,
                block_duration,
            )
            supervisor_ref = await self._get_supervisor_ref()
            res = await actor_call(
                supervisor_ref,
                "abort_request",
                model_uid,
                request_id,
                block_duration,
                _rpc_correlation_id=get_model_request_id(request),
                _rpc_operation_request_id=request_id,
            )
            self._cancel_running_task(request_id, block_duration)
            return JSONResponse(content=res)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def list_vllm_supported_model_families(self) -> JSONResponse:
        try:
            from ..model.llm.vllm.core import (
                VLLM_SUPPORTED_CHAT_MODELS,
                VLLM_SUPPORTED_MODELS,
            )

            data = {
                "chat": VLLM_SUPPORTED_CHAT_MODELS,
                "generate": VLLM_SUPPORTED_MODELS,
            }
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    def extract_guided_params(raw_body: dict) -> dict:
        kwargs = {}
        raw_extra_body: dict = raw_body.get("extra_body")  # type: ignore
        if raw_body.get("guided_json"):
            kwargs["guided_json"] = raw_body.get("guided_json")
        if raw_body.get("guided_regex") is not None:
            kwargs["guided_regex"] = raw_body.get("guided_regex")
        if raw_body.get("guided_choice") is not None:
            kwargs["guided_choice"] = raw_body.get("guided_choice")
        if raw_body.get("guided_grammar") is not None:
            kwargs["guided_grammar"] = raw_body.get("guided_grammar")
        if raw_body.get("guided_json_object") is not None:
            kwargs["guided_json_object"] = raw_body.get("guided_json_object")
        if raw_body.get("guided_decoding_backend") is not None:
            kwargs["guided_decoding_backend"] = raw_body.get("guided_decoding_backend")
        if raw_body.get("guided_whitespace_pattern") is not None:
            kwargs["guided_whitespace_pattern"] = raw_body.get(
                "guided_whitespace_pattern"
            )
        # Parse OpenAI extra_body
        if raw_extra_body is not None:
            if raw_extra_body.get("guided_json"):
                kwargs["guided_json"] = raw_extra_body.get("guided_json")
            if raw_extra_body.get("guided_regex") is not None:
                kwargs["guided_regex"] = raw_extra_body.get("guided_regex")
            if raw_extra_body.get("guided_choice") is not None:
                kwargs["guided_choice"] = raw_extra_body.get("guided_choice")
            if raw_extra_body.get("guided_grammar") is not None:
                kwargs["guided_grammar"] = raw_extra_body.get("guided_grammar")
            if raw_extra_body.get("guided_json_object") is not None:
                kwargs["guided_json_object"] = raw_extra_body.get("guided_json_object")
            if raw_extra_body.get("guided_decoding_backend") is not None:
                kwargs["guided_decoding_backend"] = raw_extra_body.get(
                    "guided_decoding_backend"
                )
            if raw_extra_body.get("guided_whitespace_pattern") is not None:
                kwargs["guided_whitespace_pattern"] = raw_extra_body.get(
                    "guided_whitespace_pattern"
                )
            if raw_extra_body.get("platform") is not None:
                kwargs["platform"] = raw_extra_body.get("platform")
            if raw_extra_body.get("format") is not None:
                kwargs["format"] = raw_extra_body.get("format")

        return kwargs

    def _convert_openai_to_anthropic(self, openai_response: dict, model: str) -> dict:
        """Compatibility wrapper for existing callers and tests."""
        response = dict(openai_response)
        try:
            result = openai_to_anthropic(response, model)
        except AnthropicProtocolError:
            # Preserve the historical helper behavior for malformed tool JSON.
            for choice in response.get("choices", []):
                message = choice.get("message") or {}
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function") or {}
                    try:
                        json.loads(function.get("arguments", "{}"))
                    except (TypeError, json.JSONDecodeError):
                        function["arguments"] = "{}"
            result = openai_to_anthropic(response, model)

        # Keep the private helper's historical shape while the public endpoint
        # uses the canonical Anthropic representation from the protocol module.
        if result["stop_reason"] == "end_turn":
            result["stop_reason"] = "stop"
        for block in result["content"]:
            if block.get("type") == "tool_use":
                block["cache_control"] = {"type": "ephemeral"}
        result["usage"].setdefault("cache_creation_input_tokens", 0)
        result["usage"].setdefault("cache_read_input_tokens", 0)
        return result

    async def sdapi_upscalers(self, request: Request) -> Response:
        try:
            from ..model.image.upscaler.core import SCALER_INFO

            upscalers = [
                {
                    "name": upscaler.value,
                    "model_name": info.get("model_name", None),
                    "model_path": None,
                    "model_url": None,
                    "scale": info["scale"],
                }
                for upscaler, info in SCALER_INFO.items()
            ]
            from ..model.image.upscaler.latent import latent_upscale_modes

            upscalers.extend(
                {
                    "name": name,
                    "model_name": None,
                    "model_path": None,
                    "model_url": None,
                    "scale": 1,
                }
                for name in [*latent_upscale_modes, "Lanczos", "Nearest"]
            )
            # follow behavior of SD webui
            upscalers.insert(
                0,
                {
                    "name": "None",
                    "model_name": None,
                    "model_path": None,
                    "model_url": None,
                    "scale": 4,
                },
            )
            return JSONResponse(content=upscalers)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_progress(self, request: Request) -> Response:
        body = await self._parse_sdapi_request(SDAPIProgress, request, query=True)
        request_id = body.request_id

        try:
            result = {
                "progress": float(
                    await actor_call(
                        await self._get_supervisor_ref(),
                        "get_progress",
                        request_id,
                        _rpc_correlation_id=get_model_request_id(request),
                        _rpc_operation_request_id=request_id,
                    )
                )
            }
            return JSONResponse(content=result)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_controlnet_model_list(self, request: Request) -> Response:
        try:
            supervisor_ref = await self._get_supervisor_ref()
            sd_models = await self._get_sd_models(supervisor_ref, request)

            if not sd_models:
                raise ValueError("No running sd models")

            correlation_id = get_model_request_id(request)
            model_list = []
            for model_uid in sd_models:
                model = correlate_model_ref(
                    await actor_call(
                        supervisor_ref,
                        "get_model",
                        model_uid,
                        _rpc_correlation_id=correlation_id,
                    ),
                    correlation_id,
                )
                result = json.loads(await model.controlnet_model_list())
                model_list.extend(result["model_list"])
            return Response(
                content=json.dumps({"model_list": model_list}),
                media_type="application/json",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_sd_models(self, supervisor_ref, request):
        models = await supervisor_ref.list_models()
        sd_models = []
        for name, info in models.items():
            if info.get("model_type") != "image":
                continue
            if not info.get("controlnet") and info.get("model_base") not in (
                "SD 1.5",
                "SD 2.0",
                "SD 2.1",
                "SDXL",
            ):
                continue
            try:
                self._check_model_access(request, name, "image")
            except HTTPException as exc:
                if exc.status_code == 403:
                    continue
                raise
            sd_models.append(name)
        return sd_models

    async def sdapi_controlnet_module_list(self, request: Request) -> Response:
        try:
            supervisor_ref = await self._get_supervisor_ref()
            sd_models = await self._get_sd_models(supervisor_ref, request)

            if not sd_models:
                raise ValueError("No running sd models")

            # random pick one model to process detect
            correlation_id = get_model_request_id(request)
            model = correlate_model_ref(
                await actor_call(
                    supervisor_ref,
                    "get_model",
                    sd_models[0],
                    _rpc_correlation_id=correlation_id,
                ),
                correlation_id,
            )
            result = await model.controlnet_module_list()
            return Response(content=result, media_type="application/json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_controlnet_control_types(self, request: Request) -> Response:
        try:
            supervisor_ref = await self._get_supervisor_ref()
            sd_models = await self._get_sd_models(supervisor_ref, request)

            if not sd_models:
                raise ValueError("No running sd models")

            # random pick one model to process detect
            correlation_id = get_model_request_id(request)
            model = correlate_model_ref(
                await actor_call(
                    supervisor_ref,
                    "get_model",
                    sd_models[0],
                    _rpc_correlation_id=correlation_id,
                ),
                correlation_id,
            )
            result = await model.controlnet_control_types()
            return Response(content=result, media_type="application/json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_controlnet_detect(self, request: Request) -> Response:
        body = await self._parse_sdapi_request(SDAPIControlNetDetect, request)
        if body.controlnet_images and not body.controlnet_input_images:
            body.controlnet_input_images = body.controlnet_images

        try:
            supervisor_ref = await self._get_supervisor_ref()

            sd_models = await self._get_sd_models(supervisor_ref, request)

            if not sd_models:
                raise ValueError("No running sd models")

            # random pick one model to process detect
            correlation_id = get_model_request_id(request)
            model = correlate_model_ref(
                await actor_call(
                    supervisor_ref,
                    "get_model",
                    sd_models[0],
                    _rpc_correlation_id=correlation_id,
                ),
                correlation_id,
            )

            kwargs = dict(body)
            kwargs.pop("controlnet_images", None)
            result = await model.controlnet_detect(**kwargs)
            return Response(content=result, media_type="application/json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_loras(self, request: Request) -> Response:
        specs = await (await self._get_supervisor_ref()).list_model_registrations(
            "image", detailed=True
        )
        loras = {}
        for spec in specs:
            if spec.get("model_family") != "lora":
                continue
            name = spec["model_name"]
            loras[name] = {
                "name": name,
                "alias": (spec.get("metadata") or {}).get("ss_output_name", name),
                "path": None,
                "metadata": spec.get("metadata") or {},
            }
        return JSONResponse(content=list(loras.values()))

    async def sdapi_interrupt(self, request: Request) -> Response:
        body = await self._parse_sdapi_request(SDAPIInterrupt, request)
        self._check_model_access(request, body.model, "image")
        model = await require_model(
            self._get_supervisor_ref, body.model, self._report_error_event
        )
        result = await actor_call(
            model,
            "abort_request",
            body.request_id,
            _rpc_correlation_id=get_model_request_id(request),
            _rpc_operation_request_id=body.request_id,
        )
        return JSONResponse(content={"status": result})


def run(
    supervisor_address: str,
    host: str,
    port: int,
    logging_conf: Optional[dict] = None,
    api_role: str = "supervisor",
):
    from ..deploy.utils import (
        update_all_formatter_addresses,
        update_logging_config_addresses,
    )

    update_all_formatter_addresses(api_role, supervisor_address)
    update_logging_config_addresses(logging_conf, api_role, supervisor_address)
    logger.info("Starting Xinference at endpoint: http://%s:%s", host, port)
    try:
        api = RESTfulAPI(
            supervisor_address=supervisor_address,
            host=host,
            port=port,
        )
        api.serve(logging_conf=logging_conf)
    except SystemExit:
        logger.warning("Failed to create socket with port %d", port)
        # compare the reference to differentiate between the cases where the user specify the
        # default port and the user does not specify the port.
        if port is XINFERENCE_DEFAULT_ENDPOINT_PORT:
            port = get_next_port()
            logger.info(f"Found available port: {port}")
            logger.info("Starting Xinference at endpoint: http://%s:%s", host, port)
            api = RESTfulAPI(
                supervisor_address=supervisor_address,
                host=host,
                port=port,
            )
            api.serve(logging_conf=logging_conf)
        else:
            raise


def run_in_subprocess(
    supervisor_address: str,
    host: str,
    port: int,
    logging_conf: Optional[dict] = None,
    environment: Optional[Dict[str, str]] = None,
    api_role: str = "supervisor",
) -> multiprocessing.Process:
    p = multiprocessing.Process(
        target=_run_with_environment,
        args=(supervisor_address, host, port, logging_conf, environment, api_role),
    )
    p.daemon = True
    p.start()
    return p


def _run_with_environment(
    supervisor_address: str,
    host: str,
    port: int,
    logging_conf: Optional[dict],
    environment: Optional[Dict[str, str]],
    api_role: str = "supervisor",
):
    if environment:
        os.environ.update(environment)
    run(supervisor_address, host, port, logging_conf, api_role=api_role)
