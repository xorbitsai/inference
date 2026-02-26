# Copyright 2022-2026 XProbe Inc.
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

"""
Langfuse integration module for Xinference.

Provides observability/tracing for all model API calls (LLM, Embedding,
Rerank, Audio, Image, Video) via a pure ASGI middleware that intercepts
requests and responses, then reports traces to a Langfuse server.

Design:
    The middleware itself does NOT read request/response bodies,
    avoiding all known issues with BaseHTTPMiddleware and file uploads.
    Instead, each API handler injects tracing metadata into `request.state`
    (e.g., model_uid, input, output, usage). The middleware simply
    measures latency and reads `request.state` after the response is sent.

Enable by setting environment variables:
    XINFERENCE_LANGFUSE_ENABLED=1
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_BASE_URL=https://cloud.langfuse.com  (or self-hosted URL)
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from ..constants import XINFERENCE_LANGFUSE_ENABLED

logger = logging.getLogger(__name__)

# Map of API path prefixes -> (model_type, operation)
_TRACED_ROUTES: Dict[str, Tuple[str, str]] = {
    "/v1/chat/completions": ("llm", "chat_completion"),
    "/v1/completions": ("llm", "completion"),
    "/anthropic/v1/messages": ("llm", "anthropic_message"),
    "/v1/flexible/infers": ("llm", "flexible_infer"),
    "/v1/embeddings": ("embedding", "create_embedding"),
    "/v1/rerank": ("rerank", "rerank"),
    "/v1/audio/transcriptions": ("audio", "transcription"),
    "/v1/audio/translations": ("audio", "translation"),
    "/v1/audio/speech": ("audio", "speech"),
    "/v1/images/generations": ("image", "generation"),
    "/v1/images/variations": ("image", "variation"),
    "/v1/images/inpainting": ("image", "inpainting"),
    "/v1/images/ocr": ("image", "ocr"),
    "/v1/images/edits": ("image", "edit"),
    "/v1/videos": ("video", "text_to_video"),
    "/sdapi/v1/txt2img": ("image", "sdapi_txt2img"),
    "/sdapi/v1/img2img": ("image", "sdapi_img2img"),
}


class LangfuseClient:
    """Singleton wrapper around the Langfuse Python SDK client."""

    _instance: Optional["LangfuseClient"] = None

    def __init__(self):
        self._client = None
        self._enabled = False

    @classmethod
    def get_instance(cls) -> "LangfuseClient":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._init()
        return cls._instance

    def _init(self):
        if not XINFERENCE_LANGFUSE_ENABLED:
            self._enabled = False
            return

        try:
            langfuse_module = importlib.import_module("langfuse")
            langfuse_cls = getattr(langfuse_module, "Langfuse")
            self._client = langfuse_cls()
            self._enabled = hasattr(self._client, "start_as_current_observation")
            if self._enabled:
                logger.info("Langfuse tracing is enabled (API mode: v3).")
            else:
                logger.warning(
                    "Langfuse SDK does not expose v3 API `start_as_current_observation`; "
                    "trace reporting is disabled."
                )
        except ImportError:
            logger.warning(
                "XINFERENCE_LANGFUSE_ENABLED is set, but the 'langfuse' package "
                "is not installed. Install it with: pip install langfuse"
            )
            self._enabled = False
        except Exception as e:
            logger.warning("Failed to initialize Langfuse client: %s", e)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def client(self):
        return self._client

    def shutdown(self):
        """Flush pending events before shutdown."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as e:
                logger.warning("Error flushing Langfuse client: %s", e)


def _match_route(path: str) -> Optional[Tuple[str, str]]:
    """Match a request path to a traced route, return (model_type, operation) or None."""
    for route_prefix, info in _TRACED_ROUTES.items():
        if path == route_prefix or path.startswith(route_prefix + "/"):
            return info
    return None


def set_langfuse_trace_data(request, **kwargs):
    """
    Helper for API handlers to inject tracing metadata into the request.

    Usage in any handler:
        from xinference.core.langfuse_integration import set_langfuse_trace_data
        set_langfuse_trace_data(request,
            model_uid="my-model",
            input={"messages": [...]},
            output={"choices": [...]},
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    Supported keys: model_uid, input, output, usage
    """
    if not hasattr(request.state, "langfuse_trace"):
        request.state.langfuse_trace = {}
    request.state.langfuse_trace.update(kwargs)


class LangfuseMiddleware:
    """
    Pure ASGI middleware that traces model API calls to Langfuse.

    Unlike BaseHTTPMiddleware, this does NOT interfere with request/response
    body streaming, so it's fully compatible with multipart/form-data file
    uploads (Audio, Image, Video).

    The middleware only records timing and reads metadata that handlers
    inject into request.state via set_langfuse_trace_data().
    """

    def __init__(self, app):
        self.app = app
        self._tracer = LangfuseClient.get_instance()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self._tracer.enabled:
            await self.app(scope, receive, send)
            return

        # Check if this path should be traced
        path = scope.get("path", "")
        route_info = _match_route(path)
        if route_info is None:
            await self.app(scope, receive, send)
            return

        model_type, operation = route_info
        start_time = time.time()

        # Track response status code
        status_code = 500  # default to error

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)

        error_msg = None
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Read tracing metadata injected by the handler
            # scope["state"] is the same dict backing request.state
            state = scope.get("state", {})
            trace_data = state.get("langfuse_trace", {})

            model_name = str(trace_data.get("model_uid", "unknown"))
            input_summary = trace_data.get("input")
            output_summary = trace_data.get("output")
            usage_raw = trace_data.get("usage")

            # Normalize usage
            usage = None
            if usage_raw and isinstance(usage_raw, dict):
                usage = {
                    "input": usage_raw.get("prompt_tokens"),
                    "output": usage_raw.get("completion_tokens"),
                    "total": usage_raw.get("total_tokens"),
                }

            try:
                self._report_trace(
                    model_type=model_type,
                    operation=operation,
                    model_name=model_name,
                    input_summary=input_summary,
                    output_summary=output_summary,
                    usage=usage,
                    duration_ms=duration_ms,
                    start_time_s=start_time,
                    end_time_s=end_time,
                    status_code=status_code,
                    error=error_msg,
                )
            except Exception as e:
                logger.warning("Failed to report Langfuse trace: %s", e)

    def _report_trace(
        self,
        model_type: str,
        operation: str,
        model_name: str,
        input_summary: Any,
        output_summary: Any,
        usage: Optional[Dict[str, Any]],
        duration_ms: float,
        start_time_s: float,
        end_time_s: float,
        status_code: int,
        error: Optional[str],
    ):
        """Create a Langfuse trace/observation for the model call."""
        client = self._tracer.client
        if client is None:
            return

        trace_name = f"xinference/{operation}"
        result_text = self._extract_result_text(output_summary)
        metadata = {
            "model_type": model_type,
            "operation": operation,
            "model_name": model_name,
            "status_code": status_code,
            "latency_ms": round(duration_ms, 2),
        }
        if result_text:
            metadata["result_text"] = result_text

        level = "ERROR" if error or status_code >= 400 else "DEFAULT"
        status_message = (
            error if error else ("OK" if status_code < 400 else f"HTTP {status_code}")
        )

        self._report_trace_v3(
            client=client,
            trace_name=trace_name,
            model_type=model_type,
            operation=operation,
            model_name=model_name,
            input_summary=input_summary,
            output_summary=output_summary,
            metadata=metadata,
            usage=usage,
            level=level,
            status_message=status_message,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )

    @staticmethod
    def _invoke_with_supported_kwargs(func, kwargs: Dict[str, Any]):
        """Invoke a callable with kwargs filtered by its signature."""
        sig = inspect.signature(func)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return func(**kwargs)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
        return func(**filtered_kwargs)

    @staticmethod
    def _extract_result_text(output_summary: Any, max_len: int = 512) -> Optional[str]:
        """Extract concise text result for metadata display."""
        if output_summary is None:
            return None

        text_value: Optional[str] = None
        if isinstance(output_summary, str):
            stripped = output_summary.strip()
            if stripped.startswith("{"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        output_summary = parsed
                    else:
                        text_value = stripped
                except Exception:
                    text_value = stripped
            else:
                text_value = stripped
        elif isinstance(output_summary, dict):
            candidate_keys = ("text", "result", "content", "transcript")
            for key in candidate_keys:
                value = output_summary.get(key)
                if isinstance(value, str) and value.strip():
                    text_value = value
                    break
            if text_value is None:
                for value in output_summary.values():
                    if isinstance(value, str) and value.strip():
                        text_value = value
                        break

        if not text_value:
            return None
        normalized = text_value.strip().replace("\n", " ")
        if len(normalized) > max_len:
            return normalized[:max_len] + "..."
        return normalized

    def _report_trace_v3(
        self,
        client,
        trace_name: str,
        model_type: str,
        operation: str,
        model_name: str,
        input_summary: Any,
        output_summary: Any,
        metadata: Dict[str, Any],
        usage: Optional[Dict[str, Any]],
        level: str,
        status_message: str,
        start_time_s: float,
        end_time_s: float,
    ):
        as_type = "generation" if model_type == "llm" else "span"
        start_time = datetime.fromtimestamp(start_time_s, tz=timezone.utc)
        end_time = datetime.fromtimestamp(end_time_s, tz=timezone.utc)
        observation_kwargs = {
            "name": trace_name,
            "as_type": as_type,
            "input": input_summary,
            "output": output_summary,
            "metadata": metadata,
            "start_time": start_time,
            "end_time": end_time,
        }
        with self._invoke_with_supported_kwargs(
            client.start_as_current_observation, observation_kwargs
        ) as observation:
            if hasattr(observation, "update"):
                update_kwargs = {
                    "name": operation,
                    "input": input_summary,
                    "output": output_summary,
                    "metadata": metadata,
                    "level": level,
                    "status_message": status_message,
                    "start_time": start_time,
                    "end_time": end_time,
                }
                if model_type == "llm":
                    update_kwargs["model"] = model_name
                    if usage:
                        update_kwargs["usage"] = usage
                elif model_name and model_name != "unknown":
                    update_kwargs["model"] = model_name
                self._invoke_with_supported_kwargs(observation.update, update_kwargs)
