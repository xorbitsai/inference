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

import json
import logging
import time
import uuid
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
            from langfuse import Langfuse

            self._client = Langfuse()
            self._enabled = True
            logger.info("Langfuse tracing is enabled.")
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
        trace_id = str(uuid.uuid4())
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
            duration_ms = (time.time() - start_time) * 1000

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
                    trace_id=trace_id,
                    model_type=model_type,
                    operation=operation,
                    model_name=model_name,
                    input_summary=input_summary,
                    output_summary=output_summary,
                    usage=usage,
                    duration_ms=duration_ms,
                    status_code=status_code,
                    error=error_msg,
                )
            except Exception as e:
                logger.warning("Failed to report Langfuse trace: %s", e)

    def _report_trace(
        self,
        trace_id: str,
        model_type: str,
        operation: str,
        model_name: str,
        input_summary: Any,
        output_summary: Any,
        usage: Optional[Dict[str, Any]],
        duration_ms: float,
        status_code: int,
        error: Optional[str],
    ):
        """Create a Langfuse trace with a generation/span for the model call."""
        client = self._tracer.client
        if client is None:
            return

        trace_name = f"xinference/{operation}"
        metadata = {
            "model_type": model_type,
            "operation": operation,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
        }

        level = "ERROR" if error or status_code >= 400 else "DEFAULT"
        status_message = error if error else ("OK" if status_code < 400 else f"HTTP {status_code}")

        trace = client.trace(
            id=trace_id,
            name=trace_name,
            input=input_summary,
            output=output_summary,
            metadata=metadata,
        )

        # For LLM operations, create a generation observation
        if model_type == "llm":
            generation_kwargs = {
                "name": operation,
                "model": model_name,
                "input": input_summary,
                "output": output_summary,
                "metadata": metadata,
                "level": level,
                "status_message": status_message,
            }
            if usage:
                generation_kwargs["usage"] = usage
            trace.generation(**generation_kwargs)
        else:
            # For non-LLM operations, create a span
            trace.span(
                name=operation,
                input=input_summary,
                output=output_summary,
                metadata=metadata,
                level=level,
                status_message=status_message,
            )
