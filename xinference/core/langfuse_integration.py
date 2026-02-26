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
Rerank, Audio, Image, Video) via a FastAPI middleware that intercepts
requests and responses, then reports traces to a Langfuse server.

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

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.types import ASGIApp

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


def _safe_parse_json(body: bytes) -> Optional[Dict[str, Any]]:
    """Try to parse JSON body, return None on failure."""
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _extract_model_name(request_body: Optional[Dict[str, Any]]) -> str:
    """Extract model name from the request body."""
    if request_body and isinstance(request_body, dict):
        return str(request_body.get("model", "unknown"))
    return "unknown"


def _extract_input_summary(
    operation: str, request_body: Optional[Dict[str, Any]]
) -> Any:
    """Extract a concise input summary suitable for Langfuse trace display."""
    if request_body is None:
        return None

    if operation == "chat_completion" or operation == "anthropic_message":
        return request_body.get("messages")
    elif operation == "completion":
        return request_body.get("prompt")
    elif operation == "create_embedding":
        inp = request_body.get("input")
        # Truncate if input is very long
        if isinstance(inp, str) and len(inp) > 500:
            return inp[:500] + "..."
        return inp
    elif operation == "rerank":
        return {
            "query": request_body.get("query"),
            "documents_count": len(request_body.get("documents", [])),
        }
    elif operation in ("transcription", "translation"):
        return {"type": operation}
    elif operation == "speech":
        return {"input": request_body.get("input", "")[:200]}
    elif operation in ("generation", "sdapi_txt2img"):
        return {"prompt": request_body.get("prompt", "")[:200]}
    else:
        # For image/video operations, return a minimal summary
        return {"operation": operation}


def _extract_usage(
    operation: str, response_body: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Extract token usage from a response body (for LLM operations)."""
    if response_body is None or not isinstance(response_body, dict):
        return None
    usage = response_body.get("usage")
    if usage and isinstance(usage, dict):
        return {
            "input": usage.get("prompt_tokens"),
            "output": usage.get("completion_tokens"),
            "total": usage.get("total_tokens"),
        }
    return None


def _extract_output_summary(
    operation: str, response_body: Optional[Dict[str, Any]]
) -> Any:
    """Extract a concise output summary from the response."""
    if response_body is None:
        return None

    if operation in ("chat_completion", "anthropic_message"):
        choices = response_body.get("choices", [])
        if choices and isinstance(choices, list):
            first = choices[0]
            if isinstance(first, dict):
                return first.get("message") or first.get("delta")
        # Anthropic format
        content = response_body.get("content")
        if content:
            return content
    elif operation == "completion":
        choices = response_body.get("choices", [])
        if choices and isinstance(choices, list):
            first = choices[0]
            if isinstance(first, dict):
                return first.get("text")
    elif operation == "create_embedding":
        data = response_body.get("data", [])
        return {"embedding_count": len(data)}
    elif operation == "rerank":
        results = response_body.get("results", [])
        return {"result_count": len(results)}

    return None


class LangfuseMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/Starlette middleware that traces model API calls to Langfuse.

    Only active when XINFERENCE_LANGFUSE_ENABLED environment variable is truthy.
    Non-model endpoints (admin, UI, metrics, etc.) are ignored.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._tracer = LangfuseClient.get_instance()

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._tracer.enabled:
            return await call_next(request)

        # Check if this path should be traced
        route_info = _match_route(request.url.path)
        if route_info is None:
            return await call_next(request)

        model_type, operation = route_info
        trace_id = str(uuid.uuid4())
        start_time = time.time()

        # Try to read the request body for tracing
        request_body = None
        try:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                body_bytes = await request.body()
                request_body = _safe_parse_json(body_bytes)
            elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
                # Starlette caches the form payload on the request object after the first call,
                # so it's safe to call here without breaking FastAPI's form parsing.
                form = await request.form()
                request_body = {}
                for k, v in form.items():
                    # Exclude file uploads from being captured in the trace body
                    if not hasattr(v, "filename"):
                        request_body[k] = v
        except Exception:
            pass

        model_name = _extract_model_name(request_body)
        input_summary = _extract_input_summary(operation, request_body)

        # Determine if this is a streaming request
        is_stream = False
        if request_body and isinstance(request_body, dict):
            # Form values are typically strings, we need to handle "true", "1", etc.
            stream_val = request_body.get("stream", False)
            if isinstance(stream_val, str):
                is_stream = stream_val.lower() in ("true", "1", "yes")
            else:
                is_stream = bool(stream_val)

        # Call the actual handler
        response = None
        error_msg = None
        response_body = None

        try:
            response = await call_next(request)

            # For non-streaming responses, capture the response body
            if not is_stream and hasattr(response, "body"):
                response_body = _safe_parse_json(response.body)
            elif not is_stream and isinstance(response, StreamingResponse):
                # For StreamingResponse (non-SSE), collect the body
                body_chunks = []
                async for chunk in response.body_iterator:
                    if isinstance(chunk, bytes):
                        body_chunks.append(chunk)
                    else:
                        body_chunks.append(chunk.encode("utf-8"))
                collected_body = b"".join(body_chunks)
                response_body = _safe_parse_json(collected_body)

                # Reconstruct the response since we consumed the iterator
                response = Response(
                    content=collected_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            status_code = response.status_code if response else 500

            # Report to Langfuse in background (non-blocking)
            try:
                self._report_trace(
                    trace_id=trace_id,
                    model_type=model_type,
                    operation=operation,
                    model_name=model_name,
                    input_summary=input_summary,
                    output_summary=_extract_output_summary(operation, response_body),
                    usage=_extract_usage(operation, response_body),
                    duration_ms=duration_ms,
                    status_code=status_code,
                    is_stream=is_stream,
                    error=error_msg,
                )
            except Exception as e:
                logger.debug("Failed to report Langfuse trace: %s", e)

        return response

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
        is_stream: bool,
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
            "is_stream": is_stream,
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
