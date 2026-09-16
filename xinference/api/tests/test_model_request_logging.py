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

import json
import logging
import os
import stat
import uuid

import pytest
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse, StreamingResponse

from .. import model_request_logging
from ..model_request_logging import (
    ModelRequestLoggingRoute,
    _ModelRequestRotatingFileHandler,
)


def _app(path: str, endpoint, *, dependencies=None) -> FastAPI:
    app = FastAPI()
    router = APIRouter(route_class=ModelRequestLoggingRoute)
    router.add_api_route(path, endpoint, methods=["POST"], dependencies=dependencies)
    app.include_router(router)
    return app


def _enable_capture(monkeypatch):
    events = []
    monkeypatch.setattr(
        model_request_logging, "XINFERENCE_MODEL_REQUEST_LOG_ENABLED", True
    )
    monkeypatch.setattr(model_request_logging, "_write_event", events.append)
    return events


def test_json_body_is_preserved_after_endpoint_parsing(monkeypatch):
    events = _enable_capture(monkeypatch)
    payload = {
        "model": "audio-model",
        "stream": False,
        "media": [{"base64": "AAECAwQ=", "latent": [0.1, 0.2, 0.3]}],
    }

    async def endpoint(request: Request):
        return JSONResponse(await request.json())

    response = TestClient(_app("/v1/audio/speech", endpoint)).post(
        "/v1/audio/speech",
        content=json.dumps(payload).encode(),
        headers={
            "content-type": "application/json",
            "request-id": "caller-request-id",
            "x-request-id": "lower-priority-id",
        },
    )

    assert response.status_code == 200
    assert response.json() == payload
    assert response.headers["x-request-id"] == "caller-request-id"
    assert [event["event"] for event in events] == [
        "model_request_started",
        "model_request_finished",
    ]
    assert events[0]["request_body"] == payload
    assert events[0]["model_uid"] == "audio-model"
    assert events[1]["success"] is True


def test_generated_request_id_preserves_xinf_prefix(monkeypatch):
    _enable_capture(monkeypatch)

    async def endpoint(request: Request):
        await request.json()
        return {"ok": True}

    response = TestClient(_app("/v1/completions", endpoint)).post(
        "/v1/completions", json={"model": "llm"}
    )
    request_id = response.headers["x-request-id"]
    assert request_id.startswith("xinf-")
    assert str(uuid.UUID(request_id.removeprefix("xinf-"))) == request_id.removeprefix(
        "xinf-"
    )


def test_invalid_request_id_is_replaced(monkeypatch):
    events = _enable_capture(monkeypatch)

    async def endpoint(request: Request):
        await request.json()
        return {"ok": True}

    response = TestClient(_app("/v1/completions", endpoint)).post(
        "/v1/completions",
        json={"model": "llm"},
        headers={"x-request-id": "x" * 257},
    )
    assert response.headers["x-request-id"].startswith("xinf-")
    assert events[0]["request_id"] == response.headers["x-request-id"]


def test_negative_content_length_is_treated_as_unknown_size():
    request = Request(
        {
            "type": "http",
            "headers": [(b"content-length", b"-1")],
        }
    )

    assert model_request_logging._body_size(request) is None


def test_body_over_size_limit_is_not_serialized(monkeypatch):
    events = _enable_capture(monkeypatch)
    monkeypatch.setattr(
        model_request_logging, "XINFERENCE_MODEL_REQUEST_LOG_BODY_MAX_BYTES", 8
    )

    async def endpoint(request: Request):
        await request.json()
        return {"ok": True}

    response = TestClient(_app("/v1/completions", endpoint)).post(
        "/v1/completions", json={"model": "a-large-model"}
    )
    assert response.status_code == 200
    assert "request_body" not in events[0]
    assert events[0]["request_body_omitted"]["reason"] == "size_limit"


def test_returned_authorization_failure_does_not_log_body(monkeypatch):
    events = _enable_capture(monkeypatch)

    async def endpoint(request: Request):
        await request.json()
        return JSONResponse(status_code=403, content={"detail": "forbidden"})

    response = TestClient(_app("/v1/completions", endpoint)).post(
        "/v1/completions", json={"model": "llm", "secret": "do-not-log"}
    )
    assert response.status_code == 403
    assert [event["event"] for event in events] == ["model_request_failed"]
    assert "request_body" not in events[0]
    assert "do-not-log" not in json.dumps(events)


def test_raised_payload_too_large_does_not_log_body(monkeypatch):
    events = _enable_capture(monkeypatch)

    async def endpoint(request: Request):
        await request.json()
        raise HTTPException(status_code=413, detail="payload too large")

    response = TestClient(_app("/v1/completions", endpoint)).post(
        "/v1/completions", json={"model": "llm", "secret": "do-not-log"}
    )
    assert response.status_code == 413
    assert [event["event"] for event in events] == ["model_request_failed"]
    assert "request_body" not in events[0]
    assert "do-not-log" not in json.dumps(events)


def test_route_matching_uses_path_without_root_path(monkeypatch):
    events = _enable_capture(monkeypatch)

    async def endpoint(request: Request):
        return JSONResponse(await request.json())

    client = TestClient(_app("/v1/completions", endpoint), root_path="/xinference")
    response = client.post("/v1/completions", json={"model": "llm"})

    assert response.status_code == 200
    assert [event["event"] for event in events] == [
        "model_request_started",
        "model_request_finished",
    ]
    assert events[0]["endpoint"] == "/v1/completions"


def test_authentication_failure_does_not_log_body(monkeypatch):
    from fastapi import Depends

    events = _enable_capture(monkeypatch)

    async def deny():
        raise HTTPException(status_code=401, detail="invalid credential")

    async def endpoint(request: Request):
        await request.json()
        return {"ok": True}

    response = TestClient(
        _app("/v1/completions", endpoint, dependencies=[Depends(deny)])
    ).post("/v1/completions", json={"model": "llm", "secret": "do-not-log"})
    assert response.status_code == 401
    assert response.headers["x-request-id"].startswith("xinf-")
    assert [event["event"] for event in events] == ["model_request_failed"]
    assert "request_body" not in events[0]
    assert "do-not-log" not in json.dumps(events)


def test_multipart_logs_fields_and_filenames_not_bytes(monkeypatch):
    events = _enable_capture(monkeypatch)

    async def endpoint(
        request: Request,
        model: str = Form(...),
        prompt_speech: UploadFile = File(...),
    ):
        assert await prompt_speech.read() == b"secret-audio-bytes"
        return {"model": model}

    response = TestClient(_app("/v1/audio/speech", endpoint)).post(
        "/v1/audio/speech",
        data={"model": "tts"},
        files={"prompt_speech": ("voice.wav", b"secret-audio-bytes")},
    )
    assert response.status_code == 200
    assert events[0]["request_body"] == {
        "model": "tts",
        "prompt_speech": {"filename": "voice.wav"},
    }
    assert "secret-audio-bytes" not in json.dumps(events)


def test_disabled_logging_does_not_wrap_stream(monkeypatch):
    monkeypatch.setattr(
        model_request_logging, "XINFERENCE_MODEL_REQUEST_LOG_ENABLED", False
    )

    def unexpected_wrap(*args, **kwargs):
        raise AssertionError("disabled logging must not wrap streaming responses")

    monkeypatch.setattr(ModelRequestLoggingRoute, "_wrap_stream", unexpected_wrap)

    async def endpoint(request: Request):
        await request.json()

        async def generate():
            yield b"one"
            yield b"two"

        return StreamingResponse(generate())

    response = TestClient(_app("/v1/chat/completions", endpoint)).post(
        "/v1/chat/completions", json={"model": "llm", "stream": True}
    )

    assert response.content == b"onetwo"
    assert response.headers["x-request-id"].startswith("xinf-")


def test_stream_completion_is_logged(monkeypatch):
    events = _enable_capture(monkeypatch)

    async def endpoint(request: Request):
        await request.json()

        async def generate():
            yield b"one"
            yield b"two"

        return StreamingResponse(generate())

    with TestClient(_app("/v1/chat/completions", endpoint)).stream(
        "POST", "/v1/chat/completions", json={"model": "llm", "stream": True}
    ) as response:
        assert b"".join(response.iter_bytes()) == b"onetwo"

    assert events[-1]["event"] == "model_request_finished"
    assert events[-1]["stream"] is True
    assert events[-1]["stream_completed"] is True


def test_stream_failure_is_logged(monkeypatch):
    events = _enable_capture(monkeypatch)

    async def endpoint(request: Request):
        await request.json()

        async def generate():
            yield b"one"
            raise RuntimeError("stream failed")

        return StreamingResponse(generate())

    with pytest.raises(RuntimeError, match="stream failed"):
        with TestClient(_app("/v1/chat/completions", endpoint)).stream(
            "POST", "/v1/chat/completions", json={"model": "llm", "stream": True}
        ) as response:
            b"".join(response.iter_bytes())

    assert events[-1]["event"] == "model_request_failed"
    assert events[-1]["stream_completed"] is False
    assert events[-1]["error"]["type"] == "RuntimeError"


def test_logger_write_failure_does_not_change_response_or_leak_body(
    monkeypatch, caplog
):
    class BrokenLogger:
        def info(self, message):
            raise OSError("disk full")

    monkeypatch.setattr(
        model_request_logging, "XINFERENCE_MODEL_REQUEST_LOG_ENABLED", True
    )
    monkeypatch.setattr(
        model_request_logging, "_get_model_request_logger", lambda: BrokenLogger()
    )

    async def endpoint(request: Request):
        await request.json()
        return {"ok": True}

    with caplog.at_level("WARNING", logger=model_request_logging.__name__):
        response = TestClient(_app("/v1/completions", endpoint)).post(
            "/v1/completions", json={"model": "llm", "secret": "do-not-leak"}
        )

    assert response.status_code == 200
    assert "do-not-leak" not in caplog.text
    assert "Failed to write model request log" in caplog.text


def test_log_scheduling_failure_does_not_change_response(monkeypatch, caplog):
    _enable_capture(monkeypatch)

    async def broken_to_thread(*args, **kwargs):
        raise RuntimeError("executor unavailable")

    monkeypatch.setattr(model_request_logging.asyncio, "to_thread", broken_to_thread)

    async def endpoint(request: Request):
        await request.json()
        return {"ok": True}

    with caplog.at_level("WARNING", logger=model_request_logging.__name__):
        response = TestClient(_app("/v1/completions", endpoint)).post(
            "/v1/completions", json={"model": "llm"}
        )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert "Failed to schedule a model request log write" in caplog.text


def test_request_log_handler_creates_parent_directory(tmp_path):
    log_file = tmp_path / "nested" / "logs" / "model_request.log"
    handler = _ModelRequestRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        backupCount=1,
        maxBytes=0,
        retention_days=1,
        encoding="utf8",
    )
    try:
        assert log_file.exists()
    finally:
        handler.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are required")
def test_rotated_request_logs_keep_restrictive_permissions(tmp_path):
    log_file = tmp_path / "model_request.log"
    handler = _ModelRequestRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        backupCount=2,
        maxBytes=1,
        retention_days=7,
        encoding="utf8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    request_logger = logging.getLogger(f"xinference.model_request.test.{id(handler)}")
    request_logger.handlers.clear()
    request_logger.addHandler(handler)
    request_logger.setLevel(logging.INFO)
    request_logger.propagate = False
    try:
        request_logger.info("first")
        request_logger.info("second")
    finally:
        request_logger.removeHandler(handler)
        handler.close()

    files = [
        path
        for path in tmp_path.glob("model_request.log*")
        if not path.name.endswith(".rotate.lock")
    ]
    assert len(files) >= 2
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in files)


def test_swallowed_stream_failure_is_logged_once(monkeypatch):
    from ..streaming_outcome import FailureOrigin, report_stream_failure

    events = _enable_capture(monkeypatch)

    async def endpoint(request: Request):
        await request.json()

        async def generate():
            yield b"one"
            try:
                raise RuntimeError("backend failed")
            except RuntimeError as exc:
                report_stream_failure(request, exc, FailureOrigin.MODEL_GENERATOR)
                yield b'event: error\\ndata: {"error":"backend failed"}\\n\\n'

        return StreamingResponse(generate())

    with TestClient(_app("/v1/chat/completions", endpoint)).stream(
        "POST", "/v1/chat/completions", json={"model": "llm", "stream": True}
    ) as response:
        body = b"".join(response.iter_bytes())

    assert body == b'oneevent: error\\ndata: {"error":"backend failed"}\\n\\n'
    terminal = [event for event in events if event["event"] != "model_request_started"]
    assert len(terminal) == 1
    assert terminal[0]["event"] == "model_request_failed"
    assert terminal[0]["status_code"] == 200
    assert terminal[0]["http_success"] is True
    assert terminal[0]["success"] is False
    assert terminal[0]["stream_completed"] is False
    assert terminal[0]["stream_outcome"] == "failed"
    assert terminal[0]["failure_origin"] == "model_generator"
    assert terminal[0]["error"] == {
        "type": "RuntimeError",
        "message": "backend failed",
    }


def test_first_stream_terminal_outcome_wins(monkeypatch):
    from ..streaming_outcome import (
        FailureOrigin,
        get_stream_outcome_reporter,
        report_stream_failure,
    )

    events = _enable_capture(monkeypatch)

    async def endpoint(request: Request):
        await request.json()
        reporter = get_stream_outcome_reporter(request)

        async def generate():
            error = ValueError("first failure")
            report_stream_failure(request, error, FailureOrigin.PROTOCOL)
            reporter.completed()
            report_stream_failure(
                request, RuntimeError("second failure"), FailureOrigin.SERVER
            )
            yield b"unchanged"

        return StreamingResponse(generate())

    response = TestClient(_app("/v1/chat/completions", endpoint)).post(
        "/v1/chat/completions", json={"model": "llm", "stream": True}
    )

    assert response.content == b"unchanged"
    assert events[-1]["event"] == "model_request_failed"
    assert events[-1]["failure_origin"] == "protocol"
    assert events[-1]["error"]["message"] == "first failure"


def test_client_disconnect_outcome_is_not_logged_as_success(monkeypatch):
    from ..streaming_outcome import report_client_disconnect

    events = _enable_capture(monkeypatch)

    async def endpoint(request: Request):
        await request.json()

        async def generate():
            yield b"one"
            report_client_disconnect(request)
            return

        return StreamingResponse(generate())

    response = TestClient(_app("/v1/chat/completions", endpoint)).post(
        "/v1/chat/completions", json={"model": "llm", "stream": True}
    )

    assert response.content == b"one"
    assert events[-1]["event"] == "model_request_failed"
    assert events[-1]["status_code"] == 200
    assert events[-1]["stream_outcome"] == "client_disconnected"
    assert events[-1]["failure_origin"] == "client"


@pytest.mark.asyncio
async def test_outward_stream_cancellation_is_not_classified_as_disconnect(monkeypatch):
    import asyncio
    import time

    events = _enable_capture(monkeypatch)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "scheme": "http",
        }
    )

    async def source():
        yield b"one"
        raise asyncio.CancelledError()

    wrapped = ModelRequestLoggingRoute._wrap_stream(
        source(),
        request,
        "request-id",
        "/v1/chat/completions",
        "model",
        "llm",
        200,
        time.perf_counter(),
    )
    assert await anext(wrapped) == b"one"
    with pytest.raises(asyncio.CancelledError):
        await anext(wrapped)

    assert events[-1]["event"] == "model_request_failed"
    assert events[-1]["stream_outcome"] == "cancelled"
    assert events[-1]["failure_origin"] == "server"
