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

"""Regression tests for the /v1/images/ocr response contract.

ModelActor.ocr always serializes the model's return value to JSON bytes
(_call_wrapper_json), and both REST clients unconditionally parse the body
with response.json(). The endpoint must therefore declare application/json:
with text/plain, aiohttp's content-type check makes the async client raise
ContentTypeError on perfectly normal OCR text.

These tests run without deepdoc-lib installed: the model actor is replaced
by a stub that runs the real DeepDocModel (onnx internals mocked) and
serializes its output exactly like ModelActor does. The full HTTP round
trip — endpoint, real uvicorn server, real sync/async clients — is real.
"""

import asyncio
import io
import threading
import time
import weakref
from unittest.mock import AsyncMock, MagicMock

import pytest
import uvicorn
from fastapi import APIRouter, FastAPI

from ...core.utils import json_dumps
from ..restful_api import RESTfulAPI

MODEL_UID = "deepdoc_contract_test"


def _make_deepdoc_model():
    from ...model.image.ocr.deepdoc import DeepDocModel

    model = DeepDocModel(
        model_uid=MODEL_UID,
        model_path="/tmp/unused",
        model_spec=MagicMock(model_ability=["ocr"]),
    )
    # one detected line as (box, (text, score)) — bypasses the real onnx models
    model._ocr = MagicMock(
        return_value=[([[0, 0], [1, 0], [1, 1], [0, 1]], ("hello world", 0.99))]
    )
    layout = MagicMock()
    layout.forward.return_value = [[{"type": "text"}]]
    model._layout_recognizer = layout
    return model


class _FakeModelRef:
    """Mimics ModelActor.ocr: runs the real adapter, then serializes the
    result to JSON bytes exactly like _call_wrapper_json does."""

    uid = MODEL_UID

    def __init__(self, model):
        self._model = model

    async def ocr(self, image, **kwargs):
        return json_dumps(self._model.ocr(image, **kwargs))


@pytest.fixture
def endpoint():
    # Skip RESTfulAPI.__init__: no supervisor, auth or on-disk stores are
    # needed to exercise create_ocr (same pattern as test_auth_advanced_toggle).
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._advanced_auth_service = None
    api._auth_service = None
    api._uid_to_model_name = {}
    api._running_tasks = weakref.WeakValueDictionary()

    supervisor = MagicMock()
    supervisor.get_model = AsyncMock(return_value=_FakeModelRef(_make_deepdoc_model()))

    async def _get_supervisor_ref():
        return supervisor

    api._get_supervisor_ref = _get_supervisor_ref
    api._report_error_event = AsyncMock()

    app = FastAPI()
    router = APIRouter()
    router.add_api_route("/v1/images/ocr", api.create_ocr, methods=["POST"])
    app.include_router(router)

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        assert time.time() < deadline, "test server failed to start"
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def _png_bytes() -> bytes:
    from PIL import Image

    bio = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(bio, format="PNG")
    return bio.getvalue()


def test_response_body_is_json(endpoint):
    import requests

    files = [
        ("model", (None, MODEL_UID)),
        ("kwargs", (None, "{}")),
        ("image", ("image", _png_bytes(), "application/octet-stream")),
    ]
    response = requests.post(f"{endpoint}/v1/images/ocr", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    # plain OCR text arrives as a JSON-encoded string, not raw text
    assert response.json() == "hello world"


def test_sync_client_plain_text(endpoint):
    from ...client.restful.restful_client import RESTfulImageModelHandle

    handle = RESTfulImageModelHandle(MODEL_UID, endpoint, auth_headers={})
    try:
        assert handle.ocr(image=_png_bytes()) == "hello world"
    finally:
        handle.close()


def test_sync_client_structured_results(endpoint):
    from ...client.restful.restful_client import RESTfulImageModelHandle

    handle = RESTfulImageModelHandle(MODEL_UID, endpoint, auth_headers={})
    try:
        payload = handle.ocr(image=_png_bytes(), return_dict=True)
        assert payload["task"] == "ocr"
        assert payload["lines"][0]["text"] == "hello world"
        assert payload["lines"][0]["score"] == pytest.approx(0.99)

        payload = handle.ocr(image=_png_bytes(), task="layout")
        assert payload == {"task": "layout", "layouts": [{"type": "text"}]}
    finally:
        handle.close()


def test_async_client_plain_text(endpoint):
    """aiohttp's response.json() rejects non-JSON content types, so this
    fails with ContentTypeError if the endpoint declares text/plain."""
    from ...client.restful.async_restful_client import AsyncRESTfulImageModelHandle

    async def run():
        handle = AsyncRESTfulImageModelHandle(MODEL_UID, endpoint, auth_headers={})
        try:
            return await handle.ocr(image=_png_bytes())
        finally:
            await handle.close()

    assert asyncio.run(run()) == "hello world"


def test_async_client_structured_results(endpoint):
    from ...client.restful.async_restful_client import AsyncRESTfulImageModelHandle

    async def run():
        handle = AsyncRESTfulImageModelHandle(MODEL_UID, endpoint, auth_headers={})
        try:
            return await handle.ocr(image=_png_bytes(), task="layout")
        finally:
            await handle.close()

    assert asyncio.run(run()) == {"task": "layout", "layouts": [{"type": "text"}]}
