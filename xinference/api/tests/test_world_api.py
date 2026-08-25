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
import json

import httpx
import pytest
from fastapi import APIRouter, FastAPI, HTTPException

from .. import restful_api as restful_api_module
from ..restful_api import RESTfulAPI


class _Request:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


class _WorldModelRef:
    uid = "world-actor"

    def __init__(self):
        self.kwargs = None
        self.abort_request_ids = []
        self.error = None

    async def world_generate(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return json.dumps(
            {"created": 1, "data": [{"url": "/world.mp4", "b64_json": None}]}
        )

    async def abort_request(self, request_id):
        self.abort_request_ids.append(request_id)


class _API:
    def __init__(self):
        self.model_type = None
        self.running_request_id = None
        self.reported_errors = []

    def _set_trace_model(self, model_uid):
        self.model_uid = model_uid

    def _set_trace_model_type(self, model_type):
        self.model_type = model_type

    def _check_model_access(self, request, model_uid, model_type):
        self.access = (model_uid, model_type)

    async def _get_supervisor_ref(self):
        raise AssertionError("require_model is patched in this test")

    async def _report_error_event(self, *args):
        self.reported_errors.append(args)

    def _add_running_task(self, request_id):
        self.running_request_id = request_id

    async def _get_model_last_error(self, uid, error):
        return error

    def handle_request_limit_error(self, error):
        raise error


@pytest.mark.asyncio
async def test_create_world_forwards_common_and_model_specific_config(monkeypatch):
    model_ref = _WorldModelRef()

    async def fake_require_model(*args):
        return model_ref

    monkeypatch.setattr(restful_api_module, "require_model", fake_require_model)
    api = _API()
    response = await RESTfulAPI.create_world(
        api,
        _Request(
            {
                "model": "world-uid",
                "prompt": "move forward",
                "image": "data:image/png;base64,aW1hZ2U=",
                "generation_config": {"num_frames": 97, "request_id": "config-id"},
                "extra_body": {"pose": "w-4", "request_id": "kwargs-id"},
            }
        ),
    )

    assert json.loads(response.body) == {
        "created": 1,
        "data": [{"url": "/world.mp4", "b64_json": None}],
    }
    assert api.model_type == "world"
    assert api.running_request_id == "kwargs-id"
    assert model_ref.kwargs == {
        "prompt": "move forward",
        "image": "data:image/png;base64,aW1hZ2U=",
        "video": None,
        "generation_config": {"num_frames": 97},
        "model_kwargs": {"pose": "w-4"},
        "request_id": "kwargs-id",
    }


@pytest.mark.asyncio
async def test_create_world_rejects_image_and_video_together():
    api = _API()
    with pytest.raises(HTTPException, match="Only one of image and video") as exc:
        await RESTfulAPI.create_world(
            api,
            _Request(
                {
                    "model": "world-uid",
                    "prompt": "move forward",
                    "image": "image",
                    "video": "video",
                }
            ),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image", "/etc/passwd"),
        ("image", "http://127.0.0.1/private.png"),
        ("video", "https://example.com/video.mp4"),
    ],
)
async def test_create_world_rejects_public_paths_and_remote_urls(field, value):
    api = _API()
    with pytest.raises(HTTPException, match="must be a base64 data URL") as exc:
        await RESTfulAPI.create_world(
            api,
            _Request(
                {
                    "model": "world-uid",
                    "prompt": "move forward",
                    field: value,
                }
            ),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_create_world_maps_model_validation_errors_to_400(monkeypatch):
    model_ref = _WorldModelRef()
    model_ref.error = ValueError("unsupported world option")

    async def fake_require_model(*args):
        return model_ref

    monkeypatch.setattr(restful_api_module, "require_model", fake_require_model)
    api = _API()
    with pytest.raises(HTTPException, match="unsupported world option") as exc:
        await RESTfulAPI.create_world(
            api,
            _Request({"model": "world-uid", "prompt": "move forward"}),
        )
    assert exc.value.status_code == 400
    assert api.reported_errors == [("world-uid", "unsupported world option")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "content_type"),
    [(b"{", "application/json"), (b'{"prompt": 1}', "application/json")],
)
async def test_create_world_route_rejects_malformed_body(content, content_type):
    api = RESTfulAPI.__new__(RESTfulAPI)
    app = FastAPI()
    router = APIRouter()
    router.add_api_route("/v1/worlds/generations", api.create_world, methods=["POST"])
    app.include_router(router)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/worlds/generations",
            content=content,
            headers={"content-type": content_type},
        )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Invalid request body:")


@pytest.mark.asyncio
async def test_create_world_stops_runner_when_request_is_cancelled(monkeypatch):
    model_ref = _WorldModelRef()
    model_ref.error = asyncio.CancelledError()

    async def fake_require_model(*args):
        return model_ref

    monkeypatch.setattr(restful_api_module, "require_model", fake_require_model)
    api = _API()
    with pytest.raises(HTTPException, match="cancelled: cancel-me") as exc:
        await RESTfulAPI.create_world(
            api,
            _Request(
                {
                    "model": "world-uid",
                    "prompt": "move forward",
                    "extra_body": {"request_id": "cancel-me"},
                }
            ),
        )
    assert exc.value.status_code == 409
    assert model_ref.abort_request_ids == ["cancel-me"]
