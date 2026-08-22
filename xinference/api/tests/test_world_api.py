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

import pytest
from fastapi import HTTPException

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

    async def world_generate(self, **kwargs):
        self.kwargs = kwargs
        return json.dumps(
            {"created": 1, "data": [{"url": "/world.mp4", "b64_json": None}]}
        )


class _API:
    def __init__(self):
        self.model_type = None
        self.running_request_id = None

    def _set_trace_model(self, model_uid):
        self.model_uid = model_uid

    def _set_trace_model_type(self, model_type):
        self.model_type = model_type

    def _check_model_access(self, request, model_uid, model_type):
        self.access = (model_uid, model_type)

    async def _get_supervisor_ref(self):
        raise AssertionError("require_model is patched in this test")

    async def _report_error_event(self, *args):
        raise AssertionError("no error is expected")

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
