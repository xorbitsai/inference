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

from unittest.mock import AsyncMock, MagicMock

import pytest

from ..restful.async_restful_client import AsyncRESTfulWorldModelHandle
from ..restful.restful_client import RESTfulWorldModelHandle


def test_sync_world_handle_maps_model_kwargs_to_extra_body():
    handle = RESTfulWorldModelHandle.__new__(RESTfulWorldModelHandle)
    handle._model_uid = "world-model"
    handle._base_url = "http://127.0.0.1:9997"
    handle.auth_headers = {"Authorization": "Bearer token"}
    handle.session = MagicMock()
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "created": 1,
        "data": [{"url": "/world.mp4", "b64_json": None}],
    }
    handle.session.post.return_value = response

    result = handle.generate(
        "move forward",
        image=b"image",
        generation_config={"num_frames": 97},
        pose="w-4",
    )

    assert result["data"][0]["url"] == "/world.mp4"
    _, kwargs = handle.session.post.call_args
    assert kwargs["json"] == {
        "model": "world-model",
        "prompt": "move forward",
        "image": "data:image/png;base64,aW1hZ2U=",
        "video": None,
        "generation_config": {"num_frames": 97},
        "extra_body": {"pose": "w-4"},
    }


@pytest.mark.asyncio
async def test_async_world_handle_maps_video_bytes():
    handle = AsyncRESTfulWorldModelHandle.__new__(AsyncRESTfulWorldModelHandle)
    handle._model_uid = "world-model"
    handle._base_url = "http://127.0.0.1:9997"
    handle.auth_headers = {}
    handle.timeout = object()
    handle.session = MagicMock()
    response = MagicMock(status=200)
    response.json = AsyncMock(
        return_value={
            "created": 1,
            "data": [{"url": None, "b64_json": "dmlkZW8="}],
        }
    )
    response.wait_for_close = AsyncMock()
    handle.session.post = AsyncMock(return_value=response)

    result = await handle.generate("explore", video=b"video")

    assert result["data"][0]["b64_json"] == "dmlkZW8="
    _, kwargs = handle.session.post.call_args
    assert kwargs["json"]["video"] == "data:video/mp4;base64,dmlkZW8="
    assert kwargs["timeout"] is handle.timeout
    response.release.assert_called_once()
    response.wait_for_close.assert_awaited_once()


def test_sync_world_handle_rejects_two_media_inputs():
    handle = RESTfulWorldModelHandle.__new__(RESTfulWorldModelHandle)
    handle.session = None
    with pytest.raises(ValueError, match="Only one of image and video"):
        handle.generate("explore", image=b"image", video=b"video")


@pytest.mark.asyncio
async def test_async_world_handle_rejects_two_media_inputs():
    handle = AsyncRESTfulWorldModelHandle.__new__(AsyncRESTfulWorldModelHandle)
    handle.session = None
    with pytest.raises(ValueError, match="Only one of image and video"):
        await handle.generate("explore", image=b"image", video=b"video")
