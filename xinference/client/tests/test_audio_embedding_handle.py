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

from xinference.client.restful.async_restful_client import AsyncRESTfulAudioModelHandle
from xinference.client.restful.restful_client import RESTfulAudioModelHandle


def test_sync_audio_handle_create_embedding():
    handle = RESTfulAudioModelHandle.__new__(RESTfulAudioModelHandle)
    handle._model_uid = "speaker-model"
    handle._base_url = "http://127.0.0.1:9997"
    handle.auth_headers = {"Authorization": "Bearer token"}
    handle.session = MagicMock()
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "object": "embedding",
        "model": "speaker-model",
        "dimensions": 3,
        "embedding": [0.1, 0.2, 0.3],
    }
    handle.session.post.return_value = response

    result = handle.create_embedding(b"encoded-audio")

    assert result["embedding"] == [0.1, 0.2, 0.3]
    _, kwargs = handle.session.post.call_args
    assert kwargs["data"] == {"model": "speaker-model"}
    assert kwargs["files"][0][1][1] == b"encoded-audio"


@pytest.mark.asyncio
async def test_async_audio_handle_create_embedding():
    handle = AsyncRESTfulAudioModelHandle.__new__(AsyncRESTfulAudioModelHandle)
    handle._model_uid = "speaker-model"
    handle._base_url = "http://127.0.0.1:9997"
    handle.auth_headers = {"Authorization": "Bearer token"}
    handle.session = MagicMock()
    response = MagicMock(status=200)
    response.json = AsyncMock(
        return_value={
            "object": "embedding",
            "model": "speaker-model",
            "dimensions": 3,
            "embedding": [0.1, 0.2, 0.3],
        }
    )
    response.wait_for_close = AsyncMock()
    handle.session.post = AsyncMock(return_value=response)

    result = await handle.create_embedding(b"encoded-audio")

    assert result["dimensions"] == 3
    response.release.assert_called_once()
    response.wait_for_close.assert_awaited_once()
