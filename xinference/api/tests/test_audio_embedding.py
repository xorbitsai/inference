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

from io import BytesIO
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile

from xinference.api.routers import audio


def test_audio_embedding_route_is_registered():
    captured = {}

    def add_api_route(path, endpoint, methods=None, **kwargs):
        captured[(path, tuple(methods or []))] = endpoint

    api = MagicMock()
    api._router.add_api_route.side_effect = add_api_route
    api.is_authenticated.return_value = False

    audio.register_routes(api)

    assert captured[("/v1/audio/embeddings", ("POST",))] is api.create_audio_embedding


@pytest.mark.asyncio
async def test_create_audio_embedding_forwards_audio_and_model_uid():
    from xinference.api import restful_api

    api = MagicMock()
    api._get_supervisor_ref = MagicMock()
    api._report_error_event = AsyncMock()
    model_ref = MagicMock(uid="replica-uid")
    model_ref.create_audio_embedding = AsyncMock(
        return_value=(
            '{"object":"embedding","model":"speaker-model",'
            '"dimensions":3,"embedding":[0.1,0.2,0.3]}'
        )
    )
    upload = UploadFile(filename="speaker.wav", file=BytesIO(b"encoded-audio"))

    with patch.object(restful_api, "require_model", AsyncMock(return_value=model_ref)):
        response = await restful_api.RESTfulAPI.create_audio_embedding(
            api,
            request=MagicMock(),
            model="speaker-model",
            file=upload,
        )

    assert response.media_type == "application/json"
    assert b'"dimensions":3' in response.body
    model_ref.create_audio_embedding.assert_awaited_once_with(
        audio=b"encoded-audio", model_uid="speaker-model"
    )
    api._set_trace_model.assert_called_once_with("speaker-model")
    api._set_trace_model_type.assert_called_once_with("audio")
    api._check_model_access.assert_called_once_with(ANY, "speaker-model", "audio")
