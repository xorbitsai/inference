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

# CI: linux

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from ...model.audio.whisper import WhisperModel
from .. import restful_api as restful_api_module
from ..restful_api import RESTfulAPI

PATHS = {
    "transcriptions": "/v1/audio/transcriptions",
    "translations": "/v1/audio/translations",
}


async def _whisper_check(**kwargs):
    # Whisper's own validation of the value it receives; no weights are loaded.
    return json.dumps(
        WhisperModel._pipeline_kwargs(
            {},
            kwargs["response_format"],
            kwargs["temperature"],
            kwargs["timestamp_granularities"],
        )
    )


@pytest.fixture
def audio_client(monkeypatch):
    model = SimpleNamespace(uid=b"whisper-rep0")
    for method in PATHS:
        setattr(model, method, AsyncMock(side_effect=_whisper_check))
    monkeypatch.setattr(
        restful_api_module, "require_model", AsyncMock(return_value=model)
    )
    api = object.__new__(RESTfulAPI)
    api._set_trace_model = Mock()
    api._set_trace_model_type = Mock()
    api._check_model_access = Mock()
    api._get_supervisor_ref = AsyncMock()
    api._report_error_event = AsyncMock()
    app = FastAPI()
    for method, path in PATHS.items():
        app.add_api_route(path, getattr(api, f"create_{method}"), methods=["POST"])
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    return SimpleNamespace(client=client, model=model)


async def _post(client, method, granularities, response_format):
    data = {"model": "whisper", "response_format": response_format}
    if granularities is not None:
        # A list becomes one repeated form field per item, as OpenAI clients send it.
        data["timestamp_granularities[]"] = granularities
    async with client:
        return await client.post(
            PATHS[method], data=data, files={"file": ("audio.wav", b"RIFF")}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", PATHS)
@pytest.mark.parametrize(
    "sent",
    [
        None,
        ["word"],
        ["segment"],
        ["word", "segment"],
        ["segment", "word"],
        ["word", "word"],
        [""],
    ],
    ids=[
        "absent",
        "word",
        "segment",
        "word+segment",
        "segment+word",
        "duplicate-word",
        "empty",
    ],
)
async def test_every_timestamp_granularity_reaches_the_model(
    audio_client, method, sent
):
    response = await _post(audio_client.client, method, sent, "json")

    assert response.status_code == 200, response.text
    call = getattr(audio_client.model, method).call_args
    assert call.kwargs["timestamp_granularities"] == sent


@pytest.mark.asyncio
@pytest.mark.parametrize("method", PATHS)
@pytest.mark.parametrize(
    "sent,return_timestamps",
    [(None, True), (["segment"], True), (["word"], "word")],
    ids=["absent", "segment", "word"],
)
async def test_single_granularity_selects_the_same_timestamps(
    audio_client, method, sent, return_timestamps
):
    response = await _post(audio_client.client, method, sent, "verbose_json")

    assert response.status_code == 200, response.text
    assert response.json()["return_timestamps"] == return_timestamps


@pytest.mark.asyncio
@pytest.mark.parametrize("method", PATHS)
@pytest.mark.parametrize(
    "sent",
    [["word", "segment"], ["segment", "word"], ["word", "word"], [""]],
    ids=["word+segment", "segment+word", "duplicate-word", "empty"],
)
async def test_unsupported_whisper_granularities_are_rejected(
    audio_client, method, sent
):
    response = await _post(audio_client.client, method, sent, "verbose_json")

    assert response.status_code == 500
    assert response.json() == {"detail": f"Unsupported timestamp_granularities: {sent}"}
