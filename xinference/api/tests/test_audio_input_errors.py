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
from types import SimpleNamespace

import pytest
import xoscar as xo
from fastapi import HTTPException

from ...core.exceptions import InvalidAudioInputError
from .. import restful_api as restful_api_module
from ..restful_api import RESTfulAPI


class _InvalidAudioActor(xo.StatelessActor):
    async def transcriptions(self, **kwargs):
        raise InvalidAudioInputError("Invalid audio file: audio is empty.")


class _Request:
    def __init__(self, form):
        self._form = form
        self.state = SimpleNamespace()

    async def form(self):
        return self._form


class _UploadFile:
    def __init__(self, content: bytes):
        self._content = content

    async def read(self):
        return self._content


class _AudioModelRef:
    uid = b"audio-model-rep0"

    async def transcriptions(self, **kwargs):
        raise InvalidAudioInputError("Invalid audio file: audio is empty.")

    async def speech(self, **kwargs):
        raise InvalidAudioInputError(
            "Invalid prompt_speech: no detectable audio was found."
        )


class _API:
    def _set_trace_model(self, model_uid):
        self.model_uid = model_uid

    def _set_trace_model_type(self, model_type):
        self.model_type = model_type

    def _check_model_access(self, request, model_uid, model_type):
        self.access = (model_uid, model_type)

    async def _get_supervisor_ref(self):
        raise AssertionError("require_model is patched in this test")

    async def _report_error_event(self, *args):
        raise AssertionError("client input errors must not be reported as model errors")


@pytest.fixture
def invalid_audio_api(monkeypatch):
    model_ref = _AudioModelRef()

    async def fake_require_model(*args):
        return model_ref

    monkeypatch.setattr(restful_api_module, "require_model", fake_require_model)
    return _API()


@pytest.mark.asyncio
async def test_transcription_invalid_audio_returns_400(invalid_audio_api):
    with pytest.raises(HTTPException) as exc_info:
        await RESTfulAPI.create_transcriptions(
            invalid_audio_api,
            _Request({}),
            model="funasr-test",
            file=_UploadFile(b""),
            language=None,
            prompt=None,
            response_format="json",
            temperature=0,
            kwargs=None,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid audio file: audio is empty."


@pytest.mark.asyncio
async def test_speech_invalid_prompt_audio_returns_400(invalid_audio_api):
    request = _Request(
        {
            "model": "megatts-test",
            "input": "test",
            "voice": "",
            "response_format": "mp3",
            "speed": 1.0,
            "stream": False,
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        await RESTfulAPI.create_speech(
            invalid_audio_api,
            request,
            prompt_speech=_UploadFile(b"silent"),
            prompt_latent=_UploadFile(b"latent"),
        )

    assert exc_info.value.status_code == 400
    assert (
        exc_info.value.detail == "Invalid prompt_speech: no detectable audio was found."
    )


@pytest.mark.asyncio
async def test_invalid_audio_detail_is_stable_across_actor_boundary(monkeypatch):
    pool = await asyncio.wait_for(
        xo.create_actor_pool(f"127.0.0.1:{xo.utils.get_next_port()}", n_process=1),
        timeout=60,
    )
    async with pool:
        model_ref = await asyncio.wait_for(
            xo.create_actor(
                _InvalidAudioActor,
                address=pool.external_address,
                allocate_strategy=xo.allocate_strategy.ProcessIndex(1),
                uid="invalid-audio-model",
            ),
            timeout=60,
        )

        async def fake_require_model(*args):
            return model_ref

        monkeypatch.setattr(restful_api_module, "require_model", fake_require_model)
        api = _API()

        with pytest.raises(HTTPException) as exc_info:
            await RESTfulAPI.create_transcriptions(
                api,
                _Request({}),
                model="funasr-test",
                file=_UploadFile(b""),
                language=None,
                prompt=None,
                response_format="json",
                temperature=0,
                kwargs=None,
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Invalid audio file: audio is empty."
        assert "address=" not in exc_info.value.detail
        assert "pid=" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_speech_stream_failure_reports_model_origin_and_runs_cleanup(monkeypatch):
    from ..streaming_outcome import (
        FailureOrigin,
        StreamState,
        get_stream_outcome_reporter,
    )

    calls = {"decrease": 0}

    class StreamingAudioModel:
        uid = b"audio-model-rep0"

        async def speech(self, **kwargs):
            async def chunks():
                yield b"audio-one"
                raise RuntimeError("audio backend failed")

            return chunks()

        async def decrease_serve_count(self):
            calls["decrease"] += 1

    async def fake_require_model(*args):
        return StreamingAudioModel()

    monkeypatch.setattr(restful_api_module, "require_model", fake_require_model)
    request = _Request(
        {
            "model": "megatts-test",
            "input": "test",
            "voice": "",
            "response_format": "mp3",
            "speed": 1.0,
            "stream": True,
        }
    )
    response = await RESTfulAPI.create_speech(
        _API(),
        request,
        prompt_speech=_UploadFile(b"prompt"),
        prompt_latent=None,
    )

    chunks = []
    with pytest.raises(RuntimeError, match="audio backend failed"):
        async for chunk in response.body_iterator:
            chunks.append(chunk)

    outcome = get_stream_outcome_reporter(request).outcome
    assert chunks
    assert outcome.state is StreamState.FAILED
    assert outcome.failure_origin is FailureOrigin.MODEL_GENERATOR
    assert outcome.error_message == "audio backend failed"
    assert calls["decrease"] == 1
