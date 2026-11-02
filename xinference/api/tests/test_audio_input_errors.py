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

import asyncio
from contextlib import aclosing
from types import SimpleNamespace
from typing import cast

import pytest
import xoscar as xo
from fastapi import HTTPException
from starlette.datastructures import FormData

from ...core.exceptions import InvalidAudioInputError
from ...core.model import ModelActor
from ...model.audio.native_async import NativeAsyncAudioModel
from .. import restful_api as restful_api_module
from ..restful_api import RESTfulAPI


class _NativeSpeechModel(NativeAsyncAudioModel):
    def __init__(self):
        super().__init__(
            "native",
            ".",
            SimpleNamespace(to_description=lambda: {"model_type": "audio"}),
        )
        self.started = asyncio.Event()
        self.next_waiting = asyncio.Event()
        self.aborting = asyncio.Event()
        self.abort_ack = asyncio.Event()
        self.released = asyncio.Event()

    async def _load(self):
        self._engine = SimpleNamespace(
            generate=self.generate, abort=self.abort, shutdown=lambda: None
        )

    async def generate(self, **kwargs):
        self.started.set()
        yield b"first-audio"
        self.next_waiting.set()
        await self.released.wait()

    async def abort(self, request_id):
        self.aborting.set()
        await self.abort_ack.wait()
        self.released.set()

    def speech(self, *args, **kwargs):
        raise AssertionError("async entrypoint required")

    async def async_speech(self, *args, **kwargs):
        async def chunks():
            async with aclosing(self._generate({})) as outputs:
                async for audio in outputs:
                    yield audio

        return chunks()


class _NativeSpeechActor(ModelActor):
    def __init__(self):
        super().__init__(
            "127.0.0.1:1",
            "127.0.0.1:2",
            _NativeSpeechModel(),
            "native-0",
            request_limits=1,
        )

    async def __post_create__(self):
        pass

    async def record_metrics(self, *args, **kwargs):
        pass

    async def prepare(self):
        await self.load()
        await self.wait_for_load()

    def observations(self):
        return {
            "serve_count": self._serve_count,
            "aborting": self._model.aborting.is_set(),
            "next_waiting": self._model.next_waiting.is_set(),
            "requests": len(self._model._requests),
        }

    def allow_abort(self):
        self._model.abort_ack.set()
        self._model.released.set()


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


async def test_speech_cleanup_releases_admission_when_remote_close_fails():
    from unittest.mock import AsyncMock

    model = SimpleNamespace(decrease_serve_count=AsyncMock())
    iterator = SimpleNamespace(
        aclose=AsyncMock(side_effect=RuntimeError("abort failed"))
    )
    with pytest.raises(RuntimeError, match="abort failed"):
        await restful_api_module._close_speech_stream(model, iterator)
    model.decrease_serve_count.assert_awaited_once()


@pytest.mark.parametrize("fails", [False, True])
async def test_pcm_failure_cannot_be_finalized_by_http_middleware(fails):
    from fastapi import FastAPI, Request
    from sse_starlette import EventSourceResponse

    app = FastAPI()

    @app.middleware("http")
    async def middleware(request: Request, call_next):
        response = await call_next(request)
        return restful_api_module._guard_speech_stream_failure(request, response)

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        async def audio():
            yield b"first audio"
            if fails:
                error = RuntimeError("incomplete audio")
                restful_api_module.report_stream_failure(
                    request, error, restful_api_module.FailureOrigin.MODEL_GENERATOR
                )
                raise error

        return EventSourceResponse(audio(), media_type="application/octet-stream")

    messages = []

    async def send(message):
        messages.append(message)

    async def receive():
        await asyncio.Event().wait()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/audio/speech",
        "raw_path": b"/v1/audio/speech",
        "query_string": b"",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 123),
        "headers": [],
    }
    if fails:
        with pytest.raises(Exception):
            await app(scope, receive, send)
    else:
        await app(scope, receive, send)
    assert any(message.get("body") == b"first audio" for message in messages)
    finalized = any(
        message["type"] == "http.response.body" and not message.get("more_body", False)
        for message in messages
    )
    assert finalized is not fails


@pytest.mark.parametrize("invalid", ["undecodable", "empty", "language"])
async def test_native_file_transcription_input_errors_are_400(monkeypatch, invalid):
    import io
    import wave

    from ...model.audio.vllm import VLLMQwen3ASRModel

    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"" if invalid == "empty" else b"\0" * 3200)
    content = b"invalid audio" if invalid == "undecodable" else data.getvalue()
    model = VLLMQwen3ASRModel("asr", ".", SimpleNamespace())
    actor = SimpleNamespace(transcriptions=model.async_transcriptions)
    monkeypatch.setattr(
        restful_api_module,
        "require_model",
        lambda *args: asyncio.sleep(0, result=actor),
    )
    with pytest.raises(HTTPException) as exc:
        await RESTfulAPI.create_transcriptions(
            _API(),
            _Request(FormData()),
            model="asr",
            file=_UploadFile(content),
            language="unsupported" if invalid == "language" else None,
            prompt=None,
            response_format="json",
            temperature=0,
            kwargs=None,
        )
    assert exc.value.status_code == 400


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
            _Request(FormData()),
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
                _Request(FormData()),
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


@pytest.mark.parametrize("disconnect_during_send", [False, True])
async def test_native_tts_http_disconnect_holds_slot_until_remote_abort(
    monkeypatch, disconnect_during_send
):
    import json

    from fastapi import FastAPI, Request

    pool = await xo.create_actor_pool(
        f"127.0.0.1:{xo.utils.get_next_port()}", n_process=1
    )
    async with pool:
        actor = await xo.create_actor(
            _NativeSpeechActor,
            address=pool.external_address,
            allocate_strategy=xo.allocate_strategy.ProcessIndex(1),
            uid="native-speech",
        )
        await actor.prepare()
        monkeypatch.setattr(
            restful_api_module,
            "require_model",
            lambda *args: asyncio.sleep(0, result=actor),
        )
        app = FastAPI()

        @app.post("/v1/audio/speech")
        async def speech(request: Request):
            return await RESTfulAPI.create_speech(
                cast(RESTfulAPI, _API()), request, None, None
            )

        first, disconnected = asyncio.Event(), asyncio.Event()
        body = json.dumps(
            {
                "model": "native",
                "input": "hello",
                "voice": "ryan",
                "response_format": "pcm",
                "stream": True,
            }
        ).encode()
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first.set()
                if disconnect_during_send:
                    await asyncio.Event().wait()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/v1/audio/speech",
            "raw_path": b"/v1/audio/speech",
            "query_string": b"",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("test", 123),
            "headers": [(b"content-type", b"application/json")],
        }
        response_task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(first.wait(), 5)
            if not disconnect_during_send:
                for _ in range(100):
                    if (await actor.observations())["next_waiting"]:
                        break
                    await asyncio.sleep(0.01)
                assert (await actor.observations())["next_waiting"]
            disconnected.set()
            for _ in range(100):
                state = await actor.observations()
                if state["aborting"] or state["serve_count"] == 0:
                    break
                await asyncio.sleep(0.01)
            assert state["aborting"] and state["serve_count"] == 1
            with pytest.raises(RuntimeError, match="Rate limit"):
                await actor.speech("next", "ryan", "pcm", 1.0, True)
            await actor.allow_abort()
            await asyncio.wait_for(response_task, 5)
            for _ in range(100):
                state = await actor.observations()
                if state["serve_count"] == 0 and state["requests"] == 0:
                    break
                await asyncio.sleep(0.01)
            assert state["serve_count"] == state["requests"] == 0
            import httpx

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                result = await asyncio.wait_for(
                    client.post(
                        "/v1/audio/speech",
                        content=body,
                        headers={"content-type": "application/json"},
                    ),
                    5,
                )
            assert result.status_code == 200 and result.content
            state = await actor.observations()
            assert state["serve_count"] == state["requests"] == 0
        finally:
            disconnected.set()
            await actor.allow_abort()
            await asyncio.wait_for(
                asyncio.gather(response_task, return_exceptions=True), 5
            )
            await xo.destroy_actor(actor)


@pytest.mark.parametrize("voice", ["omitted", None])
async def test_native_tts_default_voice_at_http_boundary(monkeypatch, voice):
    import sys

    import httpx
    import numpy as np
    from fastapi import FastAPI, Request

    from ...model.audio.vllm_omni import VLLMOmniTTSModel

    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(RequestOutputKind=SimpleNamespace(DELTA="delta")),
    )
    prompts = []

    async def generate(*, prompt, **kwargs):
        prompts.append(prompt)
        yield SimpleNamespace(multimodal_output={"audio": np.array([0.1]), "sr": 24000})

    model = VLLMOmniTTSModel("tts", ".", SimpleNamespace())
    model._samples_per_token = 1920
    model._engine = SimpleNamespace(
        default_sampling_params_list=[SimpleNamespace(output_kind="cumulative")],
        generate=generate,
    )
    model._config = SimpleNamespace(
        spk_id={"ryan": 0}, codec_language_id={"english": 1}, spk_is_dialect={}
    )
    model._tokenizer = lambda *args, **kwargs: {"input_ids": [1]}
    model._estimate_prompt = lambda **kwargs: 8
    actor = SimpleNamespace(speech=model.async_speech)
    monkeypatch.setattr(
        restful_api_module,
        "require_model",
        lambda *args: asyncio.sleep(0, result=actor),
    )
    app = FastAPI()

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        return await RESTfulAPI.create_speech(
            cast(RESTfulAPI, _API()), request, None, None
        )

    body = {"model": "tts", "input": "Hello", "response_format": "pcm"}
    if voice != "omitted":
        body["voice"] = voice
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/audio/speech", json=body)
    assert response.status_code == 200 and response.content
    assert prompts[0]["additional_information"]["speaker"] == ["ryan"]
