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

"""Exercise non-LLM request validation without loading model weights."""

# CI: linux

import asyncio
import io
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import xoscar as xo
from fastapi import FastAPI
from PIL import Image

from ...core.model import ModelActor
from ...model.audio.ace_step import AceStepModel
from ...model.audio.melotts import MeloTTSModel
from ...model.video.diffusers import DiffusersVideoModel
from ..restful_api import RESTfulAPI

JSON_ENDPOINTS = [
    (
        "/v1/embeddings",
        "create_embedding",
        {"model": "test", "input": "hello"},
        "input",
    ),
    (
        "/v1/convert_ids_to_tokens",
        "convert_ids_to_tokens",
        {"model": "test", "input": [1]},
        "input",
    ),
    (
        "/v1/rerank",
        "rerank",
        {"model": "test", "query": "hello", "documents": ["hello"]},
        "documents",
    ),
    (
        "/v1/images/generations",
        "create_images",
        {"model": "test", "prompt": "hello"},
        "prompt",
    ),
    (
        "/v1/video/generations",
        "create_videos",
        {"model": "test", "prompt": "hello"},
        "prompt",
    ),
    ("/v1/audio/speech", "create_speech", {"model": "test", "input": "hello"}, "input"),
]
# Embeddings accept extra options as top-level fields, without a serialized
# kwargs parameter. Keep that existing forwarding contract separate.
JSON_KWARGS_ENDPOINTS = JSON_ENDPOINTS[2:]
FORM_ENDPOINTS = [
    ("/v1/audio/transcriptions", "create_transcriptions", ("file",)),
    ("/v1/audio/translations", "create_translations", ("file",)),
    ("/v1/audio/speech", "create_speech", ("prompt_speech",)),
    ("/v1/images/variations", "create_variations", ("image",)),
    ("/v1/images/inpainting", "create_inpainting", ("image", "mask_image")),
    ("/v1/images/ocr", "create_ocr", ("image",)),
    ("/v1/images/docanalyze", "create_doc_analyze", ("file",)),
    ("/v1/video/generations/image", "create_videos_from_images", ("image",)),
    (
        "/v1/video/generations/flf",
        "create_videos_from_first_last_frame",
        ("first_frame", "last_frame"),
    ),
]
INVALID_KWARGS = ["{", "", "[]", "null", "1", '"text"', "true"]
MODEL_METHODS = [
    "create_embedding",
    "convert_ids_to_tokens",
    "rerank",
    "text_to_image",
    "text_to_video",
    "speech",
    "transcriptions",
    "translations",
    "image_to_image",
    "inpainting",
    "ocr",
    "docanalyze",
    "image_to_video",
    "flf_to_video",
]


@pytest.fixture
def api_client(monkeypatch):
    model = SimpleNamespace(uid=b"test-rep0")
    for method in MODEL_METHODS:
        setattr(model, method, AsyncMock(return_value=b'{"data": []}'))
    require_model = AsyncMock(return_value=model)
    monkeypatch.setattr("xinference.api.restful_api.require_model", require_model)
    api = object.__new__(RESTfulAPI)
    api._set_trace_model = Mock()
    api._set_trace_model_type = Mock()
    api._check_model_access = Mock()
    api._get_supervisor_ref = AsyncMock()
    api._report_error_event = AsyncMock()
    api._add_running_task = Mock()
    app = FastAPI()
    for path, handler in {
        (path, handler) for path, handler, *_ in JSON_ENDPOINTS + FORM_ENDPOINTS
    }:
        app.add_api_route(path, getattr(api, handler), methods=["POST"])
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )
    return SimpleNamespace(
        client=client, model=model, api=api, require_model=require_model
    )


def _assert_client_error(response, api_client):
    assert response.status_code == 400, response.text
    assert isinstance(response.json()["detail"], str)
    api_client.api._report_error_event.assert_not_awaited()
    for method in MODEL_METHODS:
        getattr(api_client.model, method).assert_not_awaited()


def _uploads(names):
    image = io.BytesIO()
    Image.new("RGB", (1, 1)).save(image, format="PNG")
    return [(name, ("test.png", image.getvalue(), "image/png")) for name in names]


@pytest.mark.asyncio
@pytest.mark.parametrize("path,handler,payload,required", JSON_ENDPOINTS)
@pytest.mark.parametrize("invalid", ["missing", "invalid", "json", "non-object"])
async def test_invalid_outer_request(
    api_client, path, handler, payload, required, invalid
):
    body = dict(payload)
    if invalid == "missing":
        body.pop(required)
    elif invalid == "invalid":
        body[required] = None
    content = {"json": b"{", "non-object": b"null"}.get(invalid)
    async with api_client.client as client:
        if content is None:
            response = await client.post(path, json=body)
        else:
            response = await client.post(
                path, content=content, headers={"content-type": "application/json"}
            )
    _assert_client_error(response, api_client)
    api_client.require_model.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("path,handler,payload,required", JSON_KWARGS_ENDPOINTS)
@pytest.mark.parametrize("kwargs", [value for value in INVALID_KWARGS if value])
async def test_invalid_json_kwargs(
    api_client, path, handler, payload, required, kwargs
):
    async with api_client.client as client:
        response = await client.post(path, json={**payload, "kwargs": kwargs})
    _assert_client_error(response, api_client)
    message = (
        "kwargs must be a valid JSON object"
        if kwargs == "{"
        else "kwargs must be a JSON object"
    )
    assert response.json() == {"detail": message}


@pytest.mark.asyncio
@pytest.mark.parametrize("path,handler,payload,required", JSON_KWARGS_ENDPOINTS)
async def test_empty_json_kwargs_preserve_route_behavior(
    api_client, path, handler, payload, required
):
    async with api_client.client as client:
        response = await client.post(path, json={**payload, "kwargs": ""})
    if handler in ("create_images", "create_videos"):
        assert response.status_code == 200, response.text
        method = "text_to_image" if handler == "create_images" else "text_to_video"
        getattr(api_client.model, method).assert_awaited_once()
        assert "kwargs" not in getattr(api_client.model, method).call_args.kwargs
    else:
        _assert_client_error(response, api_client)
        assert response.json() == {"detail": "kwargs must be a valid JSON object"}


@pytest.mark.asyncio
@pytest.mark.parametrize("path,handler,files", FORM_ENDPOINTS)
@pytest.mark.parametrize("kwargs", [value for value in INVALID_KWARGS if value])
async def test_invalid_form_kwargs(api_client, path, handler, files, kwargs):
    async with api_client.client as client:
        response = await client.post(
            path,
            data={"model": "test", "input": "hello", "kwargs": kwargs},
            files=_uploads(files),
        )
    _assert_client_error(response, api_client)
    assert response.json() == {
        "detail": (
            "kwargs must be a valid JSON object"
            if kwargs == "{"
            else "kwargs must be a JSON object"
        )
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("path,handler,payload,required", JSON_ENDPOINTS)
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
async def test_backend_errors_remain_500(
    api_client, path, handler, payload, required, error_type
):
    for method in MODEL_METHODS:
        getattr(api_client.model, method).side_effect = error_type("backend failure")
    async with api_client.client as client:
        response = await client.post(path, json=payload)
    assert response.status_code == 500
    assert response.json() == {"detail": "backend failure"}
    api_client.api._report_error_event.assert_awaited_once_with(
        "test", "backend failure"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path,handler,files", FORM_ENDPOINTS)
async def test_form_backend_value_errors_remain_500(api_client, path, handler, files):
    for method in MODEL_METHODS:
        getattr(api_client.model, method).side_effect = ValueError("backend failure")
    async with api_client.client as client:
        response = await client.post(
            path, data={"model": "test", "input": "hello"}, files=_uploads(files)
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "backend failure"}


@pytest.mark.asyncio
@pytest.mark.parametrize("path,handler,payload,required", JSON_KWARGS_ENDPOINTS)
@pytest.mark.parametrize("kwargs", [None, "{}", '{"seed": 7}'])
async def test_valid_json_kwargs(api_client, path, handler, payload, required, kwargs):
    body = {**payload, "kwargs": kwargs}
    async with api_client.client as client:
        response = await client.post(path, json=body)
    assert response.status_code == 200, response.text
    method = {
        "rerank": "rerank",
        "create_images": "text_to_image",
        "create_videos": "text_to_video",
        "create_speech": "speech",
    }[handler]
    forwarded = getattr(api_client.model, method).call_args.kwargs
    assert "kwargs" not in forwarded
    if kwargs and kwargs != "{}":
        assert forwarded["seed"] == 7


@pytest.mark.asyncio
async def test_embedding_top_level_options_are_preserved(api_client):
    async with api_client.client as client:
        response = await client.post(
            "/v1/embeddings",
            json={"model": "test", "input": "hello", "dimensions": 128},
        )
    assert response.status_code == 200
    api_client.model.create_embedding.assert_awaited_once_with(
        "hello", dimensions=128, model_uid="test"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options,message",
    [
        ({"voice": "bad"}, "only accepts `voice`"),
        ({"speed": 1.5}, "only supports `speed=1.0`"),
        ({"response_format": "nonsense"}, "supports these response formats"),
        ({"stream": True}, "only supports non-streaming generation"),
        ({"input": ""}, "requires non-empty lyrics"),
        ({"kwargs": {"instruct": ""}}, "requires a non-empty music description"),
        ({"kwargs": {"seed": True}}, "`seed` must be"),
        ({"kwargs": {"duration": 9}}, "`duration` must be"),
        ({"kwargs": {"thinking": "yes"}}, "`thinking` must be a boolean"),
        ({"kwargs": {"thinking": True}}, "LM generation options require"),
        ({"kwargs": {"unsupported": True}}, "does not support speech parameter"),
    ],
)
async def test_ace_step_validation_returns_400(api_client, options, message):
    model = AceStepModel("test", "unused", SimpleNamespace(), device="cpu")
    model._model = object()
    model._generation_params_cls = Mock()
    model._generation_config_cls = Mock()
    model._generate_music = Mock()
    api_client.model.speech.side_effect = model.speech
    body = {"model": "test", "input": "lyrics", "voice": "default", **options}
    body["kwargs"] = json.dumps({"instruct": "piano", **body.get("kwargs", {})})
    async with api_client.client as client:
        response = await client.post("/v1/audio/speech", json=body)
    assert response.status_code == 400, response.text
    assert message in response.json()["detail"]
    model._generate_music.assert_not_called()
    api_client.api._report_error_event.assert_not_awaited()


@pytest.fixture
def ace_step_actor(api_client):
    model = AceStepModel(
        "test", "unused", SimpleNamespace(to_description=lambda: {}), device="cpu"
    )
    model._model = object()
    model._generation_params_cls = Mock()
    model._generation_config_cls = Mock()
    model._generate_music = Mock()
    actor = ModelActor("test:123", "test:345", model, "test-rep0")  # type: ignore
    actor._model_state = "ready"
    actor.record_metrics = AsyncMock()
    api_client.model.speech.side_effect = actor.speech
    return SimpleNamespace(actor=actor, model=model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seed,message",
    [
        (True, "Seed must be an integer"),
        (False, "Seed must be an integer"),
        (1.5, "Seed must be an integer"),
        ("1", "Seed must be an integer"),
        (-2, "Seed must be -1 or between 0 and 2147483647"),
        (2**31, "Seed must be -1 or between 0 and 2147483647"),
    ],
)
async def test_ace_step_actor_invalid_seed_returns_400(
    api_client, ace_step_actor, seed, message
):
    async with api_client.client as client:
        response = await client.post(
            "/v1/audio/speech",
            json={
                "model": "test",
                "input": "lyrics",
                "voice": "default",
                "kwargs": json.dumps({"instruct": "piano", "seed": seed}),
            },
        )
    assert response.status_code == 400, response.text
    assert response.json() == {"detail": message}
    ace_step_actor.model._generation_params_cls.assert_not_called()
    ace_step_actor.model._generate_music.assert_not_called()
    assert ace_step_actor.actor._serve_count == 0
    api_client.api._report_error_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_ace_step_actor_backend_value_error_remains_500(
    api_client, ace_step_actor
):
    ace_step_actor.model._generate_music.side_effect = ValueError("backend failure")
    async with api_client.client as client:
        response = await client.post(
            "/v1/audio/speech",
            json={
                "model": "test",
                "input": "lyrics",
                "voice": "default",
                "kwargs": json.dumps({"instruct": "piano", "seed": 7}),
            },
        )
    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "backend failure"}
    ace_step_actor.model._generate_music.assert_called_once()
    assert ace_step_actor.actor._serve_count == 0
    api_client.api._report_error_event.assert_awaited_once_with(
        "test", "backend failure"
    )


@pytest.mark.asyncio
async def test_melo_invalid_voice_returns_400(api_client, monkeypatch):
    # Encoding is not reached for an invalid voice; no audio extras are needed.
    monkeypatch.setitem(sys.modules, "soundfile", ModuleType("soundfile"))
    model = MeloTTSModel("test", "unused", SimpleNamespace(), device="cpu")
    model._model = SimpleNamespace(
        hps=SimpleNamespace(data=SimpleNamespace(spk2id={"EN": 0})), tts_to_file=Mock()
    )
    api_client.model.speech.side_effect = model.speech
    async with api_client.client as client:
        response = await client.post(
            "/v1/audio/speech", json={"model": "test", "input": "hello", "voice": "bad"}
        )
    assert response.status_code == 400, response.text
    assert response.json() == {
        "detail": "Invalid voice: bad, available speakers: {'EN': 0}"
    }
    model._model.tts_to_file.assert_not_called()
    api_client.api._report_error_event.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,files,method",
    [
        ("/v1/video/generations", (), "text_to_video"),
        ("/v1/video/generations/image", ("image",), "image_to_video"),
        (
            "/v1/video/generations/flf",
            ("first_frame", "last_frame"),
            "firstlastframe_to_video",
        ),
    ],
)
@pytest.mark.parametrize(
    "options,message",
    [
        ({"n": 0}, "n must be at least 1"),
        ({"n": -1}, "n must be at least 1"),
        (
            {"kwargs": '{"num_inference_steps": 0}'},
            "num_inference_steps must be at least 1",
        ),
        (
            {"kwargs": '{"num_inference_steps": -1}'},
            "num_inference_steps must be at least 1",
        ),
    ],
)
@pytest.mark.parametrize("defaults", [{}, {"num_inference_steps": 0}])
async def test_minimax_h3_validation_returns_400(
    api_client, path, files, method, options, message, defaults
):
    model = DiffusersVideoModel(
        "test",
        "unused",
        SimpleNamespace(
            model_family="MiniMax-H3",
            model_ability=[],
            default_generate_config=defaults,
        ),
    )
    model._model = Mock()
    actor_method = "flf_to_video" if method == "firstlastframe_to_video" else method
    getattr(api_client.model, actor_method).side_effect = getattr(model, method)
    async with api_client.client as client:
        body = {"model": "test", "prompt": "hello", **options}
        if files:
            response = await client.post(path, data=body, files=_uploads(files))
        else:
            response = await client.post(path, json=body)
    assert response.status_code == 400, response.text
    assert response.json() == {"detail": message}
    model._model.assert_not_called()
    api_client.api._report_error_event.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,files,method",
    [
        ("/v1/video/generations", (), "text_to_video"),
        ("/v1/video/generations/image", ("image",), "image_to_video"),
        (
            "/v1/video/generations/flf",
            ("first_frame", "last_frame"),
            "firstlastframe_to_video",
        ),
    ],
)
@pytest.mark.parametrize("steps", [0, -1])
async def test_minimax_h3_invalid_default_remains_500(
    api_client, path, files, method, steps
):
    model = DiffusersVideoModel(
        "test",
        "unused",
        SimpleNamespace(
            model_family="MiniMax-H3",
            model_ability=[],
            default_generate_config={"num_inference_steps": steps},
        ),
    )
    model._model = Mock()
    actor_method = "flf_to_video" if method == "firstlastframe_to_video" else method
    getattr(api_client.model, actor_method).side_effect = getattr(model, method)
    async with api_client.client as client:
        body = {"model": "test", "prompt": "hello"}
        if files:
            response = await client.post(path, data=body, files=_uploads(files))
        else:
            response = await client.post(path, json=body)
    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "num_inference_steps must be at least 1"}
    model._model.assert_not_called()
    api_client.api._report_error_event.assert_awaited_once_with(
        "test", "num_inference_steps must be at least 1"
    )


class _ValidatingSpeechActor(xo.StatelessActor):
    async def speech(self, **kwargs):
        AceStepModel._validate_request(
            input=kwargs["input"],
            instruct="piano",
            voice=kwargs["voice"],
            response_format=kwargs["response_format"],
            speed=kwargs["speed"],
            stream=kwargs["stream"],
            seed=-1,
            duration=60,
        )


@pytest.mark.asyncio
async def test_model_validation_message_survives_actor_boundary(api_client):
    pool = await asyncio.wait_for(
        xo.create_actor_pool(f"127.0.0.1:{xo.utils.get_next_port()}", n_process=1),
        timeout=60,
    )
    async with pool:
        actor = await asyncio.wait_for(
            xo.create_actor(
                _ValidatingSpeechActor,
                address=pool.external_address,
                allocate_strategy=xo.allocate_strategy.ProcessIndex(1),
                uid="validating-speech-model",
            ),
            timeout=60,
        )
        api_client.require_model.return_value = actor
        async with api_client.client as client:
            response = await client.post(
                "/v1/audio/speech",
                json={"model": "test", "input": "lyrics", "voice": "bad"},
            )
        assert response.status_code == 400, response.text
        assert response.json() == {
            "detail": (
                "ACE-Step 1.5 only accepts `voice` as null, an empty string, or 'default'."
            )
        }
        api_client.api._report_error_event.assert_not_awaited()
