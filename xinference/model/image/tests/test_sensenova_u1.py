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

import contextlib
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import torch
from fastapi import FastAPI
from PIL import Image

from ....api.restful_api import RESTfulAPI
from ....core.model import ModelActor
from .. import sensenova_u1


def _make_model():
    model_spec = SimpleNamespace(
        model_ability=["text2image", "image2image"],
        default_generate_config={},
        to_description=lambda: {},
    )
    return sensenova_u1.SenseNovaU1Model("test", "/path", model_spec)


@pytest.mark.parametrize("kwargs", [{}, {"seed": None}, {"seed": -1}])
def test_generate_config_randomizes_default_seed(monkeypatch, kwargs):
    monkeypatch.setattr(sensenova_u1.random, "randint", lambda lower, upper: 1234)

    config = _make_model()._get_generate_config(kwargs)

    assert config["seed"] == 1234


@pytest.mark.parametrize("seed", [0, 1, 1234])
def test_generate_config_preserves_non_negative_seed(seed):
    config = _make_model()._get_generate_config({"seed": seed})

    assert config["seed"] == seed


class _Pipeline:
    """Stands in for the loaded network and records the prompts it is given."""

    def __init__(self):
        self.prompts = []

    def it2i_generate(self, tokenizer, prompt, images, *, image_size, batch_size, **_):
        self.prompts.append(prompt)
        width, height = image_size
        return torch.zeros(batch_size, 3, height, width)


def _loaded_model():
    model = _make_model()
    model._tokenizer = object()
    model._model = _Pipeline()
    return model


def _edit(model, prompt):
    return model.image_to_image(
        Image.new("RGB", (64, 64)),
        prompt,
        size="64*64",
        response_format="b64_json",
        do_resize=False,
    )


@pytest.mark.parametrize("prompt", ["an apple", ["an apple"]])
def test_image_to_image_accepts_one_prompt(prompt):
    model = _loaded_model()

    result = _edit(model, prompt)

    assert model._model.prompts == ["an apple"]
    assert len(result["data"]) == 1


@pytest.mark.parametrize("prompt", [[], ["an apple", "a pear"]])
def test_image_to_image_refuses_other_lists(prompt):
    model = _loaded_model()

    with pytest.raises(ValueError, match="requires a string prompt"):
        _edit(model, prompt)

    assert not model._model.prompts


def test_image_to_image_requires_a_prompt():
    model = _loaded_model()

    with pytest.raises(ValueError, match="requires a prompt"):
        _edit(model, None)


@pytest.fixture
def variations_client(monkeypatch):
    """/v1/images/variations through the real handler and ModelActor."""
    model = _loaded_model()
    actor = ModelActor("test:1", "test:2", model, "sensenova-0")  # type: ignore
    actor._model_state = "ready"
    actor.record_metrics = AsyncMock()
    actor._get_progressor = AsyncMock(return_value=contextlib.nullcontext())
    model_ref = SimpleNamespace(uid=b"sensenova-0", image_to_image=actor.image_to_image)
    monkeypatch.setattr(
        "xinference.api.restful_api.require_model", AsyncMock(return_value=model_ref)
    )
    api = object.__new__(RESTfulAPI)
    api._set_trace_model = Mock()
    api._set_trace_model_type = Mock()
    api._check_model_access = Mock()
    api._get_supervisor_ref = AsyncMock()
    api._report_error_event = AsyncMock()
    api._add_running_task = Mock()
    app = FastAPI()
    app.add_api_route("/v1/images/variations", api.create_variations, methods=["POST"])
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )
    return client, model._model


async def _post_variation(client, prompt):
    image = io.BytesIO()
    Image.new("RGB", (64, 64)).save(image, format="PNG")
    async with client:
        return await client.post(
            "/v1/images/variations",
            data={
                "model": "sensenova",
                "prompt": prompt,
                "size": "64*64",
                "response_format": "b64_json",
                "kwargs": json.dumps({"do_resize": False}),
            },
            files={"image": ("apple.png", image.getvalue(), "image/png")},
        )


async def test_variations_form_prompt_reaches_the_pipeline(variations_client):
    client, pipeline = variations_client

    response = await _post_variation(client, "an apple")

    assert response.status_code == 200, response.text
    assert pipeline.prompts == ["an apple"]
    assert len(response.json()["data"]) == 1


async def test_variations_repeated_prompt_fields_are_still_refused(variations_client):
    client, pipeline = variations_client

    response = await _post_variation(client, ["an apple", "a pear"])

    assert response.status_code >= 400
    assert "requires a string prompt" in response.text
    assert not pipeline.prompts
