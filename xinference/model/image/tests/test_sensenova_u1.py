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
from importlib import import_module
from pathlib import Path
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
from ..cache_manager import ImageCacheManager
from ..core import (
    IMAGE_MODEL_DESCRIPTIONS,
    create_image_model_instance,
    match_diffusion,
)
from ..engine import TransformersSenseNovaU1ImageModel


@pytest.mark.parametrize(
    ("hub", "namespace", "revision"),
    [("huggingface", "sensenova", "main"), ("modelscope", "SenseNova", "master")],
)
def test_u16_uses_shared_transformers_image_engine(tmp_path, hub, namespace, revision):
    model_name = "SenseNova-U1.6-8B-MoT"

    model = create_image_model_instance(
        model_uid="sensenova-u16",
        model_name=model_name,
        download_hub=hub,
        model_path=str(tmp_path),
    )

    assert isinstance(model, TransformersSenseNovaU1ImageModel)
    assert model.model_spec.model_engine == "transformers"
    assert model.model_spec.model_hub == hub
    assert model.model_spec.model_id == f"{namespace}/{model_name}"
    assert model.model_spec.model_revision == revision
    assert set(model.model_ability) == {"text2image", "image2image"}


@pytest.mark.parametrize(
    ("suffix", "quantization", "filename"),
    [
        ("Native", None, "SenseNova-U1.6-BF16.safetensors"),
        ("Native", "FP8-E4M3FN-W8A8", "SenseNova-U1.6-FP8-E4M3FN-W8A8.safetensors"),
        ("Native", "INT8-ConvRot-W8A8", "SenseNova-U1.6-INT8-ConvRot-W8A8.safetensors"),
        ("GGUF", None, "SenseNova-U1.6-8B-MoT-Q4_0.gguf"),
        ("GGUF", "Q6_K", "SenseNova-U1.6-8B-MoT-Q6_K.gguf"),
        ("GGUF", "Q8_0", "SenseNova-U1.6-8B-MoT-Q8_0.gguf"),
    ],
)
@pytest.mark.parametrize(
    ("hub", "namespace", "revision", "pattern_key", "download_target"),
    [
        ("huggingface", "sensenova", "main", "allow_patterns", "huggingface_hub"),
        (
            "modelscope",
            "SenseNova",
            "master",
            "allow_file_pattern",
            "modelscope.hub.snapshot_download",
        ),
    ],
)
def test_u16_variant_downloads_one_checkpoint(
    monkeypatch,
    tmp_path,
    suffix,
    quantization,
    filename,
    hub,
    namespace,
    revision,
    pattern_key,
    download_target,
):
    monkeypatch.setattr(
        "xinference.constants.XINFERENCE_CACHE_DIR", str(tmp_path / "cache")
    )
    monkeypatch.setattr(
        "xinference.constants.XINFERENCE_MODEL_DIR", str(tmp_path / "models")
    )
    model_name = f"SenseNova-U1.6-8B-MoT-{suffix}"
    expected = {
        filename,
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
    }
    if suffix == "Native" and quantization:
        expected.add(
            filename.removesuffix(".safetensors") + ".quantization_config.json"
        )

    def download(model_id, **kwargs):
        assert model_id == f"{namespace}/{model_name}"
        assert kwargs["revision"] == revision
        assert set(kwargs[pattern_key]) == expected
        snapshot = Path(kwargs.get("local_dir") or tmp_path / "snapshot")
        snapshot.mkdir(parents=True, exist_ok=True)
        for file in expected:
            (snapshot / file).touch()
        return str(snapshot)

    mocked_download = Mock(side_effect=download)
    monkeypatch.setattr(
        import_module(download_target), "snapshot_download", mocked_download
    )
    spec = match_diffusion(model_name, hub, quantization=quantization)
    assert not ImageCacheManager(spec).get_cache_status()

    model = create_image_model_instance(
        model_uid="sensenova-variant",
        model_name=model_name,
        download_hub=hub,
        quantization=quantization,
    )

    assert isinstance(model, TransformersSenseNovaU1ImageModel)
    assert model.model_ability == ["text2image"]
    assert model.model_spec.quantization == (
        quantization or ("BF16" if suffix == "Native" else "Q4_0")
    )
    assert set(p.name for p in Path(model._model_path).iterdir()) == expected
    assert ImageCacheManager(spec).get_cache_status()
    version = model.model_spec.to_version_info()[0]
    assert version["model_version"] == f"{model_name}--{model.model_spec.quantization}"
    assert version["cache_status"]
    assert len({v["model_version"] for v in IMAGE_MODEL_DESCRIPTIONS[model_name]}) == 3
    mocked_download.assert_called_once()


def test_u16_variant_rejects_unknown_quantization():
    with pytest.raises(ValueError, match="does not support quantization"):
        create_image_model_instance(
            model_uid="sensenova-variant",
            model_name="SenseNova-U1.6-8B-MoT-GGUF",
            quantization="Q5_K_M",
            enable_virtual_env=True,
        )


@pytest.mark.parametrize(
    ("device", "options", "requires_offload"),
    [
        (None, {}, True),
        ("cpu", {"vram_mode": "fast"}, True),
        ("mps", {"vram_mode": "low"}, True),
        ("cpu", {"vram_mode": "full"}, False),
        ("mps", {"vram_mode": "full"}, False),
        ("cuda:0", {}, True),
        ("xpu:0", {}, True),
    ],
)
def test_u16_gguf_validates_offload_device_before_loading(
    monkeypatch, tmp_path, device, options, requires_offload
):
    monkeypatch.syspath_prepend(
        str(Path(sensenova_u1.__file__).resolve().parents[2] / "thirdparty")
    )
    runtime_utils = import_module("sensenova_u1.utils")
    monkeypatch.setattr(
        runtime_utils, "best_available_device", lambda: torch.device("cpu")
    )
    loader = Mock(return_value=(object(), object()))
    monkeypatch.setattr(runtime_utils, "load_model_and_tokenizer", loader)
    model = create_image_model_instance(
        model_uid="sensenova-gguf",
        model_name="SenseNova-U1.6-8B-MoT-GGUF",
        model_path=str(tmp_path),
        device=device,
        **options,
    )

    resolved_device = device or "cpu"
    mode = options.get("vram_mode", "balanced")
    if requires_offload and torch.device(resolved_device).type in ("cpu", "mps"):
        with pytest.raises(
            ValueError,
            match=f"vram_mode='{mode}'.*CUDA or XPU.*device '{resolved_device}'",
        ):
            model.load()
        loader.assert_not_called()
    else:
        model.load()
        loader.assert_called_once()
        assert loader.call_args.kwargs["device"] == resolved_device
        assert loader.call_args.kwargs["for_offload"] is requires_offload


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
