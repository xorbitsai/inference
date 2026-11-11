# CI: linux

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ..stable_diffusion.core import DiffusionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [1, 2])
async def test_text_to_image_uses_pipeline(n):
    spec = SimpleNamespace(
        model_ability=["text2image"],
        default_generate_config={"num_inference_steps": 12},
    )
    model = DiffusionModel("test", model_spec=spec)
    model._get_latents = Mock(return_value=None)
    model._call_model = Mock(return_value={"data": ["image"] * n})

    result = await model.text_to_image(
        "a cat", n=n, size="512*768", response_format="b64_json", seed=42
    )

    assert result == {"data": ["image"] * n}
    model._call_model.assert_called_once_with(
        prompt="a cat",
        num_images_per_prompt=n,
        response_format="b64_json",
        num_inference_steps=12,
        width=512,
        height=768,
        seed=42,
        loras=None,
    )


def test_inpainting_applies_default_generate_config():
    spec = SimpleNamespace(
        model_ability=["inpainting"],
        default_generate_config={"num_inference_steps": 4, "guidance_scale": 1.0},
    )
    model = DiffusionModel("test", model_spec=spec)
    model._model = Mock()
    model._patch_prepare_latents = Mock(return_value=contextlib.nullcontext())
    model._call_model = Mock(return_value={"data": ["image"]})
    image, mask = Mock(), Mock()

    model.inpainting(
        image,
        mask,
        prompt="a cat",
        size="512*768",
        response_format="b64_json",
        guidance_scale=3.5,
        strength=None,
    )

    model._call_model.assert_called_once_with(
        image=image,
        mask_image=mask,
        prompt="a cat",
        num_images_per_prompt=1,
        response_format="b64_json",
        model=model._model,
        num_inference_steps=4,
        guidance_scale=3.5,
        width=512,
        height=768,
    )


def test_image_edits_with_mask_keeps_the_input_size(monkeypatch):
    # An OpenAI-style /v1/images/edits request with a mask and no size.
    import io
    import json

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from PIL import Image

    from ....api.restful_api import RESTfulAPI

    spec = SimpleNamespace(model_ability=["inpainting"], default_generate_config={})
    model = DiffusionModel("test", model_spec=spec)
    model._model = Mock()
    model._patch_prepare_latents = Mock(return_value=contextlib.nullcontext())
    model._call_model = Mock(return_value={"created": 0, "data": []})

    async def inpainting(
        image, mask_image, prompt, negative_prompt, n, size, response_format, **kwargs
    ):
        # Same call as ModelActor.inpainting makes on the model.
        kwargs["negative_prompt"] = negative_prompt
        return json.dumps(
            model.inpainting(
                image, mask_image, prompt, n, size, response_format, **kwargs
            )
        )

    model_ref = SimpleNamespace(uid=b"sd-inpainting", inpainting=inpainting)
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_name": "sd-inpainting"})
    )
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    api._report_error_event = AsyncMock()
    api._get_model_last_error = AsyncMock(side_effect=lambda uid, error: error)
    for name in (
        "_set_trace_model",
        "_set_trace_model_type",
        "_check_model_access",
        "_add_running_task",
    ):
        monkeypatch.setattr(api, name, Mock())
    monkeypatch.setattr(
        "xinference.api.restful_api.require_model", AsyncMock(return_value=model_ref)
    )
    app = FastAPI()
    app.post("/v1/images/edits")(api.create_image_edits)

    def png(mode, color):
        buffer = io.BytesIO()
        Image.new(mode, (64, 48), color).save(buffer, format="PNG")
        return buffer.getvalue()

    with TestClient(app) as client:
        response = client.post(
            "/v1/images/edits",
            data={
                "model": "sd-inpainting",
                "prompt": "a cat on the bench",
                "response_format": "b64_json",
            },
            files=[
                ("image", ("image.png", png("RGB", (0, 0, 0)), "image/png")),
                ("mask", ("mask.png", png("L", 255), "image/png")),
            ],
        )

    assert response.status_code == 200, response.text
    kwargs = model._call_model.call_args.kwargs
    assert (kwargs["width"], kwargs["height"]) == (64, 48)
    assert kwargs["prompt"] == "a cat on the bench"
    assert kwargs["mask_image"].size == (64, 48)
