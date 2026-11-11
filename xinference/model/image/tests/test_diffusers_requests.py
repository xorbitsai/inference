# CI: linux

import contextlib
from types import SimpleNamespace
from unittest.mock import Mock

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
