# CI: linux

import asyncio
import contextlib
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from ....core.exceptions import InvalidModelRequestError
from ..core import match_diffusion
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


@pytest.mark.parametrize(
    "name",
    [
        "Qwen-Image",
        "Qwen-Image-2512",
        "Qwen-Image-Edit",
        "Qwen-Image-Edit-2509",
        "Qwen-Image-Edit-2511",
    ],
)
def test_qwen_image_versions_report_lightning_steps(name):
    spec = match_diffusion(name)
    base = DiffusionModel("base", model_spec=spec)
    assert "lightning_inference_steps" not in base.model_family.to_description()
    assert spec.lightning_versions
    for version in spec.lightning_versions:
        model = DiffusionModel(
            "lightning",
            model_spec=spec,
            lightning_model_path="/tmp/local.safetensors",
            lightning_version=version,
        )
        assert model.model_family.to_description()["lightning_inference_steps"] == int(
            version[0]
        )
    assert "lightning_inference_steps" not in spec.to_description()


@pytest.fixture
def qwen_lightning(monkeypatch):
    spec = match_diffusion("Qwen-Image")
    model = DiffusionModel(
        "test",
        model_spec=spec,
        lightning_model_path="/tmp/local.safetensors",
        lightning_version="4steps-V1.0-bf16",
    )
    name = model._LIGHTNING_ADAPTER_NAME

    class Layer:
        active_adapters = ["style"]
        lora_A = {}

        def set_adapter(self, names):
            self.active_adapters = names

    layer = Layer()
    base_scheduler, lightning_scheduler = object(), object()
    observations = []
    before_generate = Mock()

    class Pipeline:
        transformer = SimpleNamespace(modules=lambda: [layer])
        scheduler = base_scheduler

        def load_lora_weights(self, path, adapter_name):
            assert adapter_name == name
            layer.lora_A[adapter_name] = object()
            layer.active_adapters = ["style", adapter_name]

        def __call__(self, prompt, image=None, mask_image=None, num_inference_steps=50):
            observations.append(
                (tuple(layer.active_adapters), self.scheduler, num_inference_steps)
            )
            before_generate(prompt)
            return SimpleNamespace(images=[Image.new("RGB", (8, 8))])

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            FlowMatchEulerDiscreteScheduler=SimpleNamespace(
                from_config=lambda config: lightning_scheduler
            )
        ),
    )
    model._model = Pipeline()
    model._load_lightning()
    monkeypatch.setattr(model, "_get_model", lambda *args, **kwargs: model._model)
    monkeypatch.setattr(model, "_process_progressor", Mock())
    return model, observations, before_generate, base_scheduler, lightning_scheduler


@pytest.mark.asyncio
@pytest.mark.parametrize("ability", ["text_to_image", "image_to_image", "inpainting"])
async def test_qwen_lightning_restores_scheduler_and_only_disables_its_adapter(
    qwen_lightning, ability
):
    model, observed, before_generate, base_scheduler, lightning_scheduler = (
        qwen_lightning
    )
    assert model.model_family.to_description()["lightning_inference_steps"] == 4
    image = Image.new("RGB", (8, 8))
    options = {"_return_images": True, "size": "8*8"}
    if ability != "text_to_image":
        options["image"] = image
    if ability == "inpainting":
        options["mask_image"] = image.convert("L")
    generate = getattr(model, ability)

    async def request(**kwargs):
        if ability == "text_to_image":
            return await generate(**options, **kwargs)
        return await asyncio.to_thread(generate, **options, **kwargs)

    await request(prompt="default")
    before_generate.side_effect = RuntimeError("generation failed")
    with pytest.raises(RuntimeError, match="generation failed"):
        await request(prompt="base", use_lightning=False, num_inference_steps=32)
    before_generate.side_effect = None
    await request(prompt="on again", use_lightning=True)
    with pytest.raises(InvalidModelRequestError, match="must be a boolean"):
        await request(prompt="invalid", use_lightning="false")
    enabled = ("style", model._LIGHTNING_ADAPTER_NAME)
    assert observed == [
        (enabled, lightning_scheduler, 4),
        (("style",), base_scheduler, 32),
        (enabled, lightning_scheduler, 4),
    ]
    assert model._model.scheduler is base_scheduler


@pytest.mark.asyncio
async def test_cancelled_qwen_request_cannot_leak_adapter_state(qwen_lightning):
    model, observed, before_generate, base_scheduler, lightning_scheduler = (
        qwen_lightning
    )
    entered, finish = threading.Event(), threading.Event()

    def wait_for_cancel(prompt):
        if prompt == "base":
            entered.set()
            assert finish.wait(5)

    before_generate.side_effect = wait_for_cancel
    base = asyncio.create_task(
        model.text_to_image("base", use_lightning=False, _return_images=True)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        base.cancel()
        with pytest.raises(asyncio.CancelledError):
            await base
        lightning = asyncio.create_task(
            model.text_to_image("next", use_lightning=True, _return_images=True)
        )
        await asyncio.sleep(0)
    finally:
        finish.set()
    await asyncio.wait_for(lightning, 5)
    assert observed == [
        (("style",), base_scheduler, 50),
        (("style", model._LIGHTNING_ADAPTER_NAME), lightning_scheduler, 4),
    ]
