# CI: linux

import asyncio
import contextlib
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from ....core.exceptions import InvalidModelRequestError
from ....types import LoRA
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
def qwen_lightning(monkeypatch, request):
    spec = match_diffusion("Qwen-Image")
    variant = getattr(request, "param", None)
    fused_style = variant == "fused"
    model = DiffusionModel(
        "test",
        model_spec=spec,
        lightning_model_path="/tmp/local.safetensors",
        lightning_version=None if variant == "unknown" else "4steps-V1.0-bf16",
        lora_model=(
            [LoRA("style", "/tmp/style.safetensors", 1.0)] if fused_style else None
        ),
    )
    name = model._LIGHTNING_ADAPTER_NAME

    class Layer:
        active_adapters = ["style"]
        lora_A = {"style": object()}
        merged_adapters = ["style"] if fused_style else []

        def set_adapter(self, names):
            self.active_adapters = names

    layer = Layer()
    base_scheduler, lightning_scheduler = object(), object()
    observations = []
    before_generate = Mock()

    class Pipeline:
        transformer = SimpleNamespace(modules=lambda: [layer])
        scheduler = base_scheduler

        def unload_lora_weights(self):
            assert layer.merged_adapters == ["style"]
            layer.lora_A.clear()
            layer.merged_adapters.clear()
            layer.set_adapter([])

        def load_lora_weights(self, path, adapter_name):
            assert adapter_name == name
            layer.lora_A[adapter_name] = object()
            layer.set_adapter([*layer.active_adapters, adapter_name])

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
async def test_qwen_rejects_lightning_without_a_loaded_adapter():
    model = DiffusionModel("base", model_spec=match_diffusion("Qwen-Image"))
    model._model = Mock()
    with pytest.raises(InvalidModelRequestError, match="Launch the model"):
        await model.text_to_image("test", use_lightning=True)
    model._model.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("qwen_lightning", ["unknown"], indirect=True)
async def test_qwen_requires_steps_for_unrecognized_lightning_checkpoint(
    qwen_lightning,
):
    model, observed, _, base_scheduler, lightning_scheduler = qwen_lightning
    assert model.model_family.to_description()["lightning_inference_steps"] is None
    with pytest.raises(InvalidModelRequestError, match="Specify num_inference_steps"):
        await model.text_to_image("test")
    await model.text_to_image("test", use_lightning=False, _return_images=True)
    await model.text_to_image("test", num_inference_steps=8, _return_images=True)
    assert [(scheduler, steps) for _, scheduler, steps in observed] == [
        (base_scheduler, 50),
        (lightning_scheduler, 8),
    ]


@pytest.mark.parametrize("qwen_lightning", ["fused"], indirect=True)
def test_qwen_detaches_fused_launch_lora_before_loading_lightning(qwen_lightning):
    model, _, _, _, _ = qwen_lightning
    (layer,) = model._model.transformer.modules()
    assert list(layer.lora_A) == [model._LIGHTNING_ADAPTER_NAME]
    assert layer.merged_adapters == []


def test_qwen_launch_default_keeps_lightning_with_request_loras(qwen_lightning):
    model, _, _, _, _ = qwen_lightning
    loras = [LoRA("style", "/tmp/style.safetensors", 0.5)]
    model._request_loras = Mock(return_value=contextlib.nullcontext())
    model._call_model("url", prompt="test", loras=loras, _return_images=True)
    model._request_loras.assert_called_once_with(
        model._model, loras, model._LIGHTNING_ADAPTER_NAME
    )


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
    entered, finish, blocked = threading.Event(), threading.Event(), threading.Event()
    lock = threading.RLock()

    class TrackingLock:
        def __enter__(self):
            if not lock.acquire(blocking=False):
                blocked.set()
                lock.acquire()

        def __exit__(self, *args):
            lock.release()

    model._lightning_lock = TrackingLock()

    def wait_for_cancel(prompt):
        if prompt == "base":
            entered.set()
            assert finish.wait(10)

    before_generate.side_effect = wait_for_cancel
    base = asyncio.create_task(
        model.text_to_image("base", use_lightning=False, _return_images=True)
    )
    lightning = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        base.cancel()
        with pytest.raises(asyncio.CancelledError):
            await base
        lightning = asyncio.create_task(
            model.text_to_image("next", use_lightning=True, _return_images=True)
        )
        assert await asyncio.to_thread(blocked.wait, 5)
        before_generate.assert_called_once_with("base")
    finally:
        finish.set()
        if lightning is not None:
            await asyncio.wait_for(lightning, 5)
    assert observed == [
        (("style",), base_scheduler, 50),
        (("style", model._LIGHTNING_ADAPTER_NAME), lightning_scheduler, 4),
    ]


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
