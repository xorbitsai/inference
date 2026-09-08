"""Real Diffusers smoke tests, enabled with XINFERENCE_SDAPI_TEST_MODEL.

Point the variable to a local StableDiffusionPipeline directory (the CI-friendly
hf-internal-testing/tiny-stable-diffusion-pipe model also works).
"""

import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from ..stable_diffusion.core import DiffusionModel
from ..utils import encode_pil_to_base64


@pytest.fixture
def pipeline_model():
    path = os.environ.get("XINFERENCE_SDAPI_TEST_MODEL")
    if not path:
        pytest.skip("Set XINFERENCE_SDAPI_TEST_MODEL to a local tiny SD pipeline")
    from diffusers import StableDiffusionPipeline

    spec = SimpleNamespace(
        model_ability=["text2image", "image2image", "inpainting"],
        model_base="SD 1.5",
        default_generate_config={},
        default_model_config={},
        model_name="tiny",
        model_revision="tiny",
    )
    model = DiffusionModel(
        "tiny",
        model_spec=spec,
        device=os.environ.get("XINFERENCE_SDAPI_TEST_DEVICE", "cpu"),
    )
    model._model = StableDiffusionPipeline.from_pretrained(path, safety_checker=None)
    model._torch_dtype = torch.float16 if model._device == "cuda" else torch.float32
    model._model.to(device=model._device, dtype=model._torch_dtype)
    model._image_batch_scheduler = None
    return model


@pytest.mark.asyncio
async def test_real_txt2img_reproducible(pipeline_model):
    kwargs = dict(
        prompt="(cat:1.2)",
        width=32,
        height=32,
        steps=2,
        seed=1,
        subseed=2,
        subseed_strength=0.4,
    )
    first = await pipeline_model.txt2img(**kwargs)
    second = await pipeline_model.txt2img(**kwargs)
    assert first["images"] == second["images"]


@pytest.mark.asyncio
async def test_real_img2img_and_mask(pipeline_model):
    image = encode_pil_to_base64(Image.new("RGB", (32, 32)))
    kwargs = dict(
        init_images=[image], prompt="cat", width=32, height=32, steps=2, seed=1
    )
    first = await pipeline_model.img2img(**kwargs)
    second = await pipeline_model.img2img(**kwargs)
    assert first["images"] == second["images"]
    mask = encode_pil_to_base64(Image.new("L", (32, 32), 255))
    result = await pipeline_model.img2img(mask=mask, **kwargs)
    assert len(result["images"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("upscaler", ["None", "Latent"])
async def test_real_hires_batch(pipeline_model, upscaler):
    result = await pipeline_model.txt2img(
        prompt="cat",
        width=32,
        height=32,
        steps=2,
        seed=1,
        batch_size=2,
        enable_hr=True,
        hr_upscaler=upscaler,
        hr_scale=2,
    )
    assert len(result["images"]) == 2


@pytest.mark.asyncio
async def test_real_multiple_controlnets(pipeline_model, monkeypatch, tmp_path):
    import math

    from diffusers import ControlNetModel

    from .. import cache_manager

    stages = int(math.log2(pipeline_model._model.vae_scale_factor)) + 1
    controlnet = ControlNetModel.from_unet(
        pipeline_model._model.unet,
        conditioning_embedding_out_channels=tuple(16 * 2**i for i in range(stages)),
    )
    controlnet.save_pretrained(tmp_path)
    specs = [
        SimpleNamespace(model_name=name, model_revision="tiny")
        for name in ("canny", "depth")
    ]
    pipeline_model._model_spec.controlnet = specs
    monkeypatch.setattr(
        cache_manager,
        "ImageCacheManager",
        lambda spec: SimpleNamespace(cache=lambda: str(tmp_path)),
    )
    units = [
        {
            "model": spec.model_name,
            "module": "none",
            "image": encode_pil_to_base64(Image.new("RGB", (32, 32))),
        }
        for spec in specs
    ]
    result = await pipeline_model.txt2img(
        prompt="cat",
        width=32,
        height=32,
        steps=2,
        seed=1,
        alwayson_scripts={"ControlNet": {"args": units}},
    )
    assert len(result["images"]) == 1


@pytest.mark.asyncio
async def test_real_lora_request_cleanup(pipeline_model, monkeypatch, tmp_path):
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    from .. import lora

    pipe = pipeline_model._model
    pipe.unet.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"]))
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(pipe.unet))
    pipe.save_lora_weights(tmp_path, unet_lora_layers=state)
    pipe.unet.delete_adapters("default")
    spec = SimpleNamespace(
        model_name="style", model_family="lora", model_uri=str(tmp_path), metadata={}
    )
    monkeypatch.setattr(lora, "available_loras", lambda: [spec])
    result = await pipeline_model.txt2img(
        prompt="cat <lora:style:0.7>", width=32, height=32, steps=2, seed=1
    )
    assert len(result["images"]) == 1
    assert all(not adapters for adapters in pipe.get_list_adapters().values())


@pytest.mark.asyncio
async def test_real_sdxl_long_weighted_prompt():
    path = os.environ.get("XINFERENCE_SDAPI_TEST_SDXL_MODEL")
    if not path:
        pytest.skip(
            "Set XINFERENCE_SDAPI_TEST_SDXL_MODEL to a local tiny SDXL pipeline"
        )
    from diffusers import StableDiffusionXLPipeline

    spec = SimpleNamespace(
        model_ability=["text2image", "image2image"],
        model_base="SDXL",
        default_generate_config={},
        default_model_config={},
        model_name="tiny-xl",
    )
    model = DiffusionModel(
        "tiny-xl",
        model_spec=spec,
        device=os.environ.get("XINFERENCE_SDAPI_TEST_DEVICE", "cpu"),
    )
    model._model = StableDiffusionXLPipeline.from_pretrained(path)
    model._torch_dtype = torch.float16 if model._device == "cuda" else torch.float32
    model._model.to(device=model._device, dtype=model._torch_dtype)
    model._image_batch_scheduler = None
    result = await model.txt2img(
        prompt="(cat:1.2) " * 40,
        negative_prompt="blur",
        width=32,
        height=32,
        steps=2,
        seed=1,
    )
    assert len(result["images"]) == 1


@pytest.mark.asyncio
async def test_real_reference_controlnet(pipeline_model):
    original_dtype = pipeline_model._model.unet.dtype
    baseline = await pipeline_model.txt2img(
        prompt="cat", width=32, height=32, steps=2, seed=1
    )
    original_forwards = [
        module.forward for module in pipeline_model._model.unet.modules()
    ]
    unit = {
        "module": "reference_only",
        "image": encode_pil_to_base64(Image.new("RGB", (32, 32))),
    }
    result = await pipeline_model.txt2img(
        prompt="cat",
        width=32,
        height=32,
        steps=2,
        seed=1,
        alwayson_scripts={"ControlNet": {"args": [unit]}},
    )
    assert len(result["images"]) == 1
    assert [
        module.forward for module in pipeline_model._model.unet.modules()
    ] == original_forwards

    assert pipeline_model._model.unet.dtype == original_dtype
    after = await pipeline_model.txt2img(
        prompt="cat", width=32, height=32, steps=2, seed=1
    )
    assert after["images"] == baseline["images"]
