"""CPU-only regression coverage for SD WebUI orchestration and helpers."""

import asyncio
import io
import socket
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from .. import rng, utils
from ..lora import process_loras, process_prompt
from ..prompt_converter import parse_prompt_attention
from ..sdapi import SDAPIDiffusionModelMixin


def encoded(size=(16, 8), mode="RGB"):
    return utils.encode_pil_to_base64(Image.new(mode, size))


class FakeModel(SDAPIDiffusionModelMixin):
    _model_spec = SimpleNamespace(model_base="SD 1.5", default_generate_config={})

    def __init__(self):
        self.calls = []

    def generate(self, kind, **kwargs):
        self.calls.append((kind, kwargs))
        width, height = map(int, kwargs["size"].split("*"))
        return [Image.new("RGB", (width, height)) for _ in range(kwargs["n"])]

    async def text_to_image(self, **kwargs):
        return self.generate("txt2img", **kwargs)

    def image_to_image(self, **kwargs):
        return self.generate("img2img", **kwargs)

    def inpainting(self, **kwargs):
        return self.generate("inpaint", **kwargs)


@pytest.fixture(autouse=True)
def cpu_rng(monkeypatch):
    monkeypatch.setattr(rng, "get_available_device", lambda: "cpu")


@pytest.mark.asyncio
async def test_txt2img_defaults_and_batch_seeds():
    model = FakeModel()
    result = await model.txt2img(seed=10, subseed=30, batch_size=2, n_iter=2)
    assert len(result["images"]) == 4
    assert result["info"]["all_seeds"] == [10, 11, 12, 13]
    assert [call[1]["seed"] for call in model.calls] == [[10, 11], [12, 13]]
    assert model.calls[0][1]["num_inference_steps"] == 20
    assert model.calls[0][1]["loras"] == []


@pytest.mark.asyncio
async def test_subseed_preserves_primary_seed():
    result = await FakeModel().txt2img(
        seed=3, subseed=7, subseed_strength=0.5, batch_size=2
    )
    assert result["info"]["all_seeds"] == [3, 3]
    assert result["info"]["all_subseeds"] == [7, 8]


@pytest.mark.asyncio
@pytest.mark.parametrize("resize_mode", [0, 1, 2])
async def test_img2img_resize_mask_and_batch(resize_mode):
    model = FakeModel()
    result = await model.img2img(
        init_images=[encoded()],
        mask=encoded(mode="L"),
        width=32,
        height=24,
        resize_mode=resize_mode,
        inpainting_mask_invert=1,
        batch_size=2,
        n_iter=2,
        seed=7,
        denoising_strength=0.5,
    )
    assert len(result["images"]) == 4
    assert all(kind == "inpaint" for kind, _ in model.calls)
    kwargs = model.calls[0][1]
    assert kwargs["image"].size == (32, 24)
    assert kwargs["mask_image"].getpixel((0, 0)) == 255
    assert kwargs["num_inference_steps"] == 40
    assert model.calls[1][1]["seed"] == [9, 10]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_base", ["SD 1.5", "other"])
@pytest.mark.parametrize("image_count", [1, 2])
@pytest.mark.parametrize("crop", [None, False, True])
@pytest.mark.parametrize("padding", [0, 8])
async def test_schema_inpainting_crop(model_base, image_count, crop, padding):
    from ....api.schemas.requests import SDAPIImg2imgRequst

    model = FakeModel()
    model._model_spec = SimpleNamespace(
        model_base=model_base, default_generate_config={}
    )
    if model_base == "other":

        def inpainting(**kwargs):
            model.calls.append(("inpaint", kwargs))
            return {"created": 0, "data": [{"b64_json": encoded()}]}

        model.inpainting = inpainting
    params = dict(
        init_images=[encoded()] * image_count,
        mask=encoded(mode="L"),
        batch_size=image_count,
        width=16,
        height=8,
        inpaint_full_res_padding=padding,
    )
    if crop is not None:
        params["inpaint_full_res"] = crop
    request = SDAPIImg2imgRequst(**params)
    await model.img2img(**request.dict())
    kind, kwargs = model.calls[0]
    assert kind == "inpaint"
    assert isinstance(kwargs["mask_image"], Image.Image)
    if image_count == 2:
        assert len(kwargs["image"]) == 2
        assert all(isinstance(image, Image.Image) for image in kwargs["image"])
    else:
        assert isinstance(kwargs["image"], Image.Image)
    if crop:
        assert kwargs["padding_mask_crop"] == padding
    else:
        assert "padding_mask_crop" not in kwargs


@pytest.mark.asyncio
async def test_hires_fix_two_stages():
    model = FakeModel()
    result = await model.txt2img(
        width=16,
        height=16,
        enable_hr=True,
        hr_upscaler="None",
        hr_scale=2,
        hr_second_pass_steps=4,
        denoising_strength=0.5,
    )
    assert [kind for kind, _ in model.calls] == ["txt2img", "img2img"]
    assert model.calls[1][1]["size"] == "32*32"
    assert model.calls[1][1]["num_inference_steps"] == 8
    assert utils.decode_base64_to_image(result["images"][0]).size == (32, 32)


@pytest.mark.asyncio
async def test_adetailer_stage_for_img2img(monkeypatch):
    from .. import adetailer

    seen = []

    def detail(model, sd_type, image, kwargs, alwayson_scripts, **extra):
        seen.append((sd_type, kwargs["seed"]))
        return image

    monkeypatch.setattr(adetailer, "process_adetailer", detail)
    await FakeModel().img2img(
        init_images=[encoded()],
        batch_size=2,
        seed=8,
        alwayson_scripts={"ADetailer": {"args": [True, {}]}},
    )
    assert seen == [("img2img", [8]), ("img2img", [9])]
    assert adetailer.extract_adetailer_params({}) == []


@pytest.mark.asyncio
async def test_cancel_waits_for_pipeline_thread():
    import threading

    entered, stopped = threading.Event(), threading.Event()

    def pipeline(_cancel_event):
        entered.set()
        assert _cancel_event.wait(2)
        stopped.set()
        raise RuntimeError("cancelled")

    task = asyncio.create_task(FakeModel._sdapi_call(pipeline))
    await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


@pytest.mark.parametrize("value", [encoded(), "data:image/png;base64," + encoded()])
def test_image_decoding(value):
    assert utils.decode_base64_to_image(value).size == (16, 8)


def test_metadata_roundtrip():
    image = Image.new("RGB", (8, 8))
    image.info["parameters"] = "weighted prompt"
    assert (
        utils.decode_base64_to_image(utils.encode_pil_to_base64(image)).info[
            "parameters"
        ]
        == "weighted prompt"
    )


def test_jpeg_metadata_roundtrip():
    pytest.importorskip("piexif")
    image = Image.new("RGB", (8, 8))
    assert utils.decode_base64_to_image(
        utils.encode_pil_to_base64(image, "jpeg")
    ).size == (8, 8)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/a",
        "http://[::1]/a",
        "http://user:pass@example.com/a",
    ],
)
def test_reject_private_urls(url):
    assert not utils.verify_url(url)
    with pytest.raises(ValueError):
        with utils._open_public_url(url):
            pass


def test_mixed_dns_is_rejected(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", (ip, 80)) for ip in ("8.8.8.8", "10.0.0.1")],
    )
    assert not utils.verify_url("http://example.com/")


def test_redirect_to_private_host_is_rejected(monkeypatch):
    import urllib3

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["host"] == "8.8.8.8"

        def urlopen(self, *args, **kwargs):
            return SimpleNamespace(
                status=302,
                headers={"Location": "http://127.0.0.1/secret"},
                close=lambda: None,
            )

        def close(self):
            pass

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **k: [
            (2, 1, 6, "", ("127.0.0.1" if host == "127.0.0.1" else "8.8.8.8", 80))
        ],
    )
    monkeypatch.setattr(urllib3, "HTTPConnectionPool", Pool)
    with pytest.raises(ValueError, match="public"):
        with utils._open_public_url("http://example.com/"):
            pass


def test_civitai_sanitizes_filename_and_cleans_partial(monkeypatch, tmp_path):
    @contextmanager
    def opened(*args):
        stream = io.BytesIO(b"weights")
        stream.headers = {
            "Content-Disposition": 'attachment; filename="../../lora.safetensors"'
        }
        yield stream, "https://cdn.example.com/file"

    monkeypatch.setattr(utils, "_open_public_url", opened)
    path = utils.download_civitai_model(
        "https://civitai.com/api/download/models/1", str(tmp_path), "token"
    )
    assert path == str(tmp_path / "lora.safetensors")
    assert (tmp_path / "lora.safetensors").read_bytes() == b"weights"


def test_lora_negative_weights_and_strict(monkeypatch):
    from .. import lora

    assert process_prompt("a <lora:style:-0.5> face")[0] == [("style", -0.5)]
    monkeypatch.setattr(lora, "available_loras", lambda: [])
    with pytest.raises(ValueError, match="LoRA not found"):
        process_loras({"prompt": "<lora:missing:1>"}, strict=True)


def test_rng_reproducibility_without_global_side_effects():
    before = torch.random.get_rng_state()
    first = rng.ImageRNG((4, 4, 4), [5, 6], [9, 10], 0.4).next()
    second = rng.ImageRNG((4, 4, 4), [5, 6], [9, 10], 0.4).next()
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])
    assert torch.equal(before, torch.random.get_rng_state())


def test_slerp_endpoints_and_prompt_weights():
    low, high = torch.ones(2, 3), torch.ones(2, 3) * 2
    assert torch.equal(rng.slerp(0, low, high), low)
    assert torch.equal(rng.slerp(1, low, high), high)
    parsed = parse_prompt_attention("(cat:1.5) [dog]")
    assert parsed[0] == ["cat", 1.5]
    assert parsed[-1][1] == pytest.approx(1 / 1.1)


@pytest.fixture
def controlnet_dependencies():
    for package in ("cv2", "skimage", "einops", "torchvision", "transformers"):
        pytest.importorskip(package)


def test_controlnet_detect_canny_and_registry(controlnet_dependencies):
    from ..controlnet import control_types, detect, list_modules

    modules = list_modules()["module_list"]
    assert "canny" in modules and "none" in modules
    result = detect("canny", [encoded((64, 64))], 64)
    assert len(result["images"]) == 1
    assert "All" in control_types(SimpleNamespace(controlnet=[]))["control_types"]


def test_lora_scale_roundtrip():
    from ....types import LoRA

    value = LoRA("style", "/tmp/style.safetensors", lora_scale=-0.5)
    assert LoRA.from_dict(value.to_dict()).lora_scale == -0.5
    assert "lora_scale" not in LoRA("default", "/tmp/default").to_dict()


def test_request_loras_cleaned_after_failure():
    from ....types import LoRA
    from ..stable_diffusion.core import DiffusionModel

    calls = []
    model = SimpleNamespace(
        get_active_adapters=lambda: ["original"],
        load_lora_weights=lambda *a, **k: calls.append(("load", k)),
        enable_lora=lambda: None,
        set_adapters=lambda *a: calls.append(("set", a)),
        delete_adapters=lambda names: calls.append(("delete", names)),
    )
    with pytest.raises(RuntimeError):
        with DiffusionModel._request_loras(model, [LoRA("style", "path", 0.5)]):
            raise RuntimeError("inference failed")
    assert calls[-2] == ("delete", ["xinference_sdapi_0"])
    assert calls[-1] == ("set", (["original"],))


@pytest.mark.asyncio
async def test_basic_non_sd_engine_is_preserved():
    model = FakeModel()
    model._model_spec = SimpleNamespace(model_base="FLUX.1")

    async def basic(**kwargs):
        return {"data": [{"b64_json": "image"}], "created": 42}

    model.text_to_image = basic
    result = await model.txt2img(prompt="cat")
    assert result["images"] == ["image"]
    with pytest.raises(ValueError, match="Stable Diffusion"):
        await model.txt2img(enable_hr=True)


def test_adetailer_installed_api_contract(monkeypatch):
    package = pytest.importorskip("adetailer")
    from .. import adetailer
    from ..sdapi import _NoProgress

    image = Image.new("RGB", (32, 32))
    mask = Image.new("L", (32, 32), 255)
    monkeypatch.setattr(adetailer, "_get_model", lambda _: "fake.pt")
    monkeypatch.setattr(
        package,
        "ultralytics_predict",
        lambda *a, **k: package.PredictOutput(
            bboxes=[[0, 0, 32, 32]], masks=[mask], confidences=[0.9], preview=image
        ),
    )
    calls = []

    def inpaint(**kwargs):
        calls.append(kwargs)
        return [image]

    result = adetailer.process_adetailer(
        SimpleNamespace(inpainting=inpaint),
        "txt2img",
        image,
        {
            "prompt": "cat",
            "negative_prompt": "",
            "size": "32*32",
            "progressor": _NoProgress(),
        },
        {
            "ADetailer": {
                "args": [
                    True,
                    False,
                    {"ad_model": "face_yolov8n.pt", "ad_prompt": "[PROMPT] face"},
                ]
            }
        },
    )
    assert result is image
    assert calls[0]["prompt"] == "cat face"


@pytest.mark.parametrize("module", ["shuffle", "inpaint", "inpaint_only"])
def test_model_free_preprocessors(module, controlnet_dependencies):
    from ..controlnet import detect
    from ..utils import encode_pil_to_base64

    encoded = encode_pil_to_base64(Image.new("RGB", (64, 64), "white"))
    result = detect(
        controlnet_module=module,
        controlnet_input_images=[encoded],
        controlnet_masks=[encoded],
        controlnet_processor_res=64,
    )
    assert len(result["images"]) == 1
