# CI: linux

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image

from ... import load_model_family_from_json
from ...core import BUILTIN_IMAGE_MODELS, create_image_model_instance
from .. import register_builtin_ocr_engines
from ..ocr_family import generate_engine_config_by_model_name
from ..unlimited_ocr import UnlimitedOCRModel
from ..unlimited_ocr2 import UnlimitedOCR2Model


def test_unlimited_ocr2_metadata_and_factory(monkeypatch):
    families = {}
    path = Path(__file__).parents[2] / "models" / "Unlimited-OCR-2.json"
    load_model_family_from_json(str(path), families)
    variants = families["Unlimited-OCR-2"]
    assert len(variants) == 2
    register_builtin_ocr_engines()
    engines = {}
    for family in variants:
        assert family.model_format == "pytorch"
        assert family.model_family == "ocr"
        assert family.model_ability == ["ocr"]
        assert (family.model_id, family.model_revision) == {
            "huggingface": ("baidu/Unlimited-OCR-2", "main"),
            "modelscope": ("PaddlePaddle/Unlimited-OCR-2", "master"),
        }[family.model_hub]
        assert 'transformers==4.52.1 ; #engine# == "transformers"' in (
            family.virtualenv.packages
        )
        assert '#system_torchvision# ; #engine# == "transformers"' in (
            family.virtualenv.packages
        )
        generate_engine_config_by_model_name(family, engines)
        assert not UnlimitedOCRModel.match(family)
    assert set(engines["Unlimited-OCR-2"]) == {"transformers"}
    assert engines["Unlimited-OCR-2"]["transformers"][0]["ocr_class"] is (
        UnlimitedOCR2Model
    )
    monkeypatch.setitem(BUILTIN_IMAGE_MODELS, "Unlimited-OCR-2", variants)
    for hub in ("huggingface", "modelscope"):
        model = create_image_model_instance(
            "unlimited-test",
            "Unlimited-OCR-2",
            download_hub=hub,
            model_path="/models/unlimited-ocr2",
            enable_virtual_env=False,
        )
        assert isinstance(model, UnlimitedOCR2Model)
        assert model.model_family.model_hub == hub
        assert model.model_ability == ["ocr"]
    assert not UnlimitedOCR2Model.match(
        SimpleNamespace(model_name="Unlimited-OCR", model_format="pytorch")
    )
    assert not UnlimitedOCR2Model.match(
        SimpleNamespace(model_name="Unlimited-OCR-2", model_format="ggufv2")
    )


@pytest.mark.parametrize("device_map", [None, "auto"])
def test_unlimited_ocr2_load(monkeypatch, device_map):
    calls = {}

    class FakeModel:
        def to(self, device):
            assert device_map is None
            calls["device"] = str(device)
            return self

        def eval(self):
            calls["eval"] = True
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model"] = {"path": path, **kwargs}
            return FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["tokenizer"] = {"path": path, **kwargs}
            return object()

    transformers = ModuleType("transformers")
    transformers.AutoModel = FakeAutoModel
    transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    kwargs = {"device_map": device_map} if device_map else {}
    model = UnlimitedOCR2Model(
        "unlimited-test",
        "/models/unlimited-ocr2",
        device="cuda:1",
        model_spec=SimpleNamespace(model_ability=["ocr"], is_builtin=True),
        cpu_offload=True,
        **kwargs,
    )
    model.load()
    assert calls["tokenizer"] == {
        "path": "/models/unlimited-ocr2",
        "trust_remote_code": True,
    }
    assert calls["model"] == {
        "path": "/models/unlimited-ocr2",
        "trust_remote_code": True,
        "use_safetensors": True,
        "torch_dtype": torch.bfloat16,
        **kwargs,
    }
    assert calls.get("device") == (None if device_map else "cuda:1")
    assert calls["eval"] is True


@pytest.mark.parametrize(
    "prompt",
    [
        None,
        "",
        "OCR",
        "<image>\nFree OCR.",
        "<image>\nFree OCR. Extract all text content from the image.",
    ],
)
def test_unlimited_ocr2_default_prompt(prompt):
    assert UnlimitedOCR2Model._normalize_prompt(prompt) == "Multi page parsing."


@pytest.mark.parametrize("multi_page", [False, True])
def test_unlimited_ocr2_infer(monkeypatch, multi_page, tmp_path):
    calls = []

    class FakeModel:
        # Match the upstream infer signature: unexpected API fields must not leak.
        def infer(
            self,
            tokenizer,
            prompt="",
            image_file="",
            output_path="",
            base_size=1024,
            image_size=640,
            crop_mode=None,
            test_compress=False,
            save_results=False,
            eval_mode=False,
            max_length=131072,
            max_new_tokens=None,
            temperature=0.0,
        ):
            assert tokenizer is model._tokenizer
            assert torch.is_inference_mode_enabled()
            calls.append(locals())
            return "  <PAGE>First page\n<PAGE>Second page  "

    model = UnlimitedOCR2Model("unlimited-test")

    def load():
        model._model = FakeModel()
        model._tokenizer = object()

    monkeypatch.setattr(model, "load", load)
    image = Image.new("RGBA", (8, 8))
    images = [image, Image.new("L", (16, 16))] if multi_page else image
    result = model.ocr(
        images,
        prompt="OCR",
        model_size="gundam",
        max_new_tokens=32,
        temperature=0.5,
        request_id="request",
        test_compress=False,
        save_results=False,
        save_dir=None,
        eval_mode=True,
    )
    assert result == "<PAGE>First page\n<PAGE>Second page"
    call = calls[-1]
    assert call["prompt"] == "Multi page parsing."
    assert call["base_size"] == 1024
    assert call["image_size"] == (1024 if multi_page else 640)
    assert call["crop_mode"] is (not multi_page)
    assert call["max_length"] == 131072
    assert call["max_new_tokens"] == 32
    assert call["temperature"] == 0.5
    assert call["eval_mode"] is True
    assert call["save_results"] is False
    assert call["output_path"] == ""
    rgb_images = call["image_file"] if multi_page else [call["image_file"]]
    assert [img.mode for img in rgb_images] == ["RGB"] * len(rgb_images)
    assert [img.size for img in rgb_images] == (
        [(8, 8), (16, 16)] if multi_page else [(8, 8)]
    )

    model.ocr(
        images,
        prompt="<image>\nExtract invoice fields as JSON.",
        model_size="base",
        save_results=True,
        save_dir=str(tmp_path),
        return_dict=True,
    )
    assert calls[-1]["prompt"] == "Extract invoice fields as JSON."
    assert calls[-1]["crop_mode"] is False
    assert calls[-1]["image_size"] == 1024
    assert calls[-1]["save_results"] is True
    assert calls[-1]["output_path"] == str(tmp_path)
    assert image.mode == "RGBA"


@pytest.mark.parametrize("image", ["invalid", [], [Image.new("RGB", (8, 8)), None]])
def test_unlimited_ocr2_rejects_invalid_image(monkeypatch, image):
    model = UnlimitedOCR2Model("unlimited-test")

    def unexpected_load():
        pytest.fail("Invalid images must be rejected before loading weights")

    monkeypatch.setattr(model, "load", unexpected_load)
    with pytest.raises(ValueError, match="Input must be a PIL Image"):
        model.ocr(image)


def test_unlimited_ocr2_requires_save_dir():
    model = UnlimitedOCR2Model("unlimited-test")
    with pytest.raises(ValueError, match="save_dir must be provided"):
        model.ocr(Image.new("RGB", (8, 8)), save_results=True)


def test_unlimited_ocr2_requires_cuda():
    model = UnlimitedOCR2Model("unlimited-test", device="cpu")
    with pytest.raises(ValueError, match="requires a CUDA device"):
        model.load()
