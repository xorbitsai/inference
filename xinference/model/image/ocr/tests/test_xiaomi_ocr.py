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
from ..xiaomi_ocr import XiaomiOCRModel


def test_xiaomi_ocr_metadata_and_factory(monkeypatch):
    families = {}
    path = Path(__file__).parents[2] / "models" / "xiaomi-ocr-0.json"
    load_model_family_from_json(str(path), families)
    variants = families["xiaomi-ocr-0"]
    assert len(variants) == 2
    register_builtin_ocr_engines()
    engines = {}
    for family in variants:
        assert family.model_format == "pytorch"
        assert family.model_family == "ocr"
        assert family.model_ability == ["ocr"]
        assert family.model_id == "SeerRay-Lab/Xiaomi-OCR-0"
        assert (
            family.model_revision
            == {
                "huggingface": "main",
                "modelscope": "master",
            }[family.model_hub]
        )
        assert any(
            package.startswith("transformers>=5.8.1")
            for package in family.virtualenv.packages
        )
        generate_engine_config_by_model_name(family, engines)

    assert set(engines["xiaomi-ocr-0"]) == {"transformers"}
    assert engines["xiaomi-ocr-0"]["transformers"][0]["ocr_class"] is XiaomiOCRModel
    monkeypatch.setitem(BUILTIN_IMAGE_MODELS, "xiaomi-ocr-0", variants)
    for hub in ("huggingface", "modelscope"):
        model = create_image_model_instance(
            "xiaomi-test",
            "xiaomi-ocr-0",
            download_hub=hub,
            model_path="/models/xiaomi",
            enable_virtual_env=False,
        )
        assert isinstance(model, XiaomiOCRModel)
        assert model.model_family.model_hub == hub
        assert model.model_ability == ["ocr"]

    assert not XiaomiOCRModel.match(
        SimpleNamespace(model_name="xiaomi-ocr-0", model_format="ggufv2")
    )
    assert not XiaomiOCRModel.match(
        SimpleNamespace(model_name="jina-ocr-v1", model_format="pytorch")
    )


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
def test_xiaomi_ocr_default_prompt(prompt):
    assert XiaomiOCRModel._normalize_prompt(prompt) == XiaomiOCRModel.DEFAULT_PROMPT


@pytest.mark.parametrize("device_map", [None, "auto"])
def test_xiaomi_ocr_load_and_infer(monkeypatch, device_map):
    class FakeInputs(dict):
        def to(self, device):
            self.device = device
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.template_kwargs = kwargs
            self.inputs = FakeInputs(input_ids=torch.tensor([[1, 2]]))
            return self.inputs

        def batch_decode(self, tokens, **kwargs):
            self.tokens = tokens
            self.decode_kwargs = kwargs
            return ["  # Recognized text\n<fcel>Table<nl>  "]

    class FakeModel:
        device = torch.device("cpu")

        def to(self, device):
            assert device_map is None
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            assert torch.is_inference_mode_enabled()
            return torch.tensor([[1, 2, 3, 4]])

    processor, loaded_model = FakeProcessor(), FakeModel()
    processor_loader, model_loader = {}, {}

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            processor_loader.update(path=path, **kwargs)
            return processor

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            model_loader.update(path=path, **kwargs)
            return loaded_model

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = FakeAutoProcessor
    transformers.AutoModelForImageTextToText = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    load_kwargs = (
        {"device_map": device_map, "dtype": torch.float16} if device_map else {}
    )
    model = XiaomiOCRModel(
        "xiaomi-test",
        "/models/xiaomi",
        device="cpu",
        model_spec=SimpleNamespace(model_ability=["ocr"], is_builtin=True),
        cpu_offload=True,
        **load_kwargs,
    )
    image = Image.new("RGBA", (8, 8))
    result = model.ocr(
        image,
        max_new_tokens=32,
        request_id="request",
        model_size="gundam",
        test_compress=False,
        save_results=False,
        save_dir=None,
        eval_mode=True,
    )
    assert result == "# Recognized text\n<fcel>Table<nl>"
    assert processor_loader == {"path": "/models/xiaomi", "trust_remote_code": True}
    assert model_loader == {
        "path": "/models/xiaomi",
        "trust_remote_code": True,
        "dtype": torch.float16 if device_map else torch.float32,
        **({"device_map": device_map} if device_map else {}),
    }
    assert loaded_model.evaluated
    assert processor.messages[0]["role"] == "user"
    content = processor.messages[0]["content"]
    assert content[0]["image"].mode == "RGB"
    assert content[1] == {"type": "text", "text": model.DEFAULT_PROMPT}
    assert processor.template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": False,
    }
    assert processor.inputs.device == loaded_model.device
    assert set(loaded_model.generate_kwargs) == {
        "input_ids",
        "max_new_tokens",
        "do_sample",
    }
    assert loaded_model.generate_kwargs["max_new_tokens"] == 32
    assert loaded_model.generate_kwargs["do_sample"] is False
    assert torch.equal(processor.tokens, torch.tensor([[3, 4]]))
    assert processor.decode_kwargs["skip_special_tokens"] is True

    model.ocr(image, prompt="<image>\nExtract invoice fields as JSON.", do_sample=True)
    assert (
        processor.messages[0]["content"][1]["text"] == "Extract invoice fields as JSON."
    )
    assert loaded_model.generate_kwargs["max_new_tokens"] == 4096
    assert loaded_model.generate_kwargs["do_sample"] is True


def test_xiaomi_ocr_rejects_invalid_image_without_loading(monkeypatch):
    model = XiaomiOCRModel("xiaomi-test")
    assert model.model_ability == []

    def unexpected_load():
        pytest.fail("Invalid images must be rejected before loading weights")

    monkeypatch.setattr(model, "load", unexpected_load)
    with pytest.raises(ValueError, match="Input must be a PIL Image"):
        model.ocr("not an image")
