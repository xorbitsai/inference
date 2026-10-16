import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image

from ... import load_model_family_from_json
from ...core import BUILTIN_IMAGE_MODELS, _select_ocr_model_family
from .. import register_builtin_ocr_engines
from ..jina_ocr import DEFAULT_PROMPT, JinaOCRModel
from ..ocr_family import OCR_ENGINES, generate_engine_config_by_model_name


def _families():
    families = {}
    path = Path(__file__).parents[2] / "models" / "jina-ocr-v1.json"
    load_model_family_from_json(str(path), families)
    return families["jina-ocr-v1"]


def test_jina_ocr_metadata_and_engine_selection(monkeypatch):
    families = _families()
    assert len(families) == 2
    assert all(f.model_format == "pytorch" for f in families)
    pytorch = {f.model_hub: f for f in families}
    assert pytorch["huggingface"].model_id == "jinaai/jina-ocr-v1"
    assert pytorch["huggingface"].model_revision == "main"
    assert pytorch["modelscope"].model_id == "jinaai/jina-ocr-v1"
    assert pytorch["modelscope"].model_revision == "master"

    monkeypatch.setitem(BUILTIN_IMAGE_MODELS, "jina-ocr-v1", families)
    assert (
        _select_ocr_model_family("jina-ocr-v1", "transformers", "modelscope")
        is pytorch["modelscope"]
    )
    with pytest.raises(ValueError, match="Image OCR model not found"):
        _select_ocr_model_family(
            "jina-ocr-v1", "llama.cpp", "huggingface", model_format="ggufv2"
        )

    previous = OCR_ENGINES.pop("jina-ocr-v1", None)
    try:
        register_builtin_ocr_engines()
        for family in families:
            generate_engine_config_by_model_name(family)
        assert (
            OCR_ENGINES["jina-ocr-v1"]["transformers"][0]["ocr_class"] is JinaOCRModel
        )
        assert "llama.cpp" not in OCR_ENGINES["jina-ocr-v1"]
    finally:
        OCR_ENGINES.pop("jina-ocr-v1", None)
        if previous is not None:
            OCR_ENGINES["jina-ocr-v1"] = previous


def test_jina_ocr_transformers_load_and_infer(monkeypatch):
    class FakeProcessor:
        def prepare_ocr_inputs(self, image, **kwargs):
            self.image = image
            self.prepare_kwargs = kwargs
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode_ocr(self, output, input_ids):
            self.decoded = output, input_ids
            return "recognized text"

    class FakeModel:
        device = torch.device("cpu")

        def to(self, device):
            self.loaded_device = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return torch.tensor([[1, 2, 3]])

    processor, loaded_model = FakeProcessor(), FakeModel()

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert (path, kwargs) == ("/models/jina", {"trust_remote_code": True})
            return processor

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == "/models/jina"
            assert kwargs == {"trust_remote_code": True, "dtype": torch.float32}
            return loaded_model

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_transformers.AutoModelForCausalLM = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = JinaOCRModel(
        "jina-test",
        "/models/jina",
        device="cpu",
        model_spec=SimpleNamespace(model_ability=["ocr"], is_builtin=True),
        cpu_offload=True,
    )
    image = Image.new("RGBA", (2, 3), "red")
    assert (
        model.ocr(image, max_new_tokens=32, request_id="request") == "recognized text"
    )
    assert processor.image.mode == "RGB"
    assert processor.prepare_kwargs == {
        "prompt": DEFAULT_PROMPT,
        "device": torch.device("cpu"),
    }
    assert loaded_model.generate_kwargs["max_new_tokens"] == 32
    assert loaded_model.generate_kwargs["do_sample"] is False
    assert "request_id" not in loaded_model.generate_kwargs
    assert model.ocr(image, prompt="Custom prompt") == "recognized text"
    assert processor.prepare_kwargs["prompt"] == "Custom prompt"
    assert (
        model.ocr(
            image,
            prompt="<image>\nFree OCR. Extract all text content from the image.",
            model_size="gundam",
            test_compress=False,
            save_results=False,
            eval_mode=True,
        )
        == "recognized text"
    )
    assert processor.prepare_kwargs["prompt"] == (
        "Free OCR. Extract all text content from the image."
    )
    assert set(loaded_model.generate_kwargs) == {
        "input_ids",
        "max_new_tokens",
        "do_sample",
    }
    assert loaded_model.generate_kwargs["max_new_tokens"] == 4096
    assert loaded_model.generate_kwargs["do_sample"] is False
