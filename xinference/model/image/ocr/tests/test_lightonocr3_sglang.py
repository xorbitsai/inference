# CI: linux

import pickle
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from ..sglang import SGLangLightOnOCR3Model


@pytest.fixture
def tokenizer(monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.template_kwargs = kwargs
            return "rendered prompt"

    tokenizer = Tokenizer()
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda path, **kwargs: tokenizer
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return tokenizer


@pytest.mark.parametrize("variant", ["0.8B", "4B"])
def test_sglang_offline_ocr(monkeypatch, tokenizer, variant):
    thread_ids = []
    calls = []

    class Engine:
        def __init__(self, **kwargs):
            thread_ids.append(threading.get_ident())
            assert kwargs["model_path"] == "/model"
            assert kwargs["context_length"] == 18000
            assert kwargs["mem_fraction_static"] == 0.65
            assert "device_map" not in kwargs

        def generate(self, prompt, image_data, sampling_params):
            thread_ids.append(threading.get_ident())
            assert prompt == "rendered prompt"
            assert image_data.mode == "RGB"
            assert image_data.width == 2048
            calls.append(sampling_params)
            return {"text": "  ![text](0,0,1000,1000) OCR  "}

        def shutdown(self):
            thread_ids.append(threading.get_ident())

    sglang = ModuleType("sglang")
    sglang.Engine = Engine
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    family = SimpleNamespace(
        model_name=f"LightOnOCR-3-{variant}", model_format="pytorch"
    )
    assert SGLangLightOnOCR3Model.match(family)
    model = SGLangLightOnOCR3Model(
        "lightonocr3",
        "/model",
        model_spec=family,
        device_map="auto",
        mem_fraction_static=0.65,
    )
    model = pickle.loads(pickle.dumps(model))
    image = Image.new("L", (2200, 100))
    assert model.ocr(image) == "![text](0,0,1000,1000) OCR"
    assert tokenizer.messages == [{"role": "user", "content": [{"type": "image"}]}]
    model.ocr(
        image,
        prompt="grounding",
        max_tokens=128,
        do_sample=False,
        temperature=0.7,
        seed=123,
        request_id="1",
    )
    assert tokenizer.messages[0]["content"][-1] == {"type": "text", "text": "grounding"}
    assert tokenizer.template_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    model.ocr(image, temperature=0.7, seed=123, sampling_seed=0)
    model.stop()
    assert calls == [
        {"max_new_tokens": 4096, "temperature": 0.1},
        {"max_new_tokens": 128, "temperature": 0.0, "sampling_seed": 123},
        {"max_new_tokens": 4096, "temperature": 0.7, "sampling_seed": 0},
    ]
    assert len(thread_ids) == 5
    assert len(set(thread_ids)) == 1
    assert model._engine_executor is None


def test_sglang_excludes_lightonocr3_1b():
    family = SimpleNamespace(model_name="LightOnOCR-3-1B", model_format="pytorch")
    assert not SGLangLightOnOCR3Model.match(family)


def test_sglang_cleans_up_after_failed_load(monkeypatch, tokenizer):
    def fail(**kwargs):
        raise RuntimeError("load failed")

    sglang = ModuleType("sglang")
    sglang.Engine = fail
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    model = SGLangLightOnOCR3Model("lightonocr3", "/model")
    with pytest.raises(RuntimeError, match="load failed"):
        model.load()
    assert model._model is None
    assert model._processor is None
    assert model._engine_executor is None
