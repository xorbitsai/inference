# CI: linux

import pickle
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from .....core.exceptions import InvalidModelRequestError
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
            self.server_args = SimpleNamespace(
                enable_deterministic_inference=False,
                resolved_dict=lambda: {"enable_deterministic_inference": True},
            )
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


@pytest.fixture
def nondeterministic_model(tokenizer):
    model = SGLangLightOnOCR3Model("lightonocr3", "/model")
    model._processor = tokenizer
    model._model = SimpleNamespace(
        server_args=SimpleNamespace(
            enable_deterministic_inference=True,
            resolved_dict=Mock(return_value={"enable_deterministic_inference": False}),
        ),
        tokenizer_manager=SimpleNamespace(
            preferred_sampling_params={},
            sampling_params_class=Mock(
                return_value=SimpleNamespace(top_k=2, max_new_tokens=2, beam_width=None)
            ),
        ),
        generate=Mock(return_value={"text": "OCR"}),
        shutdown=lambda: None,
    )
    try:
        yield model
    finally:
        model.stop()


@pytest.mark.parametrize(
    "seed_kwargs, expected_seed",
    [
        ({"seed": 123}, 123),
        ({"sampling_seed": 0}, 0),
        ({"seed": 123, "sampling_seed": None}, 123),
    ],
)
def test_sglang_rejects_seeded_sampling_when_disabled(
    monkeypatch, nondeterministic_model, seed_kwargs, expected_seed
):
    monkeypatch.setenv("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "1")
    engine = nondeterministic_model._model
    with pytest.raises(
        InvalidModelRequestError, match="enable_deterministic_inference=True"
    ):
        nondeterministic_model.ocr(
            Image.new("RGB", (8, 8)), temperature=0.7, max_tokens=2, **seed_kwargs
        )
    engine.tokenizer_manager.sampling_params_class.assert_called_once_with(
        temperature=0.7, max_new_tokens=2, sampling_seed=expected_seed
    )
    engine.generate.assert_not_called()


@pytest.mark.parametrize(
    "request_options, temperature",
    [
        ({"do_sample": False, "temperature": 0.7}, 0.0),
        ({"temperature": 0.0}, 0.0),
        ({"temperature": 0.7, "top_k": 1}, 0.7),
    ],
)
def test_sglang_preserves_seeded_greedy(
    nondeterministic_model, request_options, temperature
):
    engine = nondeterministic_model._model
    sampling_params_class = engine.tokenizer_manager.sampling_params_class
    sampling_params_class.return_value.top_k = 1
    assert (
        nondeterministic_model.ocr(
            Image.new("RGB", (8, 8)), seed=123, max_tokens=2, **request_options
        )
        == "OCR"
    )
    params = engine.generate.call_args.kwargs["sampling_params"]
    assert params["temperature"] == temperature
    assert params["sampling_seed"] == 123


def test_sglang_unseeded_sampling_skips_deterministic_check(nondeterministic_model):
    engine = nondeterministic_model._model
    assert nondeterministic_model.ocr(Image.new("RGB", (8, 8)), seed=None) == "OCR"
    engine.server_args.resolved_dict.assert_not_called()
    params = engine.generate.call_args.kwargs["sampling_params"]
    assert params == {"temperature": 0.1, "max_new_tokens": 4096}


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
