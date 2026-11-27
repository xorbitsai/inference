# CI: linux

import pickle
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from .. import vllm as adapter
from ..vllm import VLLMLightOnOCR3Model


@pytest.mark.parametrize(
    "name, prompt, longest_edge, context_length, template_options, content",
    [
        ("LightOnOCR-3-1B", "", 1540, 16384, {}, [{"type": "image"}]),
        (
            "LightOnOCR-3-0.8B",
            "grounding",
            2048,
            18000,
            {"enable_thinking": False},
            [{"type": "image"}, {"type": "text", "text": "grounding"}],
        ),
    ],
)
def test_vllm_lightonocr3_request(
    monkeypatch, name, prompt, longest_edge, context_length, template_options, content
):
    calls, threads = {}, []
    output = "![text](80,175,920,225) Receipt"

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.update(messages=messages, template_kwargs=kwargs)
            return "rendered prompt"

    class Engine:
        def get_tokenizer(self):
            threads.append(threading.get_ident())
            return Tokenizer()

        def generate(self, inputs, sampling):
            threads.append(threading.get_ident())
            calls.update(inputs=inputs, sampling=sampling)
            return [SimpleNamespace(outputs=[SimpleNamespace(text=f"  {output}  ")])]

        def shutdown(self):
            threads.append(threading.get_ident())

    def load(path, config):
        threads.append(threading.get_ident())
        calls.update(path=path, config=config)
        return Engine()

    module = ModuleType("vllm")
    module.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(adapter, "_load_vllm_model", load)
    model = VLLMLightOnOCR3Model(
        "ocr-test",
        "/models/lightonocr",
        model_spec=SimpleNamespace(model_name=name, is_builtin=True),
        gpu_memory_utilization=0.5,
    )
    model = pickle.loads(pickle.dumps(model))
    image = Image.new("RGBA", (2400, 1200))
    try:
        assert (
            model.ocr(
                image, prompt=prompt, max_tokens=64, request_id="request", top_k=10
            )
            == output
        )
        assert calls["path"] == "/models/lightonocr"
        assert calls["config"] == {
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.5,
            "max_model_len": context_length,
            "limit_mm_per_prompt": {"image": 1, "video": 0},
            "mm_processor_cache_gb": 0,
            "enable_prefix_caching": False,
        }
        assert calls["messages"] == [{"role": "user", "content": content}]
        assert calls["template_kwargs"] == {
            "tokenize": False,
            "add_generation_prompt": True,
            **template_options,
        }
        [request] = calls["inputs"]
        assert request["prompt"] == "rendered prompt"
        prepared = request["multi_modal_data"]["image"]
        assert prepared.mode == "RGB"
        assert prepared.size == (longest_edge, longest_edge // 2)
        assert image.size == (2400, 1200)
        assert calls["sampling"] == {"max_tokens": 64, "temperature": 0.1, "top_k": 10}
    finally:
        model.stop()
    assert len(threads) == 4 and len(set(threads)) == 1
    assert threads[0] != threading.get_ident()
    assert model._engine_executor is None
    assert model._model is None
    assert model._processor is None


def test_vllm_lightonocr3_tokenizer_failure_releases_engine(monkeypatch):
    calls = []

    class Engine:
        def get_tokenizer(self):
            raise RuntimeError("Tokenizer failed")

        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(adapter, "_load_vllm_model", lambda *args: Engine())
    model = VLLMLightOnOCR3Model("ocr-test", "/models/lightonocr")
    with pytest.raises(RuntimeError, match="Tokenizer failed"):
        model.load()
    assert calls == ["shutdown"]
    assert model._engine_executor is None
    assert model._model is None
    assert model._processor is None
