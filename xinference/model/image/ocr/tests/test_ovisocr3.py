# CI: linux

import pickle
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image

from ... import load_model_family_from_json
from ...core import create_image_model_instance
from .. import register_builtin_ocr_engines
from ..ocr_family import generate_engine_config_by_model_name
from ..ovisocr3 import OvisOCR3Model
from ..sglang import SGLangOvisOCR3Model
from ..vllm import VLLMOvisOCR3Model

MODELS = {
    "transformers": OvisOCR3Model,
    "vllm": VLLMOvisOCR3Model,
    "sglang": SGLangOvisOCR3Model,
}


def test_metadata_and_factory():
    families = {}
    path = Path(__file__).parents[2] / "models" / "OvisOCR3.json"
    load_model_family_from_json(str(path), families)
    variants = families["OvisOCR3"]
    assert {f.model_hub: f.model_revision for f in variants} == {
        "huggingface": "main",
        "modelscope": "master",
    }
    register_builtin_ocr_engines()
    engines = {}
    for family in variants:
        assert family.model_id == "ATH-MaaS/OvisOCR3"
        assert family.model_family == "ocr"
        assert family.model_ability == ["ocr"]
        assert family.model_format == "pytorch"
        generate_engine_config_by_model_name(family, engines)
    assert set(engines["OvisOCR3"]) == set(MODELS)
    for hub in ("huggingface", "modelscope"):
        for engine, cls in MODELS.items():
            assert engines["OvisOCR3"][engine][0]["ocr_class"] is cls
            model = create_image_model_instance(
                "ovis-test",
                "OvisOCR3",
                download_hub=hub,
                model_engine=engine,
                model_path="/models/ovis",
                enable_virtual_env=False,
            )
            try:
                assert isinstance(model, cls)
                assert model.model_family.model_hub == hub
                assert model.model_ability == ["ocr"]
            finally:
                if engine != "transformers":
                    model.stop()
    for cls in MODELS.values():
        assert not cls.match(
            SimpleNamespace(model_name="OvisOCR2", model_format="pytorch")
        )
        assert not cls.match(
            SimpleNamespace(model_name="OvisOCR3", model_format="ggufv2")
        )


@pytest.mark.parametrize("cls", MODELS.values())
def test_rejects_invalid_image_before_loading(cls, monkeypatch):
    model = cls("ovis-test")
    monkeypatch.setattr(model, "load", lambda: pytest.fail("Unexpected model load"))
    try:
        with pytest.raises(ValueError, match="Input must be a PIL Image"):
            model.ocr("invalid")
    finally:
        if cls is not OvisOCR3Model:
            model.stop()


@pytest.mark.parametrize("prompt", [None, "", "OCR", "<image>\nFree OCR."])
def test_default_prompt(prompt):
    assert OvisOCR3Model._normalize_prompt(prompt) == OvisOCR3Model.DEFAULT_PROMPT


def test_only_markdown_is_postprocessed():
    raw = '  first\n\n<img src="images/bbox_1_2_3_4.jpg" />\n\nlast  '
    assert (
        OvisOCR3Model._process_output(raw, OvisOCR3Model.DEFAULT_PROMPT, True)
        == "first\n\nlast"
    )
    assert (
        OvisOCR3Model._process_output(raw, OvisOCR3Model.DEFAULT_PROMPT, False)
        == raw.strip()
    )
    assert OvisOCR3Model._process_output(raw, "Extract JSON.", True) == raw
    prefix = "x" * 8000 + "\n"
    repeated = prefix + "abcdef" * 30
    assert (
        OvisOCR3Model._process_output(repeated, OvisOCR3Model.DEFAULT_PROMPT, True)
        == prefix + "abcdef"
    )
    assert OvisOCR3Model._process_output(repeated, "Plain OCR.", True) == repeated


@pytest.mark.parametrize("cls", [VLLMOvisOCR3Model, SGLangOvisOCR3Model])
def test_model_can_be_serialized_before_loading(cls):
    model = pickle.loads(pickle.dumps(cls("ovis-test")))
    assert model._engine_executor is None
    model.stop()


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    raw = '  {"content": "invoice"}  '

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        if kwargs["tokenize"]:
            self.inputs = FakeInputs(input_ids=torch.tensor([[1, 2]]))
            return self.inputs
        return "rendered prompt"

    def batch_decode(self, tokens, **kwargs):
        assert tokens.tolist() == [[3, 4]]
        assert kwargs["skip_special_tokens"] is True
        return [self.raw]


@pytest.mark.parametrize("device_map", [None, "auto"])
def test_transformers_load_and_infer(monkeypatch, device_map):
    processor = FakeProcessor()
    calls = {}

    class FakeModel:
        device = torch.device("cpu")

        def to(self, device):
            assert device_map is None
            self.device = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert torch.is_inference_mode_enabled()
            calls["generate"] = kwargs
            return torch.tensor([[1, 2, 3, 4]])

    loaded = FakeModel()

    def load_processor(path, **kwargs):
        calls["processor"] = (path, kwargs)
        return processor

    def load_model(path, **kwargs):
        calls["model"] = (path, kwargs)
        return loaded

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=load_processor)
    transformers.AutoModelForImageTextToText = SimpleNamespace(
        from_pretrained=load_model
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    model = OvisOCR3Model(
        "ovis-test",
        "/models/ovis",
        device="cpu",
        model_spec=SimpleNamespace(model_ability=["ocr"], is_builtin=True),
        cpu_offload=True,
        **({"device_map": device_map, "dtype": torch.bfloat16} if device_map else {}),
    )
    image = Image.new("RGBA", (8, 8))
    assert (
        model.ocr(
            image,
            prompt="<image>\nExtract JSON.",
            request_id="r",
            model_size="gundam",
            max_tokens=32,
        )
        == processor.raw
    )
    assert calls["processor"] == (
        "/models/ovis",
        {
            "trust_remote_code": True,
            "min_pixels": 448 * 448,
            "max_pixels": 2880 * 2880,
        },
    )
    assert calls["model"] == (
        "/models/ovis",
        {
            "trust_remote_code": True,
            "dtype": torch.bfloat16 if device_map else torch.float32,
            **({"device_map": device_map} if device_map else {}),
        },
    )
    assert set(calls["generate"]) == {"input_ids", "max_new_tokens", "do_sample"}
    assert calls["generate"]["max_new_tokens"] == 32
    assert calls["generate"]["do_sample"] is False
    assert processor.messages[0]["content"][0]["image"].mode == "RGB"
    assert processor.messages[0]["content"][1]["text"] == "Extract JSON."
    assert processor.template_kwargs["enable_thinking"] is False
    assert processor.inputs.device == loaded.device
    model.ocr(image)
    assert calls["generate"]["max_new_tokens"] == 16384
    assert processor.messages[0]["content"][1]["text"] == model.DEFAULT_PROMPT


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_offline_engine_load_infer_and_stop(monkeypatch, engine):
    from .. import vllm as vllm_ocr

    processor = FakeProcessor()
    calls = {}

    class FakeEngine:
        def get_tokenizer(self):
            return processor

        def generate(self, *args, **kwargs):
            calls["generate_thread"] = threading.get_ident()
            calls["args"], calls["kwargs"] = args, kwargs
            if engine == "vllm":
                return [SimpleNamespace(outputs=[SimpleNamespace(text=processor.raw)])]
            return {"text": processor.raw}

        def shutdown(self):
            calls["stop_thread"] = threading.get_ident()

    def load_engine(*args, **kwargs):
        calls["load_thread"] = threading.get_ident()
        calls["load_args"], calls["load_kwargs"] = args, kwargs
        return FakeEngine()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    vllm = ModuleType("vllm")
    vllm.SamplingParams = FakeSamplingParams
    sglang = ModuleType("sglang")
    sglang.Engine = load_engine
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *a, **k: processor
    )
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(vllm_ocr, "_load_vllm_model", load_engine)
    model = MODELS[engine](
        "ovis-test",
        "/models/ovis",
        model_spec=SimpleNamespace(model_ability=["ocr"], is_builtin=True),
    )
    try:
        assert (
            model.ocr(
                Image.new("RGBA", (8, 8)),
                "Extract JSON.",
                request_id="r",
                max_tokens=32,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            == processor.raw
        )
        assert processor.template_kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if engine == "vllm":
            assert calls["load_args"][1] == {
                "gpu_memory_utilization": 0.8,
                "gdn_prefill_backend": "triton",
                "limit_mm_per_prompt": {"image": 1},
            }
            request = calls["args"][0][0]
            assert request["multi_modal_data"]["image"][0].mode == "RGB"
            assert request["mm_processor_kwargs"]["images_kwargs"] == {
                "min_pixels": 448 * 448,
                "max_pixels": 2880 * 2880,
            }
            assert calls["args"][1].kwargs == {
                "max_tokens": 32,
                "temperature": 0.0,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            }
        else:
            assert calls["load_kwargs"] == {
                "model_path": "/models/ovis",
                "trust_remote_code": True,
                "mem_fraction_static": 0.8,
                "mm_process_config": {
                    "image": {"min_pixels": 448 * 448, "max_pixels": 2880 * 2880}
                },
            }
            assert calls["kwargs"]["image_data"].mode == "RGB"
            assert calls["kwargs"]["sampling_params"] == {
                "max_new_tokens": 32,
                "temperature": 0.0,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            }
        assert calls["generate_thread"] == calls["load_thread"]
        assert calls["generate_thread"] != threading.get_ident()
        assert model.ocr(Image.new("RGB", (8, 8))) == processor.raw.strip()
    finally:
        model.stop()
    assert calls["stop_thread"] == calls["load_thread"]
    assert model._model is None
    model.stop()
