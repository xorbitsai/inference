# CI: linux

import asyncio
import base64
import io
import pickle
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image

from ... import load_model_family_from_json
from ...cache_manager import ImageCacheManager
from ...core import BUILTIN_IMAGE_MODELS, create_image_model_instance
from .. import register_builtin_ocr_engines
from ..minicpm_ocr import (
    LlamaCppMiniCPMOCRModel,
    MiniCPMOCRModel,
    SGLangMiniCPMOCRModel,
)
from ..ocr_family import OCR_ENGINES, generate_engine_config_by_model_name
from ..vllm import VLLMMiniCPMOCRModel

MODEL_NAME = "MiniCPM-OCR-1.0"
ENGINE_CLASSES = {
    "transformers": MiniCPMOCRModel,
    "vllm": VLLMMiniCPMOCRModel,
    "sglang": SGLangMiniCPMOCRModel,
    "llama.cpp": LlamaCppMiniCPMOCRModel,
}


@pytest.mark.parametrize("model_class", ENGINE_CLASSES.values())
def test_model_can_be_serialized_before_loading(model_class):
    model = model_class("minicpm-test", "/models/minicpm")
    restored = pickle.loads(pickle.dumps(model))
    assert type(restored) is model_class
    assert restored._model is None
    assert restored._model_path == "/models/minicpm"


def _families():
    families = {}
    path = Path(__file__).parents[2] / "models" / f"{MODEL_NAME}.json"
    load_model_family_from_json(str(path), families)
    return families[MODEL_NAME]


@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
@pytest.mark.parametrize("engine", ENGINE_CLASSES)
@pytest.mark.parametrize("virtual_env", [False, True])
def test_metadata_and_factory(monkeypatch, hub, engine, virtual_env):
    families = _families()
    monkeypatch.setitem(BUILTIN_IMAGE_MODELS, MODEL_NAME, families)
    monkeypatch.setitem(OCR_ENGINES, MODEL_NAME, {})
    register_builtin_ocr_engines()
    for family in families:
        assert family.model_family == "ocr"
        assert family.model_ability == ["ocr"]
        assert (
            family.model_revision
            == {"huggingface": "main", "modelscope": "master"}[family.model_hub]
        )
        owner = "openbmb" if family.model_hub == "huggingface" else "OpenBMB"
        suffix = "-gguf" if family.model_format == "ggufv2" else ""
        assert family.model_id == f"{owner}/{MODEL_NAME}{suffix}"
        generate_engine_config_by_model_name(family)
    assert set(OCR_ENGINES[MODEL_NAME]) == set(ENGINE_CLASSES)
    gguf = engine == "llama.cpp"
    model = create_image_model_instance(
        "minicpm-test",
        MODEL_NAME,
        download_hub=hub,
        model_engine=engine,
        model_format="ggufv2" if gguf else "pytorch",
        quantization="Q4_K_M" if gguf else None,
        model_path="/models/minicpm",
        enable_virtual_env=virtual_env,
    )
    assert type(model) is ENGINE_CLASSES[engine]
    assert model.model_family.model_hub == hub
    assert model.model_family.model_engine == engine
    assert model.model_ability == ["ocr"]
    assert not ENGINE_CLASSES[engine].match(
        SimpleNamespace(model_name="other-ocr", model_format="pytorch")
    )


@pytest.mark.parametrize("gguf", [False, True])
def test_default_engine(monkeypatch, gguf):
    families = _families()
    monkeypatch.setitem(BUILTIN_IMAGE_MODELS, MODEL_NAME, families)
    monkeypatch.setitem(OCR_ENGINES, MODEL_NAME, {})
    register_builtin_ocr_engines()
    for family in families:
        generate_engine_config_by_model_name(family)
    options = {"model_format": "ggufv2", "quantization": "Q8_0"} if gguf else {}
    model = create_image_model_instance(
        "minicpm-test",
        MODEL_NAME,
        model_path="/models/minicpm",
        enable_virtual_env=False,
        **options,
    )
    assert type(model) is (LlamaCppMiniCPMOCRModel if gguf else MiniCPMOCRModel)


@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
def test_gguf_downloads_selected_quantization_and_projector(monkeypatch, tmp_path, hub):
    import huggingface_hub

    import xinference.model.utils as utils

    spec = next(
        f
        for f in _families()
        if f.model_hub == hub
        and f.model_format == "ggufv2"
        and f.quantization == "Q8_0"
    )
    manager = ImageCacheManager(spec)
    manager._cache_dir = str(tmp_path / "cache")
    calls = []

    def download(repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        result = tmp_path / filename
        result.write_bytes(b"gguf")
        return str(result)

    if hub == "huggingface":
        monkeypatch.setattr(utils, "IS_NEW_HUGGINGFACE_HUB", True)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    else:
        module = ModuleType("modelscope.hub.file_download")
        module.model_file_download = download
        monkeypatch.setitem(sys.modules, "modelscope", ModuleType("modelscope"))
        monkeypatch.setitem(sys.modules, "modelscope.hub", ModuleType("modelscope.hub"))
        monkeypatch.setitem(sys.modules, "modelscope.hub.file_download", module)
    monkeypatch.setattr(
        utils, "retry_download", lambda fn, name, info, *args, **kw: fn(*args, **kw)
    )
    assert Path(manager.cache_ocr_gguf()).is_file()
    manager.cache_ocr_gguf()
    assert calls == [
        (spec.model_id, "MiniCPM-OCR-1_0-Q8_0.gguf", spec.model_revision),
        (spec.model_id, "mmproj-model-f16-4x.gguf", spec.model_revision),
    ]


@pytest.mark.parametrize(
    "prompt",
    [None, "", "OCR", "<image>\nFree OCR. Extract all text content from the image."],
)
def test_default_prompt(prompt):
    assert MiniCPMOCRModel._normalize_prompt(prompt) == "Text Recognition:"


@pytest.mark.parametrize(
    "prompt",
    [
        "Table Recognition:",
        "Formula Recognition:",
        "Seal Recognition:",
        "Chart to Table:",
        'Extract the "date". Format the output as a JSON object.',
    ],
)
def test_custom_prompt(prompt):
    assert MiniCPMOCRModel._normalize_prompt("<image>\n" + prompt) == prompt


@pytest.mark.parametrize("model_class", ENGINE_CLASSES.values())
def test_invalid_image_rejected_before_loading(monkeypatch, model_class):
    model = model_class("minicpm-test")
    assert model.model_ability == []
    monkeypatch.setattr(model, "load", lambda: pytest.fail("Must not load weights"))
    with pytest.raises(ValueError, match="Input must be a PIL Image"):
        model.ocr("invalid")


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "image and text prompt"


@pytest.mark.parametrize("device_map", [None, "auto"])
def test_transformers_load_and_infer(monkeypatch, device_map):
    class Inputs(dict):
        def to(self, device):
            self.device = device
            return self

    class Processor(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            super().apply_chat_template(messages, **kwargs)
            self.inputs = Inputs(input_ids=torch.tensor([[1, 2]]))
            return self.inputs

        def batch_decode(self, tokens, **kwargs):
            self.tokens = tokens
            return ["  recognized text  "]

    class Model:
        device = torch.device("cpu")

        def to(self, device):
            assert device_map is None
            self.device = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            self.kwargs = kwargs
            assert torch.is_inference_mode_enabled()
            return torch.tensor([[1, 2, 3, 4]])

    processor, loaded_model = Processor(), Model()
    calls = {}

    def load_model(path, **kwargs):
        calls.update(path=path, **kwargs)
        return loaded_model

    module = ModuleType("transformers")
    module.AutoModelForImageTextToText = SimpleNamespace(from_pretrained=load_model)
    module.AutoProcessor = SimpleNamespace(from_pretrained=lambda *a, **kw: processor)
    monkeypatch.setitem(sys.modules, "transformers", module)
    model = MiniCPMOCRModel(
        "minicpm-test",
        "/models/minicpm",
        device="cpu",
        cpu_offload=True,
        **(
            {"device_map": device_map, "torch_dtype": torch.float16}
            if device_map
            else {}
        ),
    )
    result = model.ocr(
        Image.new("RGBA", (4, 4)),
        max_tokens=32,
        request_id="request",
        model_size="gundam",
        test_compress=False,
        save_results=False,
        save_dir=None,
        eval_mode=True,
    )
    assert result == "recognized text"
    assert calls == {
        "path": "/models/minicpm",
        "trust_remote_code": False,
        **(
            {"device_map": "auto", "torch_dtype": torch.float16}
            if device_map
            else {"dtype": torch.float32}
        ),
    }
    assert processor.template_kwargs["enable_thinking"] is False
    assert processor.messages[0]["content"][0]["image"].mode == "RGB"
    assert processor.messages[0]["content"][1]["text"] == "Text Recognition:"
    assert processor.inputs.device == loaded_model.device
    assert loaded_model.kwargs.keys() == {"input_ids", "max_new_tokens", "do_sample"}
    assert loaded_model.kwargs["max_new_tokens"] == 32
    assert loaded_model.kwargs["do_sample"] is False
    assert torch.equal(processor.tokens, torch.tensor([[3, 4]]))


def test_vllm_load_infer_and_stop_on_same_thread(monkeypatch):
    from .. import vllm as adapter

    threads, calls = [], {}
    tokenizer = FakeTokenizer()

    class Model:
        def get_tokenizer(self):
            threads.append(threading.get_ident())
            return tokenizer

        def generate(self, inputs, params):
            threads.append(threading.get_ident())
            calls.update(inputs=inputs, sampling=params)
            return [SimpleNamespace(outputs=[SimpleNamespace(text="  vllm text  ")])]

        def shutdown(self):
            threads.append(threading.get_ident())

    def load(path, config):
        threads.append(threading.get_ident())
        calls.update(path=path, config=config)
        return Model()

    module = ModuleType("vllm")
    module.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(adapter, "_load_vllm_model", load)
    model = VLLMMiniCPMOCRModel("minicpm-test", "/models/minicpm")

    async def run():
        await asyncio.to_thread(model.load)
        return model.ocr(
            Image.new("RGBA", (4, 4)), prompt="Table Recognition:", max_tokens=32
        )

    try:
        assert asyncio.run(run()) == "vllm text"
        assert calls["config"]["max_model_len"] == 40960
        assert calls["config"]["limit_mm_per_prompt"] == {"image": 1, "video": 0}
        assert calls["config"]["mm_processor_kwargs"] == {"downsample_mode": "4x"}
        assert calls["sampling"] == {"max_tokens": 32, "temperature": 0.0}
        assert calls["inputs"][0]["multi_modal_data"]["image"].mode == "RGB"
        assert tokenizer.messages[0]["content"][1]["text"] == "Table Recognition:"
        assert tokenizer.template_kwargs["enable_thinking"] is False
    finally:
        model.stop()
    assert len(set(threads)) == 1
    assert model._executor is None
    assert model._model is None


def test_sglang_load_infer_and_stop(monkeypatch):
    calls, threads = {}, []
    tokenizer = FakeTokenizer()

    class Engine:
        def __init__(self, **kwargs):
            threads.append(threading.get_ident())
            calls["config"] = kwargs

        def generate(self, prompt=None, sampling_params=None, image_data=None):
            threads.append(threading.get_ident())
            calls["generate"] = {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "image_data": image_data,
            }
            return {"text": "  sglang text  "}

        def shutdown(self):
            threads.append(threading.get_ident())
            calls["stopped"] = True

    module = ModuleType("sglang")
    module.Engine = Engine
    monkeypatch.setitem(sys.modules, "sglang", module)
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *a, **kw: tokenizer
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    model = SGLangMiniCPMOCRModel("minicpm-test", "/models/minicpm", cpu_offload=True)
    try:
        assert (
            model.ocr(Image.new("RGBA", (4, 4)), max_tokens=16, request_id="request")
            == "sglang text"
        )
        assert calls["config"] == {
            "model_path": "/models/minicpm",
            "context_length": 40960,
            "trust_remote_code": False,
        }
        request = calls["generate"]
        assert request["sampling_params"] == {"max_new_tokens": 16, "temperature": 0.0}
        image = Image.open(
            io.BytesIO(base64.b64decode(request["image_data"].split(",", 1)[1]))
        )
        assert image.mode == "RGB"
        assert tokenizer.template_kwargs["enable_thinking"] is False
    finally:
        model.stop()
    assert calls["stopped"]
    assert len(set(threads)) == 1
    assert model._executor is None
    assert model._model is None


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_failed_load_releases_executor(monkeypatch, engine):
    from .. import vllm as adapter

    def fail(*args, **kwargs):
        raise RuntimeError("failed to load OCR engine")

    if engine == "vllm":
        monkeypatch.setattr(adapter, "_load_vllm_model", fail)
    else:
        module = ModuleType("sglang")
        module.Engine = fail
        monkeypatch.setitem(sys.modules, "sglang", module)
        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: FakeTokenizer()
        )
        monkeypatch.setitem(sys.modules, "transformers", transformers)
    model = ENGINE_CLASSES[engine]("minicpm-test", "/models/minicpm")
    with pytest.raises(RuntimeError, match="failed to load OCR engine"):
        model.load()
    assert model._model is None
    assert model._tokenizer is None
    assert model._executor is None
    model.stop()


@pytest.mark.parametrize("use_directory", [False, True])
def test_llama_cpp_load_and_request(monkeypatch, tmp_path, use_directory):
    model_path = tmp_path / "MiniCPM-OCR-1_0-Q8_0.gguf"
    projector = tmp_path / "mmproj-model-f16-4x.gguf"
    model_path.write_bytes(b"gguf")
    projector.write_bytes(b"gguf")
    calls = {}

    class Params:
        def __init__(self):
            self.mmproj = SimpleNamespace(path=None)

    class Server:
        def __init__(self, params):
            calls["params"] = params

        def handle_chat_completions(self, request, callback):
            calls["request"] = request
            callback({"choices": [{"message": {"content": "  llama text  "}}]})

    module = ModuleType("xllamacpp")
    module.CommonParams = Params
    module.Server = Server
    monkeypatch.setitem(sys.modules, "xllamacpp", module)
    spec = next(f for f in _families() if getattr(f, "quantization", None) == "Q8_0")
    model = LlamaCppMiniCPMOCRModel(
        "minicpm-test",
        str(tmp_path if use_directory else model_path),
        model_spec=spec,
    )
    assert (
        model.ocr(
            Image.new("RGBA", (4, 4)), prompt="Formula Recognition:", max_tokens=16
        )
        == "llama text"
    )
    assert calls["params"].model == str(model_path)
    assert calls["params"].mmproj.path == str(projector)
    assert calls["params"].use_jinja is True
    assert calls["request"]["max_tokens"] == 16
    assert calls["request"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert (
        calls["request"]["messages"][0]["content"][1]["text"] == "Formula Recognition:"
    )
    model.stop()
    assert model._model is None


def test_llama_cpp_reports_errors(monkeypatch):
    model = LlamaCppMiniCPMOCRModel("minicpm-test")
    model._model = SimpleNamespace(
        handle_chat_completions=lambda request, callback: callback(
            {"error": "failed OCR"}
        )
    )
    with pytest.raises(RuntimeError, match="failed OCR"):
        model.ocr(Image.new("RGB", (4, 4)))
