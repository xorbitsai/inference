# CI: linux

import pickle
import sys
import threading
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from .. import vllm as adapter
from ..vllm import VLLMUnlimitedOCR2Model


@pytest.fixture
def fake_vllm(monkeypatch):
    module = ModuleType("vllm")
    module.__spec__ = ModuleSpec("vllm", None)
    module.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", module)
    calls = {}

    class Engine:
        def generate(self, inputs, sampling_params):
            calls.update(
                inputs=inputs,
                sampling=sampling_params,
                generate_thread=threading.get_ident(),
            )
            return [
                SimpleNamespace(
                    outputs=[SimpleNamespace(text="  <PAGE><BLOCK>Invoice 12345  ")]
                )
            ]

        def shutdown(self):
            calls["shutdown_thread"] = threading.get_ident()

    def load(path, config):
        calls.update(path=path, config=config, load_thread=threading.get_ident())
        return Engine()

    monkeypatch.setattr(adapter, "_load_vllm_model", load)
    return calls


def test_vllm_unlimited_ocr2_load_and_stop(fake_vllm):
    model = VLLMUnlimitedOCR2Model(
        "ocr-test",
        "/models/unlimited",
        device="cuda",
        model_spec=SimpleNamespace(model_ability=["ocr"], is_builtin=True),
        cpu_offload=2,
        device_map="auto",
        torch_dtype="bfloat16",
        gpu_memory_utilization=0.5,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 2},
    )
    try:
        model.load()
        config = fake_vllm["config"]
        assert fake_vllm["path"] == "/models/unlimited"
        assert config["trust_remote_code"] is True
        assert config["cpu_offload_gb"] == 2
        assert config["gpu_memory_utilization"] == 0.5
        assert config["max_model_len"] == 8192
        assert config["max_num_batched_tokens"] == 2048
        assert config["enable_prefix_caching"] is False
        assert config["limit_mm_per_prompt"] == {"image": 2}
        assert config["logits_processors"] == [model.PAGE_GUARD]
        assert not {"cpu_offload", "device_map", "torch_dtype"} & config.keys()
        assert fake_vllm["load_thread"] != threading.get_ident()
        model.ocr(Image.new("RGB", (16, 16)))
    finally:
        model.stop()
    assert (
        fake_vllm["load_thread"]
        == fake_vllm["generate_thread"]
        == fake_vllm["shutdown_thread"]
    )
    assert model._model is None
    assert model._executor is None
    model.stop()


@pytest.mark.parametrize("already_registered", [False, True])
def test_vllm_unlimited_ocr2_keeps_custom_processors(fake_vllm, already_registered):
    processors = ["custom:Processor"]
    if already_registered:
        processors.append(VLLMUnlimitedOCR2Model.PAGE_GUARD)
    original = processors.copy()
    model = VLLMUnlimitedOCR2Model(
        "ocr-test", device="cuda", logits_processors=processors, trust_remote_code=False
    )
    try:
        model.load()
        assert fake_vllm["config"]["logits_processors"] == [
            "custom:Processor",
            model.PAGE_GUARD,
        ]
        assert fake_vllm["config"]["trust_remote_code"] is False
        assert processors == original
    finally:
        model.stop()


def test_vllm_unlimited_ocr2_load_failure_releases_thread(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Cannot load weights")

    monkeypatch.setattr(adapter, "_load_vllm_model", fail)
    model = VLLMUnlimitedOCR2Model("ocr-test", device="cuda")
    with pytest.raises(RuntimeError, match="Cannot load weights"):
        model.load()
    assert model._model is None
    assert model._executor is None


def test_vllm_unlimited_ocr2_requires_cuda():
    model = VLLMUnlimitedOCR2Model("ocr-test", device="cpu")
    with pytest.raises(ValueError, match="requires a CUDA device"):
        model.load()
    assert model._executor is None


def test_vllm_unlimited_ocr2_is_picklable_before_loading():
    model = VLLMUnlimitedOCR2Model("ocr-test", "/models/unlimited", device="cuda")
    restored = pickle.loads(pickle.dumps(model))
    assert restored._executor is None
    assert restored._model_path == model._model_path


@pytest.mark.parametrize("page_count", [None, 1, 2])
def test_vllm_unlimited_ocr2_document_input(fake_vllm, page_count):
    model = VLLMUnlimitedOCR2Model("ocr-test", device="cuda")
    rgba = Image.new("RGBA", (16, 8))
    images = rgba if page_count is None else [rgba] * page_count
    count = page_count or 1
    try:
        result = model.ocr(
            images,
            prompt="<image>\nFree OCR.",
            max_new_tokens=64,
            temperature=0.4,
            top_p=0.8,
            repetition_penalty=1.1,
            max_page_tokens=120,
            request_id="request",
            test_compress=False,
            eval_mode=True,
            return_dict=True,
        )
        assert result == "<PAGE><BLOCK>Invoice 12345"
        assert len(fake_vllm["inputs"]) == 1
        request = fake_vllm["inputs"][0]
        assert request["prompt"] == "<PAGE><image>" * count + "Multi page parsing."
        pages = request["multi_modal_data"]["image"]
        assert [page.mode for page in pages] == ["RGB"] * count
        assert [page.size for page in pages] == [(16, 8)] * count
        assert rgba.mode == "RGBA"
        assert request["mm_processor_kwargs"] == {"crop_mode": page_count is None}
        assert vars(fake_vllm["sampling"]) == {
            "max_tokens": 64,
            "temperature": 0.4,
            "skip_special_tokens": False,
            "extra_args": {"max_page_tokens": 120},
            "top_p": 0.8,
            "repetition_penalty": 1.1,
        }
        model.ocr(images, prompt="Read the invoice.", crop_mode=True)
        assert fake_vllm["sampling"].max_tokens == 8192 * count
        assert fake_vllm["inputs"][0]["mm_processor_kwargs"] == {"crop_mode": True}
        assert fake_vllm["inputs"][0]["prompt"].endswith("Read the invoice.")
    finally:
        model.stop()


@pytest.mark.parametrize("existing_file", [False, True])
def test_vllm_unlimited_ocr2_leaves_saving_to_client(
    fake_vllm, tmp_path, existing_file
):
    save_dir = tmp_path / "caller-output"
    result_path = save_dir / "result.md"
    if existing_file:
        save_dir.mkdir()
        result_path.write_text("existing content", encoding="utf-8")

    model = VLLMUnlimitedOCR2Model("ocr-test", device="cuda")
    try:
        result = model.ocr(
            Image.new("RGB", (16, 16)), save_results=True, save_dir=str(save_dir)
        )
        assert result == "<PAGE><BLOCK>Invoice 12345"
        if existing_file:
            assert result_path.read_text(encoding="utf-8") == "existing content"
        else:
            assert not save_dir.exists()
    finally:
        model.stop()


def test_vllm_unlimited_ocr2_explicit_page_prompt(fake_vllm):
    model = VLLMUnlimitedOCR2Model("ocr-test", device="cuda")
    prompt = "<PAGE><image>Read the invoice."
    try:
        model.ocr(
            Image.new("RGB", (16, 16)),
            prompt=prompt,
            model_size="base",
            max_tokens=32,
            max_new_tokens=64,
        )
        assert fake_vllm["inputs"][0]["prompt"] == prompt
        assert fake_vllm["inputs"][0]["mm_processor_kwargs"] == {"crop_mode": False}
        assert fake_vllm["sampling"].max_tokens == 32
    finally:
        model.stop()


@pytest.mark.parametrize("image", ["invalid", [], [Image.new("RGB", (8, 8)), None]])
def test_vllm_unlimited_ocr2_invalid_image(monkeypatch, image):
    model = VLLMUnlimitedOCR2Model("ocr-test")
    monkeypatch.setattr(model, "load", lambda: pytest.fail("Unexpected weight loading"))
    with pytest.raises(ValueError, match="Input must be a PIL Image"):
        model.ocr(image)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"model_size": "tiny"}, "model_size must be"),
        ({"prompt": "<PAGE><image><PAGE><image>OCR"}, "one <PAGE><image> per image"),
        ({"prompt": "<PAGE>OCR"}, "one <PAGE><image> per image"),
    ],
)
def test_vllm_unlimited_ocr2_invalid_options(monkeypatch, kwargs, message):
    model = VLLMUnlimitedOCR2Model("ocr-test")
    monkeypatch.setattr(model, "load", lambda: pytest.fail("Unexpected weight loading"))
    with pytest.raises(ValueError, match=message):
        model.ocr(Image.new("RGB", (8, 8)), **kwargs)


@pytest.mark.parametrize("outputs", [[], [SimpleNamespace(outputs=[])]])
def test_vllm_unlimited_ocr2_empty_output(fake_vllm, outputs):
    model = VLLMUnlimitedOCR2Model("ocr-test", device="cuda")
    try:
        model.load()
        model._model.generate = lambda *args: outputs
        assert model.ocr(Image.new("RGB", (16, 16))) == ""
    finally:
        model.stop()
