# CI: linux

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from ..... import constants
from ...cache_manager import ImageCacheManager
from ...core import create_image_model_instance
from ..lighton_ocr3 import LightOnOCR3Model
from ..sglang import SGLangLightOnOCR3Model
from ..vllm import VLLMLightOnOCR3Model

MODEL_NAMES = ("LightOnOCR-3-0.8B", "LightOnOCR-3-1B", "LightOnOCR-3-4B")


@pytest.fixture
def transformers_runtime(monkeypatch):
    runtime = ModuleType("transformers")
    runtime.__version__ = "5.5.4"
    for name in (
        "AutoProcessor",
        "LightOnOcrForConditionalGeneration",
        "LightOnOcrProcessor",
        "Qwen3_5ForConditionalGeneration",
    ):
        setattr(runtime, name, Mock(name=name))
    monkeypatch.setitem(sys.modules, "transformers", runtime)
    return runtime


@pytest.mark.parametrize(
    "hub, revision", [("huggingface", "main"), ("modelscope", "master")]
)
@pytest.mark.parametrize(
    "model_name, engine, model_class",
    [
        *((name, None, LightOnOCR3Model) for name in MODEL_NAMES),
        *((name, "vllm", VLLMLightOnOCR3Model) for name in MODEL_NAMES),
        ("LightOnOCR-3-0.8B", "sglang", SGLangLightOnOCR3Model),
        ("LightOnOCR-3-4B", "sglang", SGLangLightOnOCR3Model),
    ],
)
def test_lightonocr3_builtin_factory(
    monkeypatch, transformers_runtime, model_name, hub, revision, engine, model_class
):
    if model_class is not LightOnOCR3Model:
        monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))

    def cache(cache_manager):
        assert cache_manager._model_family.model_hub == hub
        return "/models/lightonocr"

    monkeypatch.setattr(ImageCacheManager, "cache", cache)
    model = create_image_model_instance(
        "lightonocr3-test",
        model_name,
        download_hub=hub,
        model_engine=engine,
        enable_virtual_env=False,
    )
    assert type(model) is model_class
    assert model.model_ability == ["ocr"]
    assert model.model_family.model_id == f"lightonai/{model_name}"
    assert model.model_family.model_hub == hub
    assert model.model_family.model_revision == revision
    assert model.model_family.model_format == "pytorch"
    assert model.model_family.model_engine == (engine or "transformers")
    assert model._model_path == "/models/lightonocr"


@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_transformers_incompatible_host_rejected_before_cache(
    monkeypatch, hub, model_name
):
    runtime = ModuleType("transformers")
    runtime.__version__ = "4.53.3"
    runtime.AutoProcessor = Mock()
    monkeypatch.setitem(sys.modules, "transformers", runtime)
    # Exercise the explicit request even when the global default enables venvs.
    monkeypatch.setattr(constants, "XINFERENCE_ENABLE_VIRTUAL_ENV", True)
    cache = Mock(return_value="/models/lightonocr")
    monkeypatch.setattr(ImageCacheManager, "cache", cache)

    with pytest.raises(ImportError, match=r"transformers>=5\.5\.4") as exc:
        create_image_model_instance(
            "lightonocr3-test",
            model_name,
            download_hub=hub,
            enable_virtual_env=False,
        )

    assert model_name in str(exc.value)
    assert (
        "LightOnOcrForConditionalGeneration"
        if model_name == "LightOnOCR-3-1B"
        else "Qwen3_5ForConditionalGeneration"
    ) in str(exc.value)
    cache.assert_not_called()


def test_transformers_host_below_required_version_rejected_with_available_classes(
    monkeypatch, transformers_runtime
):
    transformers_runtime.__version__ = "5.5.3"
    cache = Mock()
    monkeypatch.setattr(ImageCacheManager, "cache", cache)
    with pytest.raises(ImportError, match=r"Found transformers==5\.5\.3"):
        create_image_model_instance(
            "lightonocr3-test", "LightOnOCR-3-1B", enable_virtual_env=False
        )
    cache.assert_not_called()


@pytest.mark.parametrize("missing_class", ["LightOnOcrProcessor", "AutoProcessor"])
def test_transformers_missing_processor_rejected_before_cache(
    monkeypatch, transformers_runtime, missing_class
):
    delattr(transformers_runtime, missing_class)
    cache = Mock()
    monkeypatch.setattr(ImageCacheManager, "cache", cache)
    model_name = (
        "LightOnOCR-3-1B"
        if missing_class == "LightOnOcrProcessor"
        else "LightOnOCR-3-4B"
    )
    with pytest.raises(ImportError, match=missing_class):
        create_image_model_instance(
            "lightonocr3-test", model_name, enable_virtual_env=False
        )
    cache.assert_not_called()


@pytest.mark.parametrize(
    "enable_virtual_env, skip_preflight", [(True, False), (False, True)]
)
def test_transformers_cache_without_checking_host(
    monkeypatch, enable_virtual_env, skip_preflight
):
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    monkeypatch.setattr(constants, "XINFERENCE_ENABLE_VIRTUAL_ENV", False)
    cache = Mock(return_value="/models/lightonocr")
    monkeypatch.setattr(ImageCacheManager, "cache", cache)

    model = create_image_model_instance(
        "lightonocr3-test",
        "LightOnOCR-3-1B",
        enable_virtual_env=enable_virtual_env,
        _xinference_skip_runtime_preflight=skip_preflight,
    )

    assert model._model_path == "/models/lightonocr"
    cache.assert_called_once_with()


def test_transformers_preflight_uses_selected_python_executable(monkeypatch):
    from .. import lighton_ocr3

    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    run = Mock(return_value=SimpleNamespace(stdout="5.5.4\n"))
    monkeypatch.setattr(lighton_ocr3.subprocess, "run", run)
    cache = Mock(return_value="/models/lightonocr")
    monkeypatch.setattr(ImageCacheManager, "cache", cache)

    model = create_image_model_instance(
        "lightonocr3-test",
        "LightOnOCR-3-1B",
        enable_virtual_env=False,
        _xinference_model_python_executable="/model-venv/bin/python",
    )

    command = run.call_args.args[0]
    assert command[0] == "/model-venv/bin/python"
    assert command[-2:] == ["LightOnOcrForConditionalGeneration", "LightOnOcrProcessor"]
    assert run.call_args.kwargs["env"]["PYTHONPATH"] == ""
    cache.assert_called_once_with()
    assert "_xinference_model_python_executable" not in model._kwargs


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (None, []),
        ("", []),
        ("OCR", []),
        ("<image>\nFree OCR. Extract all text content from the image.", []),
        ("<image>\nConvert this document to clean markdown format.", []),
        ("<image>\n<|grounding|>Convert the document to markdown.", ["grounding"]),
        ("grounding", ["grounding"]),
    ],
)
def test_lightonocr3_chat_template_content(prompt, expected):
    messages = LightOnOCR3Model._messages(prompt)
    assert messages == [
        {
            "role": "user",
            "content": [{"type": "image"}]
            + [{"type": "text", "text": text} for text in expected],
        }
    ]


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_lightonocr3_transformers_generation(monkeypatch, model_name):
    is_1b = model_name == "LightOnOCR-3-1B"
    processor = Mock()
    inputs = SimpleNamespace(
        to=Mock(return_value={"input_ids": torch.tensor([[1, 2]])})
    )
    processor.apply_chat_template.return_value = inputs
    processor.decode.return_value = "  ![title](1,2,3,4)\n# Document  "
    checkpoint = Mock(device=torch.device("cpu"), dtype=torch.float32)
    checkpoint.to.return_value = checkpoint
    checkpoint.eval.return_value = checkpoint
    checkpoint.generate.return_value = torch.tensor([[1, 2, 3, 4]])

    model_loader = Mock()
    model_loader.from_pretrained.return_value = checkpoint
    processor_loader = Mock()
    processor_loader.from_pretrained.return_value = processor
    transformers = ModuleType("transformers")
    if is_1b:
        transformers.LightOnOcrForConditionalGeneration = model_loader
        transformers.LightOnOcrProcessor = processor_loader
    else:
        transformers.Qwen3_5ForConditionalGeneration = model_loader
        transformers.AutoProcessor = processor_loader
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    spec = SimpleNamespace(
        model_name=model_name, model_ability=["ocr"], is_builtin=True
    )
    model = LightOnOCR3Model(
        "test", "/models/lightonocr", device="cpu", model_spec=spec
    )
    image = Image.new("RGBA", (1024, 4096))
    result = model.ocr(image, prompt="grounding", max_tokens=32, eval_mode=True)

    assert result == "![title](1,2,3,4)\n# Document"
    content = processor.apply_chat_template.call_args.args[0][0]["content"]
    longest_edge = 1540 if is_1b else 2048
    assert content[0]["image"].mode == "RGB"
    assert content[0]["image"].size == (longest_edge // 4, longest_edge)
    assert image.size == (1024, 4096)
    assert content[1] == {"type": "text", "text": "grounding"}
    template_kwargs = processor.apply_chat_template.call_args.kwargs
    if is_1b:
        assert "enable_thinking" not in template_kwargs
        assert model_loader.from_pretrained.call_args.kwargs["key_mapping"] == {
            r"^language_model\.model\.": "model.language_model."
        }
    else:
        assert template_kwargs["enable_thinking"] is False
        assert "key_mapping" not in model_loader.from_pretrained.call_args.kwargs
    inputs.to.assert_called_once_with(device=torch.device("cpu"), dtype=torch.float32)
    generation_kwargs = checkpoint.generate.call_args.kwargs
    assert generation_kwargs["max_new_tokens"] == 32
    assert generation_kwargs["temperature"] == 0.1
    assert "max_tokens" not in generation_kwargs
    assert "eval_mode" not in generation_kwargs
    assert processor.decode.call_args.args[0].tolist() == [3, 4]
