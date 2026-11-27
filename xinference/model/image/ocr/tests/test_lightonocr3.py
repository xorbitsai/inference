# CI: linux

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from ...core import create_image_model_instance
from ..lighton_ocr3 import LightOnOCR3Model
from ..sglang import SGLangLightOnOCR3Model
from ..vllm import VLLMLightOnOCR3Model

MODEL_NAMES = ("LightOnOCR-3-0.8B", "LightOnOCR-3-1B", "LightOnOCR-3-4B")


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
def test_lightonocr3_builtin_factory(model_name, hub, revision, engine, model_class):
    model = create_image_model_instance(
        "lightonocr3-test",
        model_name,
        download_hub=hub,
        model_engine=engine,
        model_path="/models/lightonocr",
        enable_virtual_env=False,
    )
    assert type(model) is model_class
    assert model.model_ability == ["ocr"]
    assert model.model_family.model_id == f"lightonai/{model_name}"
    assert model.model_family.model_hub == hub
    assert model.model_family.model_revision == revision
    assert model.model_family.model_format == "pytorch"
    assert model.model_family.model_engine == (engine or "transformers")


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
