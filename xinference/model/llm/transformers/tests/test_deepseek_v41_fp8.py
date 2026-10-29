# CI: linux
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ..core import PytorchChatModel
from ..deepseek_v41 import DeepSeekV41PytorchChatModel


@pytest.mark.parametrize(
    "architecture,model_format,expected",
    [
        ("DeepseekV41ForCausalLM", "fp8", True),
        ("UnrelatedForCausalLM", "fp8", False),
        ("DeepseekV41ForCausalLM", "ggufv2", False),
        ("DeepseekV41ForCausalLM", "pytorch", True),
    ],
)
def test_deepseek_v41_fp8_match(architecture, model_format, expected):
    family = Mock()
    family.architectures = [architecture]
    family.model_ability = ["chat", "vision", "reasoning", "tools"]
    family.has_architecture.side_effect = lambda *names: architecture in names
    family.matches_supported_architectures.return_value = False
    result = DeepSeekV41PytorchChatModel.match_json(
        family,
        SimpleNamespace(model_format=model_format),
        "fp8" if model_format == "fp8" else "none",
    )
    assert (result is True) is expected


def test_unsloth_gguf_registry():
    from ....utils import generate_model_file_names_with_quantization_parts
    from ...llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("DeepSeek-V4.1-Flash-1017")
    specs = [s for s in family.model_specs if s.model_format == "ggufv2"]
    assert specs
    for spec in specs:
        assert spec.model_id == "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
        files, first, merge = generate_model_file_names_with_quantization_parts(spec)
        assert merge
        assert len(files) == int(files[-1].split("-of-")[-1].split(".")[0])
        assert spec.multimodal_projectors == ["mmproj-BF16.gguf", "mmproj-F16.gguf"]


def make_chat_model():
    model = object.__new__(DeepSeekV41PytorchChatModel)
    model.model_uid = "test"
    model.model_family = SimpleNamespace(
        model_name="DeepSeek-V4.1-Flash-1017",
        architectures=["DeepseekV41ForCausalLM"],
        model_family=None,
        chat_template="",
        model_ability=["chat", "vision"],
        stop=[],
        stop_token_ids=[1],
    )
    model.reasoning_parser = None
    model.tool_parser = None
    model._batch_scheduler = None
    model.allow_batch = False
    return model


def test_checkpoint_template_and_text_content():
    model = make_chat_model()
    model._tokenizer = Mock()
    model._tokenizer.get_chat_template.return_value = "checkpoint template"
    model._tokenizer.apply_chat_template.return_value = "user: hello"
    prompt = model._get_full_prompt(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        None,
        {},
    )
    assert prompt == "user: hello"
    args, kwargs = model._tokenizer.apply_chat_template.call_args
    assert args[0][0]["content"] == "hello"
    assert "chat_template" not in kwargs


def test_vision_fails_explicitly():
    model = make_chat_model()
    with pytest.raises(ValueError, match="text inputs only"):
        model._get_full_prompt(
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "x"}}],
                }
            ],
            None,
            {},
        )


@pytest.mark.parametrize("stream", [False, True])
def test_first_chat_uses_direct_generation(stream):
    model = make_chat_model()
    model._tokenizer = Mock()
    model._tokenizer.encode.return_value = [42]
    model.build_streaming_iter = Mock(return_value=(iter(["hello"]), 2))
    import asyncio

    result = asyncio.run(
        model.chat([{"role": "user", "content": "hi"}], {"stream": stream})
    )
    if stream:
        chunks = list(result)
        assert any(
            c["choices"][0].get("delta", {}).get("content") == "hello" for c in chunks
        )
    else:
        assert result["choices"][0]["message"]["content"] == "hello"
    model.build_streaming_iter.assert_called_once()


@pytest.mark.parametrize("version,expected", [("4.57.6", False), ("5.6.0", True)])
def test_transformers_version_guard(monkeypatch, version, expected):
    import importlib.metadata

    monkeypatch.setattr(PytorchChatModel, "check_lib", classmethod(lambda cls: True))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: version)
    assert (DeepSeekV41PytorchChatModel.check_lib() is True) is expected
