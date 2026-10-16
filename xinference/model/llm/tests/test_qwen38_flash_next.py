import json
from pathlib import Path

import pytest
from jinja2 import Environment
from packaging.version import Version


def _family():
    from xinference.model.llm.llm_family import BUILTIN_LLM_FAMILIES

    return next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "qwen3.8-Flash-Next")


def test_flash_next_catalog():
    family = _family()
    assert family.architectures == ["Qwen4ExpForConditionalGeneration"]
    assert family.context_length == 262144
    assert {"vision", "tools", "reasoning", "hybrid"} <= set(family.model_ability)
    assert {s.model_format for s in family.model_specs} == {"pytorch", "fp8"}
    assert all(s.model_size_in_billions == 125 for s in family.model_specs)
    assert all(s.activated_size_in_billions == 6 for s in family.model_specs)
    records = json.loads(
        (Path(__file__).parents[1] / "models/qwen3.8-Flash-Next.json").read_text()
    )
    assert all(
        "Flash-Next" in src["model_id"]
        for s in records[0]["model_specs"]
        for src in s["model_src"].values()
    )


@pytest.mark.parametrize("fmt", ["pytorch", "fp8"])
@pytest.mark.parametrize("virtualenv", [False, True])
@pytest.mark.parametrize("runtime", ["0.29.0", "0.30.0.dev0", "0.30.0rc1", "0.30.0"])
@pytest.mark.parametrize("cached", [False, True])
def test_flash_next_vllm_match(monkeypatch, fmt, virtualenv, runtime, cached):
    from xinference.model.llm.vllm import core

    family = _family()
    spec = next(s for s in family.model_specs if s.model_format == fmt)
    monkeypatch.setattr(core, "VLLM_INSTALLED", not virtualenv)
    monkeypatch.setattr(core, "VLLM_VERSION", Version(runtime))
    monkeypatch.setattr(core, "_virtual_env_allows_missing_vllm", lambda: virtualenv)
    monkeypatch.setattr(
        core.VLLMMultiModel, "_has_cuda_device", classmethod(lambda cls: True)
    )
    monkeypatch.setattr(core.VLLMMultiModel, "_is_linux", classmethod(lambda cls: True))
    # A previously populated global list must not bypass the version gate.
    monkeypatch.setattr(
        core,
        "VLLM_SUPPORTED_MULTI_MODEL_LIST",
        ["Qwen4ExpForConditionalGeneration"] if cached else [],
    )
    result = core.VLLMMultiModel.match_json(family, spec, spec.quantization)
    if virtualenv or Version(runtime) >= Version("0.30.0.dev0"):
        assert result is True
    else:
        assert result[0] is False
        assert "requires vLLM >= 0.30.0.dev0" in result[1]
        assert f"found {runtime}" in result[1]


def test_flash_next_template():
    template = Environment().from_string(_family().chat_template)
    messages = [{"role": "user", "content": "Hello"}]
    thinking = template.render(messages=messages, add_generation_prompt=True)
    direct = template.render(
        messages=messages, add_generation_prompt=True, enable_thinking=False
    )
    assert thinking.endswith("<|im_start|>assistant\n<think>\n")
    assert direct.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "\\n" not in direct


@pytest.mark.parametrize("runtime,expected", [("0.29.0", False), ("0.30.0.dev0", True)])
def test_flash_next_vllm_supported_list(monkeypatch, runtime, expected):
    from xinference.model.llm.vllm import core

    monkeypatch.setattr(core, "_get_effective_vllm_version", lambda: Version(runtime))
    monkeypatch.setattr(core, "VLLM_SUPPORTED_MULTI_MODEL_LIST", [])
    monkeypatch.setattr(core, "VLLM_SUPPORTED_CHAT_MODELS", [])
    monkeypatch.setattr(core, "VLLM_SUPPORTED_MODELS", [])
    core._update_vllm_supported_lists()
    assert (
        "Qwen4ExpForConditionalGeneration" in core.VLLM_SUPPORTED_MULTI_MODEL_LIST
    ) is expected


def test_flash_next_virtualenv_vllm_requirement():
    from xinference.model.llm.vllm import core

    assert core._get_virtualenv_vllm_version(_family()) == Version("0.30.0.dev0")


def test_flash_next_multimodal_and_tool_template():
    template = Environment().from_string(_family().chat_template)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                },
                {"type": "text", "text": "Describe the image"},
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "describe", "arguments": {"detail": "high"}}},
            ],
        },
        {"role": "tool", "content": "A cat"},
    ]
    prompt = template.render(messages=messages, add_generation_prompt=True)
    assert "<|vision_start|><|image_pad|><|vision_end|>Describe the image" in prompt
    assert (
        "<tool_call>\n<function=describe>\n<parameter=detail>\nhigh\n</parameter>"
        in prompt
    )
    assert "<tool_response>\nA cat\n</tool_response>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")


@pytest.mark.parametrize("field", ["model_name", "model_family"])
@pytest.mark.parametrize("effort", ["low", "medium", "xhigh"])
def test_flash_next_api_reasoning_effort(field, effort):
    from xinference.api.restful_api import RESTfulAPI
    from xinference.model.llm.utils import ChatModelMixin

    assert RESTfulAPI._is_qwen38_description({field: _family().model_name})
    body = {"reasoning_effort": effort}
    raw_kwargs = {"reasoning_effort": effort}
    kwargs = dict(raw_kwargs)
    RESTfulAPI._apply_qwen38_reasoning_effort(body, raw_kwargs, kwargs)
    assert "reasoning_effort" not in kwargs
    assert kwargs["chat_template_kwargs"]["reasoning_effort"] == effort
    prompt = ChatModelMixin()._build_from_raw_template(
        [{"role": "user", "content": "Hello"}],
        _family().chat_template,
        **kwargs["chat_template_kwargs"],
    )
    if effort == "medium":
        assert "Reasoning effort is set to" not in prompt
    else:
        assert f"Reasoning effort is set to {effort}." in prompt


@pytest.mark.parametrize(
    "body",
    [
        {"reasoning_effort": "high"},
        {
            "reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": "xhigh"},
        },
    ],
)
def test_flash_next_api_rejects_invalid_reasoning_effort(body):
    from fastapi import HTTPException

    from xinference.api.restful_api import RESTfulAPI

    with pytest.raises(HTTPException) as exc:
        RESTfulAPI._apply_qwen38_reasoning_effort(body, dict(body), dict(body))
    assert exc.value.status_code == 400


def test_flash_next_does_not_offer_generic_transformers():
    from xinference.model.llm.llm_family import check_engine_by_spec_parameters
    from xinference.model.llm.transformers.core import PytorchChatModel

    family = _family()
    spec = next(s for s in family.model_specs if s.model_format == "pytorch")
    result = PytorchChatModel.match_json(family, spec, spec.quantization)
    assert result[0] is False
    assert "dedicated multimodal Transformers adapter" in result[1]
    with pytest.raises(ValueError, match="cannot be run on engine"):
        check_engine_by_spec_parameters(
            "Transformers", family.model_name, "pytorch", 125, "none"
        )
