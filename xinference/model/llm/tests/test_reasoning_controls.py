# CI: linux
# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from .. import BUILTIN_LLM_FAMILIES
from ..core import LLM
from ..reasoning_controls import ReasoningEffortConfig, normalize_reasoning_controls
from ..utils import ChatModelMixin


def family(name):
    return next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == name)


@pytest.mark.parametrize(
    "value,expected",
    [(1, 1), (100, 100), ("none", "none"), ("low", 50), ("high", 75), ("max", 100)],
)
def test_deepseek_native_effort(value, expected):
    capability = family("DeepSeek-V4.1-Flash-1017").get_reasoning_effort()
    config = normalize_reasoning_controls({"reasoning_effort": value}, capability)
    assert config == {"chat_template_kwargs": {"reasoning_effort": expected}}


@pytest.mark.parametrize(
    "value", [True, False, 0, 101, 1.5, "75", "medium", [], {}, None]
)
def test_invalid_deepseek_effort(value):
    with pytest.raises(ValueError, match="Invalid reasoning_effort"):
        normalize_reasoning_controls(
            {"reasoning_effort": value},
            family("DeepSeek-V4.1-Flash-1017").get_reasoning_effort(),
        )


@pytest.mark.parametrize("value", ["low", "high", "max"])
def test_glm_effort_reaches_official_template(value):
    model = family("glm-5.3")
    config = normalize_reasoning_controls(
        {"reasoning_effort": value}, model.get_reasoning_effort()
    )
    prompt = ChatModelMixin()._build_from_raw_template(
        [{"role": "user", "content": "hello"}],
        model.chat_template,
        **config["chat_template_kwargs"],
    )
    assert f"Reasoning Effort: {value.capitalize()}" in prompt


def test_no_silent_glm_medium_fallback():
    with pytest.raises(ValueError, match="low, high, max"):
        normalize_reasoning_controls(
            {"reasoning_effort": "medium"}, family("glm-5.3").get_reasoning_effort()
        )


def test_equivalent_aliases_and_independent_budget_do_not_mutate_request():
    request = {
        "reasoning_effort": "high",
        "max_tokens": 4096,
        "thinking_budget": 1024,
        "chat_template_kwargs": '{"reasoning_effort": 75, "custom": true}',
    }
    original = copy.deepcopy(request)
    config = normalize_reasoning_controls(
        request, family("DeepSeek-V4.1-Flash-1017").get_reasoning_effort()
    )
    assert request == original
    assert config["chat_template_kwargs"] == {
        "reasoning_effort": 75,
        "custom": True,
        "thinking_budget": 1024,
        "enable_thinking": True,
    }


@pytest.mark.parametrize(
    "config",
    [
        {
            "reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": "high"},
        },
        {"thinking_budget": 10, "chat_template_kwargs": {"thinking_budget": 20}},
        {"thinking_budget": True},
        {"thinking_budget": 0},
        {"thinking_budget": "10"},
        {"thinking_budget": 10, "max_tokens": 10},
        {"thinking_budget": 10, "chat_template_kwargs": {"enable_thinking": False}},
    ],
)
def test_conflicts_and_invalid_budgets(config):
    with pytest.raises(ValueError):
        normalize_reasoning_controls(config, family("glm-5.3").get_reasoning_effort())


def test_unknown_capability_is_not_inferred_from_reasoning_ability():
    assert family("qwen3").get_reasoning_effort() is None
    with pytest.raises(NotImplementedError, match="declared native"):
        normalize_reasoning_controls({"reasoning_effort": "high"}, None)


def test_custom_family_inherits_or_declares_effort():
    custom = family("glm-5.3").copy(deep=True)
    custom.model_family = "glm-5.3"
    custom.model_name = "local-glm"
    custom.reasoning_effort = None
    assert custom.to_description()["reasoning_effort"]["values"] == [
        "low",
        "high",
        "max",
    ]
    custom.reasoning_effort = ReasoningEffortConfig(values=["medium"])
    assert custom.get_reasoning_effort().values == ["medium"]


def test_budget_preflight_checks_backend_and_options():
    from ..mlx.core import MLXChatModel
    from ..reasoning_parser import ReasoningParser

    model = object.__new__(MLXChatModel)
    model.model_family = family("glm-5.3")
    model.reasoning_parser = ReasoningParser(
        reasoning_start_tag="<think>", reasoning_end_tag="</think>"
    )
    model._tokenizer = SimpleNamespace(
        encode=lambda text, **kw: list(text.encode()), unk_token_id=None, eos_token_id=0
    )
    config = model.prepare_reasoning_request({"thinking_budget": 10, "max_tokens": 30})
    assert config["_mlx_thinking_budget"] == 10
    assert config["chat_template_kwargs"]["enable_thinking"] is True
    with pytest.raises(NotImplementedError, match="structured output"):
        model.prepare_reasoning_request(
            {"thinking_budget": 10, "response_format": {"type": "json_object"}}
        )
    model.reasoning_parser = None
    with pytest.raises(NotImplementedError, match="reasoning markers"):
        model.prepare_reasoning_request({"thinking_budget": 10})


@pytest.mark.parametrize(
    "constraint",
    [
        {"guided_json": {"type": "object"}},
        {"guided_json": {}},
        {"guided_regex": "[a-z]+"},
        {"guided_choice": ["yes", "no"]},
        {"guided_grammar": 'root ::= "yes"'},
        {"guided_json_object": True},
    ],
)
def test_budget_rejects_guided_constraints_before_decoding(constraint):
    model = SimpleNamespace(
        _reasoning_effort_capability=lambda: None,
        reasoning_parser=SimpleNamespace(reasoning_end_tag="</think>"),
        _apply_thinking_budget=Mock(),
    )
    with pytest.raises(NotImplementedError, match="structured output"):
        LLM.prepare_reasoning_request(
            model, {"thinking_budget": 10, "max_tokens": 30, **constraint}
        )
    model._apply_thinking_budget.assert_not_called()
    assert LLM.prepare_reasoning_request(model, constraint) == constraint


def test_budget_accepts_inactive_guided_options():
    model = SimpleNamespace(
        _reasoning_effort_capability=lambda: None,
        reasoning_parser=SimpleNamespace(reasoning_end_tag="</think>"),
        _apply_thinking_budget=Mock(),
    )
    config = LLM.prepare_reasoning_request(
        model,
        {
            "thinking_budget": 10,
            "max_tokens": 30,
            "guided_json": None,
            "guided_json_object": False,
            "guided_regex": None,
            "guided_choice": [],
            "guided_grammar": "",
            "guided_decoding_backend": "xgrammar",
            "guided_whitespace_pattern": "[ ]*",
        },
    )
    model._apply_thinking_budget.assert_called_once_with(config)


def test_anthropic_effort_is_native_and_preserves_system_prompt():
    from ..transformers.core import PytorchChatModel

    model = object.__new__(PytorchChatModel)
    model.model_family = family("glm-5.3")
    model.model_uid = "glm"
    model.reasoning_parser = None
    model.count_tokens = lambda *args: 10
    messages = [
        {"role": "system", "content": "Keep this unchanged"},
        {"role": "user", "content": "hi"},
    ]
    prepared = LLM.prepare_anthropic_request(
        model,
        messages,
        {
            "reasoning_effort": "high",
            "_anthropic_requested_features": ["reasoning_effort"],
        },
    )
    assert prepared["messages"] == messages
    assert (
        prepared["generate_config"]["chat_template_kwargs"]["reasoning_effort"]
        == "high"
    )
    assert model.get_anthropic_capabilities().reasoning_effort == "native"


@pytest.mark.parametrize(
    "engine,native",
    [
        ("vllm", "thinking_token_budget"),
        ("sglang", "max_thinking_tokens"),
        ("mlx", "_mlx_thinking_budget"),
    ],
)
def test_budget_is_only_enforced_on_pd_decode(engine, native):
    from ..mlx.core import MLXChatModel
    from ..sglang.core import SGLANGChatModel
    from ..vllm.core import VLLMChatModel

    model = object.__new__(
        {"vllm": VLLMChatModel, "sglang": SGLANGChatModel, "mlx": MLXChatModel}[engine]
    )
    config = {
        "max_tokens": 1,
        native: 1024,
        "_pd_kv_transfer_params": {"do_remote_decode": True},
        "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 1024},
    }
    model._apply_thinking_budget(config)
    assert native not in config
    assert config["chat_template_kwargs"]["thinking_budget"] == 1024


@pytest.mark.parametrize(
    "capability",
    [
        {"values": []},
        {"values": ["high", "high"]},
        {"values": ["high"], "integer_range": [100, 1]},
        {"values": ["high"], "aliases": {"high": 50}},
        {"values": ["high"], "integer_range": [1, 100], "aliases": {"medium": 50}},
        {"values": ["high"], "integer_range": [1, 100], "aliases": {"high": 101}},
    ],
)
def test_invalid_capability_declarations(capability):
    with pytest.raises(ValueError):
        ReasoningEffortConfig.parse_obj(capability)


def test_effort_capabilities_are_version_specific():
    old = family("DeepSeek-V4-Flash").get_reasoning_effort()
    updated = family("DeepSeek-V4-Flash-0731").get_reasoning_effort()
    assert old.values == ["high", "max"]
    assert updated.values == ["low", "high", "max"]
    with pytest.raises(ValueError):
        old.normalize("low", "reasoning_effort")
    assert updated.normalize("low", "reasoning_effort") == "low"


def test_runtime_budget_capability_checks_mlx_without_generating():
    from ..mlx.core import MLXChatModel
    from ..reasoning_parser import ReasoningParser

    model = object.__new__(MLXChatModel)
    model.model_family = family("glm-5.3")
    model.reasoning_parser = ReasoningParser(
        reasoning_start_tag="<think>", reasoning_end_tag="</think>"
    )
    model._tokenizer = SimpleNamespace(
        encode=lambda text, **kw: list(text.encode()), eos_token_id=0
    )
    assert model.get_reasoning_capabilities() == {"supports_thinking_budget": True}
    model._xavier = object()
    assert model.get_reasoning_capabilities() == {"supports_thinking_budget": False}
    model._xavier = None
    model.reasoning_parser = None
    assert model.get_reasoning_capabilities() == {"supports_thinking_budget": False}
