# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ...reasoning_parser import KimiK3ReasoningParser, ReasoningParser
from ..reasoning import configure_structured_reasoning, create_reasoning_plugin


class FakeVLLMReasoningParser:
    def __init__(self, tokenizer, *args, **kwargs):
        self.model_tokenizer = tokenizer


@pytest.fixture
def vllm_reasoning(monkeypatch):
    registered = {}
    manager = SimpleNamespace(
        register_module=lambda name, module: registered.setdefault(name, module)
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        SimpleNamespace(
            StructuredOutputsConfig=SimpleNamespace(
                __dataclass_fields__={"reasoning_parser_plugin": object()}
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.reasoning",
        SimpleNamespace(
            ReasoningParser=FakeVLLMReasoningParser, ReasoningParserManager=manager
        ),
    )
    delta = SimpleNamespace(model_fields={"reasoning": object()})
    delta = type(
        "DeltaMessage",
        (),
        {
            "model_fields": delta.model_fields,
            "__init__": lambda self, **kw: self.__dict__.update(kw),
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.openai.engine.protocol",
        SimpleNamespace(DeltaMessage=delta),
    )
    return registered


@pytest.mark.parametrize("kind", ["generic", "kimi"])
def test_auto_plugin_reuses_existing_parser_and_is_loadable(
    tmp_path, vllm_reasoning, kind
):
    parser = (
        ReasoningParser(
            reasoning_start_tag="<think>",
            reasoning_end_tag="</think>",
            enable_thinking=False,
            auto_insert_start_tag=False,
        )
        if kind == "generic"
        else KimiK3ReasoningParser()
    )
    config = {"structured_outputs_config": {"backend": "xgrammar"}}
    configure_structured_reasoning(config, parser, str(tmp_path))
    structured = config["structured_outputs_config"]
    assert structured["backend"] == "xgrammar"
    plugin = Path(structured["reasoning_parser_plugin"])
    original = plugin.read_bytes()
    assert create_reasoning_plugin(parser, str(tmp_path)) == (
        structured["reasoning_parser"],
        str(plugin),
    )
    assert plugin.read_bytes() == original
    # No registration in the launcher is needed; import in EngineCore does it.
    assert not vllm_reasoning
    runpy.run_path(str(plugin))
    adapter_type = vllm_reasoning[structured["reasoning_parser"]]
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(map(ord, text)))
    adapter = adapter_type(tokenizer)
    assert type(adapter.parser) is type(parser)
    text = parser.reasoning_start_tag + "reason" + parser.reasoning_end_tag
    assert adapter.is_reasoning_end(tokenizer.encode(text))
    assert adapter.is_reasoning_end_streaming(
        tokenizer.encode(text), iter(tokenizer.encode(parser.reasoning_end_tag))
    )
    assert adapter.extract_content_ids(
        tokenizer.encode(text + "{}")
    ) == tokenizer.encode("{}")
    delegated = Mock(return_value=("reason", "answer"))
    adapter.parser.extract_reasoning_content = delegated
    assert adapter.extract_reasoning("output", object()) == ("reason", "answer")
    delegated.assert_called_once_with("output")
    delegated_stream = Mock(
        return_value={"reasoning_content": "reason", "content": "answer"}
    )
    adapter.parser.extract_reasoning_content_streaming = delegated_stream
    result = adapter.extract_reasoning_streaming("old", "oldnew", "new", [], [], [])
    assert result.reasoning == "reason" and result.content == "answer"
    delegated_stream.assert_called_once_with("old", "oldnew", "new")


@pytest.mark.parametrize(
    "explicit",
    [
        {"reasoning_parser": "custom"},
        {"reasoning_parser": ""},
        {"reasoning_parser_plugin": "/custom.py"},
    ],
)
def test_explicit_engine_settings_are_preserved(tmp_path, vllm_reasoning, explicit):
    parser = ReasoningParser(
        reasoning_start_tag="<think>", reasoning_end_tag="</think>"
    )
    config = {"structured_outputs_config": dict(explicit)}
    configure_structured_reasoning(config, parser, str(tmp_path))
    assert config["structured_outputs_config"] == explicit
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "parser",
    [
        None,
        ReasoningParser(),
        ReasoningParser(
            reasoning_start_tag="analysis", reasoning_end_tag="assistantfinal"
        ),
    ],
)
def test_missing_or_nonliteral_boundaries_do_not_auto_configure(
    tmp_path, vllm_reasoning, parser
):
    config = {}
    configure_structured_reasoning(config, parser, str(tmp_path))
    assert config == {} and not list(tmp_path.iterdir())


def test_old_engine_without_plugin_support_still_launches(
    tmp_path, vllm_reasoning, monkeypatch
):
    monkeypatch.setitem(
        sys.modules, "vllm.config", SimpleNamespace(StructuredOutputsConfig=object)
    )
    config = {}
    parser = ReasoningParser(
        reasoning_start_tag="<think>", reasoning_end_tag="</think>"
    )
    configure_structured_reasoning(config, parser, str(tmp_path))
    assert config == {}


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.asyncio
async def test_cleaned_template_config_keeps_request_thinking_mode(
    monkeypatch, enabled
):
    from ...core import chat_context_var
    from .. import core

    class StopBeforeSampling(Exception):
        pass

    monkeypatch.setitem(
        sys.modules, "vllm.sampling_params", SimpleNamespace(SamplingParams=object)
    )
    model = object.__new__(core.VLLMChatModel)
    model.reasoning_parser = ReasoningParser(enable_thinking=not enabled)
    model._sanitize_generate_config = Mock(side_effect=StopBeforeSampling)
    context = chat_context_var.set({"enable_thinking": enabled})
    try:
        with pytest.raises(StopBeforeSampling):
            await core.VLLMModel.async_generate.__wrapped__(
                model, "prompt", {"max_tokens": 128}
            )
    finally:
        chat_context_var.reset(context)
    config = model._sanitize_generate_config.call_args.args[0]
    assert config.get("skip_special_tokens", True) is not enabled


@pytest.mark.parametrize(
    "explicit",
    [
        {"reasoning_parser": "qwen3"},
        {"reasoning_parser_plugin": "/native.py"},
        {"reasoning_parser": ""},
    ],
)
def test_top_level_engine_override_prevents_automatic_dict_injection(
    tmp_path, vllm_reasoning, explicit
):
    parser = ReasoningParser(
        reasoning_start_tag="<think>", reasoning_end_tag="</think>"
    )
    config = dict(explicit)
    configure_structured_reasoning(config, parser, str(tmp_path))
    assert config == explicit
    assert "structured_outputs_config" not in config
    assert not list(tmp_path.iterdir())


def test_explicit_config_object_is_preserved(tmp_path, vllm_reasoning):
    configured = SimpleNamespace(reasoning_parser="native")
    config = {"structured_outputs_config": configured}
    configure_structured_reasoning(config, ReasoningParser(), str(tmp_path))
    assert config["structured_outputs_config"] is configured
