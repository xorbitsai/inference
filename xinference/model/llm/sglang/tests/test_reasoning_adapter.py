# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import sys
from types import SimpleNamespace

import pytest

from ...reasoning_parser import ReasoningParser
from .. import reasoning


@pytest.fixture
def native(monkeypatch, tmp_path):
    class Base:
        def __init__(self, think_excluded_tokens=None, **kwargs):
            self.__dict__.update(kwargs)
            self.think_excluded_tokens = think_excluded_tokens

    builtin = object()

    class Parser:
        DetectorMap = {"builtin": builtin}

        def __init__(self, model_type=None, stream_reasoning=True, tokenizer=None):
            pass

    parser = Parser
    module = SimpleNamespace(
        ReasoningParser=parser,
        BaseReasoningFormatDetector=Base,
        StreamingParseResult=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.parser.reasoning_parser", module)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.plugins", SimpleNamespace(load_plugins=lambda: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.constrained.reasoner_grammar_backend",
        SimpleNamespace(
            ReasonerGrammarBackend=lambda enable_strict_thinking=False: None
        ),
    )
    monkeypatch.setattr(
        reasoning,
        "entry_points",
        lambda **kwargs: [SimpleNamespace(name=reasoning.PLUGIN_NAME)],
    )
    monkeypatch.setattr(reasoning, "XINFERENCE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [])
    return parser, builtin


def test_adapter_reuses_tags_and_existing_extraction(native):
    parser, builtin = native
    config = {}
    existing = ReasoningParser(
        reasoning_content=False,
        reasoning_start_tag="<custom>",
        reasoning_end_tag="</custom>",
    )
    reasoning.configure_structured_reasoning(config, existing)
    name = config["reasoning_parser"]
    detector = parser.DetectorMap.get(name)()
    assert parser.DetectorMap.get("builtin") is builtin
    assert detector.think_end_token == "</custom>"
    parsed = detector.detect_and_parse('<custom>thought</custom>{"answer":42}')
    assert parsed.reasoning_text == "thought"
    assert parsed.normal_text == '{"answer":42}'
    increments = [
        detector.parse_streaming_increment(part)
        for part in ["<custom>", "thought", "</custom>", "42"]
    ]
    assert "".join(part.reasoning_text or "" for part in increments) == "thought"
    assert "".join(part.normal_text or "" for part in increments) == "42"
    assert detector.parser.reasoning_content is True
    assert existing.reasoning_content is False


def test_spawned_registry_cli_preload_and_integrity(native, monkeypatch):
    parser, builtin = native
    config = {}
    reasoning.configure_structured_reasoning(
        config,
        ReasoningParser(reasoning_start_tag="<think>", reasoning_end_tag="</think>"),
    )
    name = config["reasoning_parser"]
    parser.DetectorMap = {"builtin": builtin}
    monkeypatch.setattr(sys, "argv", ["server", "--reasoning-parser=" + name])
    reasoning.register()
    assert name in parser.DetectorMap
    registry = parser.DetectorMap
    reasoning.register()
    assert parser.DetectorMap is registry
    parser.DetectorMap = {"builtin": builtin}
    reasoning._config_path(name).write_text("{}")
    with pytest.raises(ValueError, match="configuration changed"):
        reasoning.register()


def test_explicit_parser_and_unsupported_registry_preserved(native, monkeypatch):
    config = {"reasoning_parser": "native"}
    reasoning.configure_structured_reasoning(
        config,
        ReasoningParser(reasoning_start_tag="<think>", reasoning_end_tag="</think>"),
    )
    assert config == {"reasoning_parser": "native"}
    config = {}
    monkeypatch.setattr(reasoning, "entry_points", lambda **kwargs: [])
    reasoning.configure_structured_reasoning(
        config,
        ReasoningParser(reasoning_start_tag="<think>", reasoning_end_tag="</think>"),
    )
    assert config == {}


def test_plugin_whitelist_and_immutable_per_model_config(native, monkeypatch):
    parser, _ = native
    monkeypatch.setenv("SGLANG_PLUGINS", "xinference_xavier")
    first, second = {}, {}
    reasoning.configure_structured_reasoning(
        first,
        ReasoningParser(reasoning_start_tag="<think>", reasoning_end_tag="</think>"),
    )
    first_content = reasoning._config_path(first["reasoning_parser"]).read_text()
    reasoning.configure_structured_reasoning(
        second,
        ReasoningParser(reasoning_start_tag="<custom>", reasoning_end_tag="</custom>"),
    )
    assert first["reasoning_parser"] != second["reasoning_parser"]
    assert (
        reasoning._config_path(first["reasoning_parser"]).read_text() == first_content
    )
    import os

    assert os.environ["SGLANG_PLUGINS"] == "xinference_xavier,xinference_reasoning"
    assert (
        parser.DetectorMap.get(second["reasoning_parser"])().think_end_token
        == "</custom>"
    )


def test_harmony_not_registered_as_literal_token_parser(native):
    config = {}
    reasoning.configure_structured_reasoning(
        config,
        ReasoningParser(
            reasoning_start_tag="analysis", reasoning_end_tag="assistantfinal"
        ),
    )
    assert config == {}


@pytest.mark.parametrize("eos_alias", [False, True])
def test_strict_thinking_excludes_stop_tokens_but_preserves_reasoning_end(
    native, eos_alias
):
    parser, _ = native
    config = {}
    stop = ["<stop>", "multi-token stop", "<unknown>", "</custom>"]
    reasoning.configure_structured_reasoning(
        config,
        ReasoningParser(reasoning_start_tag="<custom>", reasoning_end_tag="</custom>"),
        stop,
    )
    tokens = {
        "<custom>": [1],
        "</custom>": [2],
        "<eos>": [2 if eos_alias else 3],
        "<stop>": [4],
        "multi-token stop": [5, 6],
        "<unknown>": [0],
    }
    tokenizer = SimpleNamespace(
        eos_token="<eos>",
        unk_token_id=0,
        encode=lambda text, **kwargs: tokens[text],
    )
    detector = parser.DetectorMap.get(config["reasoning_parser"])(tokenizer=tokenizer)
    assert config["enable_strict_thinking"] is True
    assert detector.think_excluded_tokens == (
        ["<stop>"] if eos_alias else ["<stop>", "<eos>"]
    )
    assert detector.get_think_end_token_ids(tokenizer) == [2]
    assert stop == ["<stop>", "multi-token stop", "<unknown>", "</custom>"]


def test_strict_thinking_respects_explicit_disable(native):
    config = {"enable_strict_thinking": False}
    reasoning.configure_structured_reasoning(
        config,
        ReasoningParser(reasoning_start_tag="<think>", reasoning_end_tag="</think>"),
    )
    assert config["enable_strict_thinking"] is False


@pytest.mark.parametrize("backend", [None, "xgrammar"])
def test_default_grammar_backend_enables_strict_thinking(native, backend):
    config = {"grammar_backend": backend}
    reasoning.configure_structured_reasoning(
        config,
        ReasoningParser(reasoning_start_tag="<think>", reasoning_end_tag="</think>"),
    )
    assert config["enable_strict_thinking"] is True


@pytest.mark.parametrize("backend", ["none", "outlines", "llguidance"])
def test_explicit_other_grammar_backend_is_preserved(native, backend):
    config = {"grammar_backend": backend}
    reasoning.configure_structured_reasoning(
        config,
        ReasoningParser(reasoning_start_tag="<think>", reasoning_end_tag="</think>"),
    )
    assert "enable_strict_thinking" not in config
    assert config["grammar_backend"] == backend


def test_older_reasoning_backend_does_not_receive_new_option(native, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.constrained.reasoner_grammar_backend",
        SimpleNamespace(ReasonerGrammarBackend=lambda: None),
    )
    config = {}
    reasoning.configure_structured_reasoning(
        config,
        ReasoningParser(reasoning_start_tag="<think>", reasoning_end_tag="</think>"),
    )
    assert "reasoning_parser" in config
    assert "enable_strict_thinking" not in config


def test_legacy_parser_without_tokenizer_disables_automatic_strict_thinking(
    native, monkeypatch, caplog
):
    parser, _ = native

    class LegacyParser:
        DetectorMap = parser.DetectorMap

        def __init__(self, model_type=None, stream_reasoning=True):
            pass

    monkeypatch.setattr(
        sys.modules["sglang.srt.parser.reasoning_parser"],
        "ReasoningParser",
        LegacyParser,
    )
    config = {}
    reasoning.configure_structured_reasoning(
        config,
        ReasoningParser(reasoning_start_tag="<think>", reasoning_end_tag="</think>"),
    )
    assert "reasoning_parser" in config
    assert "enable_strict_thinking" not in config
    assert "automatic strict thinking is disabled" in caplog.text


def test_detector_without_tokenizer_has_no_exclusions(native):
    parser, _ = native
    config = {}
    reasoning.configure_structured_reasoning(
        config,
        ReasoningParser(reasoning_start_tag="<think>", reasoning_end_tag="</think>"),
        ["<stop>"],
    )
    detector = parser.DetectorMap.get(config["reasoning_parser"])(tokenizer=None)
    assert detector.think_excluded_tokens == []
    assert detector.detect_and_parse("<think>thought</think>42").normal_text == "42"


def test_missing_reasoning_backend_preserves_parser_registration(native, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "sglang.srt.constrained.reasoner_grammar_backend", None
    )
    config = {}
    reasoning.configure_structured_reasoning(
        config,
        ReasoningParser(reasoning_start_tag="<think>", reasoning_end_tag="</think>"),
    )
    assert "reasoning_parser" in config
    assert "enable_strict_thinking" not in config


def test_model_load_passes_family_stop_to_reasoning_adapter(native, monkeypatch):
    from ..core import SGLANGModel

    class StopBeforeRuntime(Exception):
        pass

    model = object.__new__(SGLANGModel)
    model._model_config = {"reasoning_content": True}
    model.model_family = SimpleNamespace(stop=["<stop>"])
    model.reasoning_parser = ReasoningParser(
        reasoning_start_tag="<think>", reasoning_end_tag="</think>"
    )
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(__version__="0.5.21"))
    monkeypatch.setattr(model, "_sanitize_model_config", lambda config: config)
    monkeypatch.setattr(model, "prepare_parse_reasoning_content", lambda *a, **k: None)
    monkeypatch.setattr(model, "prepare_parse_tool_calls", lambda: None)

    def configure(config, parser, stop=None):
        assert config is model._model_config
        assert parser is model.reasoning_parser
        assert stop is model.model_family.stop
        raise StopBeforeRuntime

    monkeypatch.setattr(reasoning, "configure_structured_reasoning", configure)
    with pytest.raises(StopBeforeRuntime):
        model.load()


def test_older_detector_constructor_is_preserved(native, monkeypatch):
    class LegacyBase:
        def __init__(
            self, think_start_token, think_end_token, force_reasoning, stream_reasoning
        ):
            self.think_end_token = think_end_token

    monkeypatch.setattr(
        sys.modules["sglang.srt.parser.reasoning_parser"],
        "BaseReasoningFormatDetector",
        LegacyBase,
    )
    parser, _ = native
    config = {}
    reasoning.configure_structured_reasoning(
        config,
        ReasoningParser(reasoning_start_tag="<think>", reasoning_end_tag="</think>"),
    )
    detector = parser.DetectorMap.get(config["reasoning_parser"])()
    assert detector.think_end_token == "</think>"
    assert detector.detect_and_parse("<think>thought</think>42").normal_text == "42"
