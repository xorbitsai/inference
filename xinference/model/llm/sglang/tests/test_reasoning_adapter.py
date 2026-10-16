# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import sys
from types import SimpleNamespace

import pytest

from ...reasoning_parser import ReasoningParser
from .. import reasoning


@pytest.fixture
def native(monkeypatch, tmp_path):
    class Base:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    builtin = object()
    parser = SimpleNamespace(DetectorMap={"builtin": builtin})
    module = SimpleNamespace(
        ReasoningParser=parser,
        BaseReasoningFormatDetector=Base,
        StreamingParseResult=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.parser.reasoning_parser", module)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.plugins", SimpleNamespace(load_plugins=lambda: None)
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
