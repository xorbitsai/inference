import sys
from types import SimpleNamespace


def test_apply_response_format_sets_grammar(monkeypatch):
    from xinference.model.llm.llama_cpp.core import _apply_response_format

    fake_xllamacpp = SimpleNamespace(json_schema_to_grammar=lambda schema: "GRAMMAR")
    monkeypatch.setitem(sys.modules, "xllamacpp", fake_xllamacpp)

    cfg = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                }
            },
        }
    }

    _apply_response_format(cfg)

    assert "response_format" not in cfg
    assert cfg["json_schema"]["required"] == ["a"]
    assert cfg["grammar"] == "GRAMMAR"


def test_apply_response_format_handles_conversion_failure(monkeypatch):
    from xinference.model.llm.llama_cpp.core import _apply_response_format

    def _raise(_):
        raise ValueError("bad schema")

    fake_xllamacpp = SimpleNamespace(json_schema_to_grammar=_raise)
    monkeypatch.setitem(sys.modules, "xllamacpp", fake_xllamacpp)

    cfg = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {"b": {"type": "string"}},
                    "required": ["b"],
                }
            },
        }
    }

    _apply_response_format(cfg)

    assert "response_format" not in cfg
    assert cfg["json_schema"]["required"] == ["b"]
    assert "grammar" not in cfg


def test_apply_response_format_ignores_non_schema(monkeypatch):
    from xinference.model.llm.llama_cpp.core import _apply_response_format

    cfg = {"response_format": {"type": "json_object"}}
    _apply_response_format(cfg)
    assert "grammar" not in cfg
    assert "json_schema" not in cfg


def test_apply_response_format_uses_real_xllamacpp_if_available():
    import importlib.util

    import pytest

    if importlib.util.find_spec("xllamacpp") is None:
        pytest.skip("xllamacpp not installed")

    import importlib

    xllamacpp = importlib.import_module("xllamacpp")
    if not hasattr(xllamacpp, "json_schema_to_grammar"):
        pytest.skip("xllamacpp does not expose json_schema_to_grammar")

    from xinference.model.llm.llama_cpp.core import _apply_response_format

    cfg = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {"c": {"type": "integer"}},
                    "required": ["c"],
                }
            },
        }
    }

    _apply_response_format(cfg)

    assert "response_format" not in cfg
    # Real xllamacpp should attach grammar alongside json_schema
    assert "json_schema" in cfg
    assert "grammar" in cfg and cfg["grammar"]
