import importlib
import importlib.util
import json
import sys
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict

import openai
import pytest
from pydantic import BaseModel

from xinference.client import Client

from ..core import _apply_response_format


class CarType(str, Enum):
    sedan = "sedan"
    suv = "SuV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


def _load_json_from_message(message: Any) -> Dict[str, Any]:
    def _strip_think(text: str) -> str:
        stripped = text.lstrip()
        if stripped.startswith("<think>"):
            if "</think>" in stripped:
                stripped = stripped.split("</think>", 1)[1]
            else:
                stripped = stripped.split("<think>", 1)[1]
        return stripped.lstrip()

    raw_content = message.content
    if isinstance(raw_content, str):
        return json.loads(_strip_think(raw_content))

    if isinstance(raw_content, list):
        text_blocks = []
        for block in raw_content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    text_blocks.append(_strip_think(block["text"]))
                continue

            block_type = getattr(block, "type", None)
            block_text = getattr(block, "text", None)
            if block_type == "text" and block_text:
                text_blocks.append(_strip_think(block_text))

        if text_blocks:
            return json.loads("".join(text_blocks))

    pytest.fail(f"Unexpected message content format: {raw_content!r}")
    raise AssertionError("Unreachable")


def test_apply_response_format_sets_grammar(monkeypatch):
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
    assert "json_schema" not in cfg
    assert cfg["grammar"] == "GRAMMAR"


def test_apply_response_format_handles_conversion_failure(monkeypatch):
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
    cfg = {"response_format": {"type": "json_object"}}
    _apply_response_format(cfg)
    assert "grammar" not in cfg
    assert "json_schema" not in cfg


def test_apply_response_format_uses_real_xllamacpp_if_available():
    if importlib.util.find_spec("xllamacpp") is None:
        pytest.skip("xllamacpp not installed")
    xllamacpp = importlib.import_module("xllamacpp")
    if not hasattr(xllamacpp, "json_schema_to_grammar"):
        pytest.skip("xllamacpp does not expose json_schema_to_grammar")

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
    # Real xllamacpp should prefer grammar to avoid passing both
    assert "json_schema" not in cfg
    assert "grammar" in cfg and cfg["grammar"]


def test_llamacpp_qwen3_json_schema(setup):
    endpoint, _ = setup
    client = Client(endpoint)
    model_uid = client.launch_model(
        model_name="qwen3",
        model_engine="llama.cpp",
        model_size_in_billions="0_6",
        model_format="ggufv2",
        quantization="Q4_K_M",
        n_gpu=None,
    )

    try:
        api_client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
        completion = api_client.chat.completions.create(
            model=model_uid,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Generate a JSON containing the brand, model, and car_type of"
                        " an iconic 90s car."
                    ),
                }
            ],
            temperature=0,
            max_tokens=128,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "car-description",
                    "schema": CarDescription.model_json_schema(),
                },
            },
        )

        parsed = _load_json_from_message(completion.choices[0].message)
        car_description = CarDescription.model_validate(parsed)
        assert car_description.brand
        assert car_description.model
    finally:
        if model_uid is not None:
            client.terminate_model(model_uid)
