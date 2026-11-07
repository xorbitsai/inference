# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ..core import LLM
from ..sglang.core import SGLANGChatModel
from ..vllm.core import VLLMChatModel


@pytest.mark.parametrize(
    "value,expected",
    [(4096, 4096), (None, None), (True, None), (0, None), ("4096", None)],
)
def test_runtime_context_does_not_fall_back_to_registry_or_launch(value, expected):
    model = SimpleNamespace(
        _context_length=value,
        model_family=SimpleNamespace(context_length=131072),
        _model_config={"max_model_len": 32768},
    )
    assert LLM.get_runtime_context_length(model) == expected


def test_vllm_resolved_engine_length_changes_after_reload():
    model = VLLMChatModel.__new__(VLLMChatModel)
    model._is_vllm_v1 = lambda: True
    model._engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192))
    model._set_context_length()
    assert model.get_runtime_context_length() == 8192
    model._engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096))
    model._set_context_length()
    assert model.get_runtime_context_length() == 4096


@pytest.fixture
def sglang_runtime(monkeypatch):
    model = SGLANGChatModel.__new__(SGLANGChatModel)
    model._engine = SimpleNamespace(url="http://127.0.0.1:1234")
    model._model_config = {"api_key": "test", "context_length": 131072}
    model.model_family = SimpleNamespace(context_length=262144)
    response = MagicMock()
    response.__enter__.return_value = response
    # ModelList/ModelCard returned by SGLang 0.5.21's /v1/models:
    # sgl-project/sglang@e00930c5489053f26d86b179cee0d087f846acbb
    response.json.return_value = {
        "object": "list",
        "data": [
            {
                "id": "served-model-alias",
                "object": "model",
                "created": 0,
                "owned_by": "sglang",
                "root": "served-model-alias",
                "parent": None,
                "max_model_len": 8192,
            }
        ],
    }
    get = MagicMock(return_value=response)
    monkeypatch.setattr("requests.get", get)
    return model, response, get


def test_sglang_reads_loaded_runtime_and_keeps_engine_auth(sglang_runtime):
    model, _, get = sglang_runtime
    assert model.get_runtime_context_length() == 8192
    get.assert_called_once_with(
        "http://127.0.0.1:1234/v1/models",
        headers={"Authorization": "Bearer test"},
        timeout=2,
    )


def test_sglang_ignores_lora_cards(sglang_runtime):
    model, response, _ = sglang_runtime
    response.json.return_value["data"].insert(
        0,
        {
            "id": "lora-adapter",
            "object": "model",
            "created": 0,
            "owned_by": "sglang",
            "root": "/models/lora-adapter",
            "parent": "served-model-alias",
            "max_model_len": None,
        },
    )
    assert model.get_runtime_context_length() == 8192


def test_sglang_reads_updated_limit_after_reload(sglang_runtime):
    model, response, _ = sglang_runtime
    assert model.get_runtime_context_length() == 8192
    response.json.return_value["data"][0]["max_model_len"] = 4096
    assert model.get_runtime_context_length() == 4096


@pytest.mark.parametrize("value", [None, True, 0, -1, "8192"])
def test_sglang_unknown_runtime_does_not_use_launch_or_registry_limit(
    sglang_runtime, value
):
    model, response, _ = sglang_runtime
    response.json.return_value["data"][0]["max_model_len"] = value
    assert model.get_runtime_context_length() is None


def test_sglang_missing_model_card_does_not_use_launch_or_registry_limit(
    sglang_runtime,
):
    model, response, _ = sglang_runtime
    response.json.return_value["data"] = []
    assert model.get_runtime_context_length() is None


def test_sglang_not_loaded_does_not_query_runtime(sglang_runtime):
    model, _, get = sglang_runtime
    model._engine = None
    assert model.get_runtime_context_length() is None
    get.assert_not_called()


@pytest.mark.parametrize(
    "address,expected_url",
    [
        ("http://127.0.0.1:1234", "http://127.0.0.1:1234/native/props"),
        ("http://0.0.0.0:1234", "http://127.0.0.1:1234/native/props"),
        ("http://[::]:1234", "http://[::1]:1234/native/props"),
    ],
)
def test_llama_cpp_reads_actual_slot_capacity_without_dividing_again(
    monkeypatch, address, expected_url
):
    from ..llama_cpp.core import XllamaCppModel

    model = XllamaCppModel.__new__(XllamaCppModel)
    model._llm = SimpleNamespace(listening_address=address)
    model._llamacpp_model_config = {
        "n_ctx": 8193,
        "n_parallel": 2,
        "api_prefix": "/native",
        "api_keys": ["test"],
    }
    response = MagicMock()
    response.__enter__.return_value = response
    # The engine can pad total capacity; report the resolved per-slot value.
    response.json.return_value = {
        "default_generation_settings": {"n_ctx": 4352},
        "total_slots": 2,
    }
    get = MagicMock(return_value=response)
    monkeypatch.setattr("requests.get", get)
    assert model.get_runtime_context_length() == 4352
    get.assert_called_once_with(
        expected_url, headers={"Authorization": "Bearer test"}, timeout=2
    )


@pytest.mark.parametrize("explicit,expected", [(None, 8192), (4096, 4096)])
def test_multimodal_reads_loaded_nested_config(explicit, expected):
    model = SimpleNamespace(
        _context_length=None,
        _model=SimpleNamespace(
            config=SimpleNamespace(
                text_config=SimpleNamespace(max_position_embeddings=8192)
            )
        ),
        _pytorch_model_config={"context_length": explicit},
        model_family=SimpleNamespace(context_length=131072),
    )
    assert LLM.get_runtime_context_length(model) == expected
    model._model.config = SimpleNamespace()
    model._pytorch_model_config = {}
    assert LLM.get_runtime_context_length(model) is None


def test_trt_uses_resolved_engine_limit_and_adapter_cap():
    from ..trt_llm.core import TRTLLMModel

    model = TRTLLMModel.__new__(TRTLLMModel)
    model._context_length = 32768
    model._engine = SimpleNamespace(args=SimpleNamespace(max_seq_len=8192))
    assert model.get_runtime_context_length() == 8192
    model._context_length = 4096
    assert model.get_runtime_context_length() == 4096
