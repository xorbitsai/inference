# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def engine(monkeypatch):
    factory = Mock()
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(AsyncEngineArgs=object, __version__="0.21.0"),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.config", SimpleNamespace(KVTransferConfig=SimpleNamespace)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.engine.async_llm_engine",
        SimpleNamespace(AsyncLLMEngine=SimpleNamespace(from_engine_args=factory)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.usage.usage_lib",
        SimpleNamespace(UsageContext=SimpleNamespace(ENGINE_CONTEXT="engine")),
    )
    spec = importlib.util.spec_from_file_location(
        "xinference.model.llm.vllm.xavier._test_engine",
        Path(__file__).parents[1] / "engine.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, factory


@pytest.mark.parametrize(
    "role,kv_role",
    [("prefill", "kv_producer"), ("decode", "kv_consumer"), ("hybrid", "kv_both")],
)
def test_v1_connector_configuration(engine, role, kv_role):
    module, factory = engine
    args = SimpleNamespace(additional_config={"custom": 1}, enforce_eager=False)
    config = {"role": role, "rank": 2, "block_tracker_uid": b"tracker"}
    module.XavierEngine.from_engine_args(args, xavier_config=config)
    assert args.kv_transfer_config.kv_role == kv_role
    assert args.kv_transfer_config.kv_connector == "XavierConnector"
    assert args.additional_config["custom"] == 1
    json.dumps(args.additional_config)
    assert args.enforce_eager
    assert config == {"role": role, "rank": 2, "block_tracker_uid": b"tracker"}
    factory.assert_called_once()


def test_v0_uses_legacy_adapter(engine, monkeypatch):
    module, factory = engine
    module.VLLM_VERSION = "0.7.3"
    legacy = Mock()
    monkeypatch.setitem(
        sys.modules,
        "xinference.model.llm.vllm.xavier.legacy_engine",
        SimpleNamespace(XavierEngine=legacy),
    )
    module.XavierEngine.from_engine_args(object(), xavier_config={"rank": 1})
    legacy.from_engine_args.assert_called_once()
    factory.assert_not_called()


def test_incompatible_v1_fails_before_engine_start(engine):
    module, factory = engine
    module.VLLM_VERSION = "0.11.0"
    with pytest.raises(RuntimeError, match="0.21.0"):
        module.XavierEngine.from_engine_args(object())
    factory.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [("tensor_parallel_size", 2), ("pipeline_parallel_size", 2), ("enable_lora", True)],
)
def test_unsafe_parallelism_and_lora_fail_before_engine_creation(engine, field, value):
    module, factory = engine
    args = SimpleNamespace(**{field: value})
    with pytest.raises(ValueError):
        module.XavierEngine.from_engine_args(args, xavier_config={"role": "prefill"})
    factory.assert_not_called()
