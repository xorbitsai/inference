# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest


@dataclass
class KVTransferConfig032:
    # Strict subset of vLLM 0.32's fields used by the adapter.
    kv_connector: str
    kv_connector_module_path: str
    engine_id: Optional[str]
    kv_role: str
    kv_connector_extra_config: dict
    kv_load_failure_policy: str


@dataclass
class LegacyKVTransferConfig(KVTransferConfig032):
    kv_rank: Optional[int] = None
    kv_parallel_size: int = 1


@pytest.fixture
def engine(monkeypatch):
    # Record the original state even when absent, before engine.py sets it.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "test")
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD")
    factory = Mock()
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(AsyncEngineArgs=object, __version__="0.21.0"),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        SimpleNamespace(KVTransferConfig=LegacyKVTransferConfig),
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
    "role,kv_role,kv_rank",
    [
        ("prefill", "kv_producer", 0),
        ("decode", "kv_consumer", 1),
        ("hybrid", "kv_both", 2),
    ],
)
@pytest.mark.parametrize("vllm_version", ["0.21.0", "0.31.0", "0.32.0"])
@pytest.mark.parametrize("overrides", [{}, {"kv_rank": 3, "kv_parallel_size": 4}])
def test_v1_connector_configuration(
    engine, monkeypatch, role, kv_role, kv_rank, vllm_version, overrides
):
    module, factory = engine
    module.VLLM_VERSION = vllm_version
    transfer_config = (
        KVTransferConfig032 if vllm_version == "0.32.0" else LegacyKVTransferConfig
    )
    monkeypatch.setattr(sys.modules["vllm.config"], "KVTransferConfig", transfer_config)
    args = SimpleNamespace(
        additional_config={"custom": 1},
        enforce_eager=False,
        create_model_config=lambda: SimpleNamespace(is_hybrid=False),
    )
    config = {"role": role, "rank": 2, "block_tracker_uid": b"tracker", **overrides}
    module.XavierEngine.from_engine_args(args, xavier_config=config)
    assert args.kv_transfer_config.kv_role == kv_role
    assert args.kv_transfer_config.kv_connector == "XavierConnector"
    assert args.kv_transfer_config.kv_load_failure_policy == "recompute"
    if vllm_version != "0.32.0":
        assert args.kv_transfer_config.kv_rank == overrides.get("kv_rank", kv_rank)
        assert args.kv_transfer_config.kv_parallel_size == overrides.get(
            "kv_parallel_size", 2
        )
    assert args.additional_config["custom"] == 1
    assert args.additional_config == {"custom": 1}
    assert (
        args.kv_transfer_config.kv_connector_extra_config["xavier_config"][
            "block_tracker_uid"
        ]
        == b"tracker"
    )
    json.dumps(args.additional_config)
    assert not args.enforce_eager
    assert getattr(args, "disable_hybrid_kv_cache_manager", None) is None
    assert args.enable_prefix_caching is True
    assert config == {
        "role": role,
        "rank": 2,
        "block_tracker_uid": b"tracker",
        **overrides,
    }
    factory.assert_called_once()


def test_transport_changes_preserve_user_compilation_config(engine):
    module, _ = engine
    additional = {"custom": {"graph_option": 1}}
    for rank in [1, 2]:
        args = SimpleNamespace(
            additional_config=additional,
            create_model_config=lambda: SimpleNamespace(is_hybrid=False),
        )
        module.XavierEngine._patch_v1_engine_args(
            args,
            {
                "rank": rank,
                "rank_address": f"localhost:{1000 + rank}",
                "block_tracker_uid": b"runtime-only",
            },
        )
        assert args.additional_config is additional
        assert (
            args.kv_transfer_config.kv_connector_extra_config["xavier_config"]["rank"]
            == rank
        )


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("launch_eager", [False, True])
@pytest.mark.parametrize("xavier_eager", [None, False, True])
def test_eager_defaults_and_explicit_settings(
    engine, recurrent, launch_eager, xavier_eager
):
    module, _ = engine
    args = SimpleNamespace(
        enforce_eager=launch_eager,
        create_model_config=lambda: SimpleNamespace(is_hybrid=recurrent),
    )
    config = {} if xavier_eager is None else {"enforce_eager": xavier_eager}
    module.XavierEngine.from_engine_args(args, xavier_config=config)
    expected = launch_eager or (recurrent if xavier_eager is None else xavier_eager)
    assert args.enforce_eager is expected


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


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_heterogeneous_engine_selects_cross_connector_and_fail_policy(engine, role):
    module, factory = engine
    args = SimpleNamespace(create_model_config=lambda: SimpleNamespace(is_hybrid=False))
    module.XavierEngine.from_engine_args(
        args,
        xavier_config={"role": role, "heterogeneous": True, "gpu_cache_bytes": 0},
    )
    transfer = args.kv_transfer_config
    assert transfer.kv_connector == "CrossEngineConnector"
    assert (
        transfer.kv_connector_module_path
        == "xinference.model.llm.vllm.xavier.cross_engine"
    )
    assert transfer.kv_load_failure_policy == "fail"
    assert transfer.kv_role == ("kv_producer" if role == "prefill" else "kv_consumer")
    factory.assert_called_once()


def test_incompatible_v1_fails_before_engine_start(engine):
    module, factory = engine
    module.VLLM_VERSION = "0.11.0"
    with pytest.raises(RuntimeError, match="0.21.0"):
        module.XavierEngine.from_engine_args(object())
    factory.assert_not_called()


@pytest.mark.parametrize("configured", [None, "fork"])
def test_v1_defaults_to_spawn_before_engine_creation(engine, monkeypatch, configured):
    module, factory = engine
    if configured is None:
        monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    else:
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", configured)

    def create(*args):
        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == (configured or "spawn")

    factory.side_effect = create
    module.XavierEngine.from_engine_args(
        SimpleNamespace(create_model_config=lambda: SimpleNamespace(is_hybrid=False))
    )
    factory.assert_called_once()


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


def test_gpu_path_rejects_legacy_before_engine_start(engine):
    module, factory = engine
    module.VLLM_VERSION = "0.7.3"
    with pytest.raises(RuntimeError, match="GPU transfer requires vLLM"):
        module.XavierEngine.from_engine_args(
            object(), xavier_config={"gpu_cache_bytes": 0}
        )
    factory.assert_not_called()


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("vllm_version", ["0.21.0", "0.22.0"])
@pytest.mark.parametrize("disable_hma", [None, True, False])
def test_hma_defaults_only_for_recurrent_models(
    engine, recurrent, vllm_version, disable_hma
):
    module, _ = engine
    module.VLLM_VERSION = vllm_version
    args = SimpleNamespace(
        create_model_config=lambda: SimpleNamespace(is_hybrid=recurrent),
        disable_hybrid_kv_cache_manager=disable_hma,
    )
    module.XavierEngine.from_engine_args(args)
    expected = (
        False
        if recurrent and vllm_version == "0.21.0" and disable_hma is None
        else disable_hma
    )
    assert args.disable_hybrid_kv_cache_manager is expected
    assert args.enable_prefix_caching is (not recurrent)
    if recurrent:
        assert args.async_scheduling is False


@pytest.mark.parametrize("setting", ["enable_prefix_caching", "async_scheduling"])
def test_recurrent_rejects_unsupported_settings_before_start(engine, setting):
    module, factory = engine
    args = SimpleNamespace(
        create_model_config=lambda: SimpleNamespace(is_hybrid=True),
        **{setting: True},
    )
    with pytest.raises(
        ValueError, match="enable_prefix_caching=False and async_scheduling=False"
    ):
        module.XavierEngine.from_engine_args(args)
    factory.assert_not_called()
