# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def connector_module(monkeypatch):
    """Exercise the real connector on CPU, stubbing only vLLM's import boundary."""

    class Base:
        def __init__(self, vllm_config, role, kv_cache_config):
            self._kv_cache_config = kv_cache_config
            self._kv_transfer_config = vllm_config.kv_transfer_config

    class Metadata:
        pass

    class HMA:
        pass

    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        SimpleNamespace(
            KVConnectorBase_V1=Base,
            KVConnectorMetadata=Metadata,
            KVConnectorRole=object,
            SupportsHMA=HMA,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.core.sched.output",
        SimpleNamespace(SchedulerOutput=object),
    )
    name = "xinference.model.llm.vllm.xavier._cpu_connector"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "v1_connector.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def connector_config():
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        lora_config=None,
        model_config=SimpleNamespace(is_multimodal_model=False),
        kv_transfer_config=SimpleNamespace(
            get_from_extra_config=lambda *a: {"rank": 1, "role": "prefill"},
            is_kv_producer=True,
            is_kv_consumer=True,
        ),
    )


@pytest.fixture
def connector(connector_module, connector_config):
    caches = SimpleNamespace(
        num_blocks=8,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["layer"],
                kv_cache_spec=SimpleNamespace(),
            )
        ],
    )
    instance = connector_module.XavierConnector(connector_config, None, caches)
    yield instance
    instance.shutdown()
