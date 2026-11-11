# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def adapter(monkeypatch):
    class Executor:
        def get_kv_cache_specs(self):
            return self.specs

    for name, values in {
        "vllm.v1.executor.abstract": {"Executor": Executor},
        "vllm.v1.executor.uniproc_executor": {
            "UniProcExecutor": type("UniProcExecutor", (Executor,), {})
        },
        "vllm.v1.executor.multiproc_executor": {
            "MultiprocExecutor": type("MultiprocExecutor", (Executor,), {})
        },
        "vllm.v1.kv_cache_interface": {
            "KVCacheSpec": SimpleNamespace,
            "FullAttentionSpec": type("Attention", (SimpleNamespace,), {}),
            "MambaSpec": type("Recurrent", (SimpleNamespace,), {}),
        },
    }.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**values))
    spec = importlib.util.spec_from_file_location(
        "xinference.model.llm.vllm.xavier._test_prefix_executor",
        Path(__file__).parents[1] / "prefix_executor.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("block_size", "alignment", "same_page", "manual", "expected"),
    [
        (1568, 1, True, None, 32),
        (84, 1, True, None, 28),
        (84, 3, True, None, 21),
        (1568, 1, False, None, None),
        (32, 1, True, None, 16),
        (83, 1, True, None, None),
        (1568, 1, True, 56, 56),
    ],
)
@pytest.mark.parametrize("backend", ["UniProc", "Multiproc"])
def test_executor_resolves_matching_unit_from_worker_specs(
    adapter, backend, block_size, alignment, same_page, manual, expected
):
    executor = getattr(adapter, f"XavierHybrid{backend}Executor")()
    executor.cache_config = SimpleNamespace(prefix_match_unit=manual)
    executor.specs = [
        {
            "attention": adapter.FullAttentionSpec(
                block_size=block_size,
                page_size_bytes=1024,
                prefix_cacheable=True,
                tokens_per_state=alignment,
            ),
            "state": adapter.MambaSpec(
                block_size=block_size,
                page_size_bytes=1024 if same_page else 2048,
                prefix_cacheable=True,
                tokens_per_state=-1,
                mamba_cache_mode="align",
            ),
        }
    ]
    assert executor.get_kv_cache_specs() is executor.specs
    assert executor.cache_config.prefix_match_unit == expected
